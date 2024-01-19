import torch
import torchaudio
import re
import json

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import ClassLabel, load_dataset, load_metric

import random
import pandas as pd

# only for notebooks:
# from IPython.display import display, HTML
# from huggingface_hub import notebook_login
# notebook_login()

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2ForCTC,
    TrainingArguments,
    AutoModelForCTC,
    Wav2Vec2Processor,
    Trainer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print("Torch device: ", mps_device)
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")
    raise Exception("MPS device not found.")


dataset_name = "jmaczan/TORGO-very-small"
model_name = "jmaczan/asr-dysarthria-wav2vec2-v0"
processor_name = model_name


dataset = load_dataset(dataset_name, split="train")

print(dataset)

dataset = dataset.train_test_split(test_size=0.3)

"""clean up data"""

chars_to_ignore_regex = '[\,\?\.\!\-\;\:"]'


def remove_special_characters(batch):
    batch["transcription"] = re.sub(
        chars_to_ignore_regex, "", batch["transcription"]
    ).lower()
    return batch


dataset = dataset.map(remove_special_characters)


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(
        dataset
    ), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    # notebook only: display(HTML(df.to_html()))


show_random_elements(dataset["train"].remove_columns(["audio"]))


def extract_all_chars(batch):
    all_text = " ".join(batch["transcription"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


vocabs = dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset.column_names["train"],
)

vocab_list = list(set(vocabs["train"]["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
print(len(vocab_dict))

with open("vocab.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file)


tokenizer = Wav2Vec2CTCTokenizer(
    "./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
)

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=False,
)


processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

print(dataset["train"][0]["audio"]["path"])

dataset["train"][0]["audio"]

import IPython.display as ipd
import numpy as np
import random

rand_int = random.randint(0, len(dataset["train"]))

print(dataset["train"][rand_int]["transcription"])
ipd.Audio(data=np.asarray(dataset["train"][rand_int]["audio"]["array"]), rate=16000)

rand_int = random.randint(0, len(dataset["train"]))

print("Target text:", dataset["train"][rand_int]["transcription"])
print(
    "Input array shape:", np.asarray(dataset["train"][rand_int]["audio"]["array"]).shape
)
print("Sampling rate:", dataset["train"][rand_int]["audio"]["sampling_rate"])


def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids
    return batch


dataset = dataset.map(
    prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4
)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

wer_metric = load_metric("wer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

model.freeze_feature_extractor()


training_args = TrainingArguments(
    output_dir="training_arguments",
    group_by_length=True,
    per_device_train_batch_size=32,
    evaluation_strategy="steps",
    num_train_epochs=1,
    fp16=True,
    gradient_checkpointing=True,
    save_steps=50,
    eval_steps=50,
    logging_steps=50,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=10,
    save_total_limit=10,
)


trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor,
)

trainer.train()

processor = Wav2Vec2Processor.from_pretrained(processor_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
