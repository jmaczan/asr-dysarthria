# TODO:
# Dependencies in requirements.txt
# !pip install datasets>=1.18.3
# !pip install transformers==4.11.3
# !pip install torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# !pip install jiwer
# !pip install accelerate -U
# !pip install torchvision torchaudio pydub
## We need also !apt install git-lfs

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
import os
from datasets import load_metric
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
import torch
from dataclasses import dataclass
from typing import Dict, List, Union
import numpy as np

# TODO: make them argparse params
user_name = "jmaczan"
repo_name = "wav2vec2-large-xls-r-300m-dysarthria-big-dataset"
hf_full_name = f"{user_name}/{repo_name}"
data_path = directory_path = "/teamspace/uploads/uaspeechall/data"


def auth_into_hf():
    from huggingface_hub import HfApi, HfFolder

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    folder = HfFolder()
    folder.save_token(token)


def load_uaspeech_from_parquets(
    directory_path="/teamspace/uploads/uaspeechall/data", num_files=1, chunk_size=10000
):
    import pyarrow.parquet as pq
    from datasets import Dataset, concatenate_datasets
    import glob
    import pyarrow as pa

    def chunk_generator(table, chunk_size=10000):
        num_rows = table.num_rows
        for i in range(0, num_rows, chunk_size):
            yield table.slice(i, min(chunk_size, num_rows - i)).to_batches()[0]

    def process_parquet_files(directory_path, num_files, chunk_size=10000):
        parquet_files = glob.glob(f"{directory_path}/*.parquet")
        for file in parquet_files[
            :num_files
        ]:  # Only process the specified number of files
            print(f"Processing file: {file}")
            table = pq.read_table(file, memory_map=True)
            for chunk in chunk_generator(table, chunk_size):
                yield Dataset(pa.Table.from_batches([chunk]))

    all_datasets = []
    for chunk_dataset in process_parquet_files(directory_path, num_files, chunk_size):
        all_datasets.append(chunk_dataset)

    # Combine all chunks into a single dataset
    return concatenate_datasets(all_datasets)


def remove_special_characters(batch):
    import re

    chars_to_remove_regex = "[\,\?\.\!\-\;\:\"\“\%\‘\”\�']"

    batch["transcription"] = re.sub(
        chars_to_remove_regex, "", batch["transcription"]
    ).lower()
    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch["transcription"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def build_vocabulary(dataset):
    return dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=dataset.column_names,
    )


def build_vocab_json(train_dataset, test_dataset):
    vocab_train = build_vocabulary(train_dataset)
    vocab_test = build_vocabulary(test_dataset)

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    import json

    with open("vocab.json", "w") as vocab_file:
        json.dump(vocab_dict, vocab_file)


def prepare_dataset(batch, processor):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids
    return batch


def build_feature_extractor():
    return Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )


def build_tokenizer():
    return Wav2Vec2CTCTokenizer.from_pretrained(
        "./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )


def build_processor(tokenizer, feature_extractor):
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


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
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


def train_model(config):
    auth_into_hf()
    dataset = load_uaspeech_from_parquets(config["data_path"])
    dataset = dataset.train_test_split(test_size=0.2)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_dataset = train_dataset.map(remove_special_characters)
    test_dataset = test_dataset.map(remove_special_characters)

    build_vocab_json(train_dataset, test_dataset)

    tokenizer = build_tokenizer()
    tokenizer.push_to_hub(repo_name)

    feature_extractor = build_feature_extractor()

    processor = build_processor(tokenizer, feature_extractor)

    train_dataset = train_dataset.map(
        lambda batch: prepare_dataset(batch=batch, processor=processor),
        remove_columns=train_dataset.column_names,
    )
    test_dataset = test_dataset.map(
        lambda batch: prepare_dataset(batch=batch, processor=processor),
        remove_columns=test_dataset.column_names,
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    wer_metric = load_metric("wer")

    model = Wav2Vec2ForCTC.from_pretrained(
        config["model_name"],
        attention_dropout=config["attention_dropout"],
        hidden_dropout=config["hidden_dropout"],
        feat_proj_dropout=config["feat_proj_dropout"],
        mask_time_prob=config["mask_time_prob"],
        layerdrop=config["layerdrop"],
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    model.freeze_feature_extractor()  # TODO: perhaps not always we want to do it

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        group_by_length=True,
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=config["num_epochs"],
        gradient_checkpointing=True,
        fp16=True,
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        logging_steps=config["logging_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        save_total_limit=5,
        push_to_hub=config["push_to_hub"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
    )

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    if config["push_to_hub"]:
        trainer.push_to_hub()

    return trainer.evaluate()


if __name__ == "__main__":
    config = {
        "data_path": "/teamspace/uploads/uaspeechall/data",
        "model_name": "facebook/wav2vec2-large-xls-r-300m",
        "output_dir": "wav2vec2-large-xls-r-300m-dysarthria-big-dataset",
        "batch_size": 16,
        "num_epochs": 30,
        "save_steps": 200,
        "eval_steps": 200,
        "logging_steps": 200,
        "learning_rate": 3e-4,
        "warmup_steps": 500,
        "push_to_hub": True,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "feat_proj_dropout": 0.0,
        "mask_time_prob": 0.05,
        "layerdrop": 0.0,
    }

    train_model(config)
