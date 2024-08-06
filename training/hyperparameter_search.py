import optuna
from transformers import Wav2Vec2ForCTC, TrainingArguments
from train import (
    auth_into_hf,
    load_uaspeech_from_parquets,
    remove_special_characters,
    build_vocab_json,
    build_tokenizer,
    build_feature_extractor,
    build_processor,
    prepare_dataset,
    DataCollatorCTCWithPadding,
    compute_metrics,
    Trainer,
)
from datasets import load_metric


def objective(trial):
    # Authentication and data loading
    auth_into_hf()
    dataset = load_uaspeech_from_parquets()
    dataset = dataset.train_test_split(test_size=0.2)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_dataset = train_dataset.map(remove_special_characters)
    test_dataset = test_dataset.map(remove_special_characters)

    build_vocab_json(train_dataset, test_dataset)

    tokenizer = build_tokenizer()
    feature_extractor = build_feature_extractor()
    processor = build_processor(tokenizer, feature_extractor)

    train_dataset = train_dataset.map(
        prepare_dataset, remove_columns=train_dataset.column_names
    )
    test_dataset = test_dataset.map(
        prepare_dataset, remove_columns=test_dataset.column_names
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    wer_metric = load_metric("wer")

    # Hyperparameters to optimize
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    attention_dropout = trial.suggest_uniform("attention_dropout", 0.0, 0.5)
    hidden_dropout = trial.suggest_uniform("hidden_dropout", 0.0, 0.5)
    feat_proj_dropout = trial.suggest_uniform("feat_proj_dropout", 0.0, 0.5)
    mask_time_prob = trial.suggest_uniform("mask_time_prob", 0.0, 0.5)
    layerdrop = trial.suggest_uniform("layerdrop", 0.0, 0.5)
    warmup_steps = trial.suggest_int("warmup_steps", 100, 1000)

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-xls-r-300m",
        attention_dropout=attention_dropout,
        hidden_dropout=hidden_dropout,
        feat_proj_dropout=feat_proj_dropout,
        mask_time_prob=mask_time_prob,
        layerdrop=layerdrop,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    model.freeze_feature_extractor()

    training_args = TrainingArguments(
        output_dir="./results",
        group_by_length=True,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=5,  # Reduced for faster trials
        gradient_checkpointing=True,
        fp16=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        save_total_limit=2,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
    )

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

    # Evaluate the model
    eval_result = trainer.evaluate()

    return eval_result["eval_wer"]


def run_optimization():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  # Adjust the number of trials as needed

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    run_optimization()
