import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
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
from training.hparams_search_config import config
from metric import asr_metric


def objective(trial):
    try:

        auth_into_hf()
        dataset = load_uaspeech_from_parquets(
            config["data_path"], num_files=config["hparam_search_num_files"]
        )
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

        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
        attention_dropout = trial.suggest_uniform("attention_dropout", 0.0, 0.5)
        hidden_dropout = trial.suggest_uniform("hidden_dropout", 0.0, 0.5)
        feat_proj_dropout = trial.suggest_uniform("feat_proj_dropout", 0.0, 0.5)
        mask_time_prob = trial.suggest_uniform("mask_time_prob", 0.0, 0.1)
        layerdrop = trial.suggest_uniform("layerdrop", 0.0, 0.5)
        warmup_steps = trial.suggest_int("warmup_steps", 100, 1000)

        model = Wav2Vec2ForCTC.from_pretrained(
            config["model_name"],
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
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            save_total_limit=2,
            push_to_hub=config["push_to_hub"],
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

        eval_result = trainer.evaluate()

        result = eval_result[asr_metric]
        del model, trainer
        torch.cuda.empty_cache()

        return result
    except Exception as e:
        print(e)
        return float("inf")


def run_optimization():
    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=200, interval_steps=20),
        storage="sqlite:///optuna.db",
        study_name="wav2vec2_optimization",
        load_if_exists=True,
        sampler=TPESampler(seed=10),
    )
    study.optimize(objective, n_trials=20, timeout=3600)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    study.trials_dataframe().to_csv("optimization_results.csv")


if __name__ == "__main__":
    run_optimization()
