import os
from typing import Optional
import torch

import fire
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Trainer, TrainingArguments

from hypencoder_cb.modeling.hypencoder import (
    HypencoderDualEncoder,
    HypencoderDualEncoderConfig,
    TextDualEncoder,
)
from hypencoder_cb.modeling.shared import BaseDualEncoderConfig
from hypencoder_cb.train.args import (
    HypencoderDataConfig,
    HypencoderModelConfig,
    HypencoderTrainerConfig,
    HypencoderTrainingConfig,
)
from hypencoder_cb.train.data_collator import GeneralDualEncoderCollator

DEFAULT_CACHE_DIR = os.environ.get(
    "HYPENCODER_CACHE_DIR", os.path.expanduser("~/.cache/hypencoder")
)


def load_model(model_config: HypencoderModelConfig):
    config_cls_lookup = {
        "hypencoder": HypencoderDualEncoderConfig,
        "biencoder": BaseDualEncoderConfig,
    }

    model_cls_lookup = {
        "hypencoder": HypencoderDualEncoder,
        "biencoder": TextDualEncoder,
    }

    config_cls = config_cls_lookup[model_config.model_type]
    model_cls = model_cls_lookup[model_config.model_type]

    if model_config.checkpoint_path is not None:
        # Load the checkpoint with its original config to preserve hyperhead weights
        print(f"Loading checkpoint from {model_config.checkpoint_path}")
        model = model_cls.from_pretrained(model_config.checkpoint_path)
        
        # Replace transformer with TAS-B if model_name_or_path differs from checkpoint
        from transformers import AutoModel
        new_query_model_path = model_config.query_encoder_kwargs.get('model_name_or_path')
        new_passage_model_path = model_config.passage_encoder_kwargs.get('model_name_or_path')
        
        if new_query_model_path and new_query_model_path != model.query_encoder.transformer.config._name_or_path:
            print(f"Replacing query transformer: {model.query_encoder.transformer.config._name_or_path} -> {new_query_model_path}")
            model.query_encoder.transformer = AutoModel.from_pretrained(new_query_model_path)
            
        if new_passage_model_path and new_passage_model_path != model.passage_encoder.transformer.config._name_or_path:
            print(f"Replacing passage transformer: {model.passage_encoder.transformer.config._name_or_path} -> {new_passage_model_path}")
            model.passage_encoder.transformer = AutoModel.from_pretrained(new_passage_model_path)
            
        # If shared encoder, make sure they point to the same transformer
        if model_config.shared_encoder:
            model.passage_encoder.transformer = model.query_encoder.transformer
            
        # Update config with new settings (loss, freezing, etc.) but keep the loaded hyperhead
        # Convert OmegaConf objects to plain Python objects for JSON serialization
        model.config.loss_type = OmegaConf.to_container(model_config.loss_type, resolve=True)
        model.config.loss_kwargs = OmegaConf.to_container(model_config.loss_kwargs, resolve=True)
        model._get_similarity_loss(model.config)
        # Reinitialize the forward kwargs list to match the new loss functions
        model.similarity_loss_forward_kwargs = [
            {} for _ in range(len(model.similarity_losses))
        ]
        
        # Update freezing settings
        freeze_transformer = model_config.query_encoder_kwargs.get('freeze_transformer', False)
        for param in model.query_encoder.transformer.parameters():
            param.requires_grad = not freeze_transformer
        if not model_config.shared_encoder:
            freeze_passage = model_config.passage_encoder_kwargs.get('freeze_transformer', False)
            for param in model.passage_encoder.transformer.parameters():
                param.requires_grad = not freeze_passage
    else:
        # Create new model from scratch
        config = config_cls(
            query_encoder_kwargs=OmegaConf.to_container(
                model_config.query_encoder_kwargs
            ),
            passage_encoder_kwargs=OmegaConf.to_container(
                model_config.passage_encoder_kwargs
            ),
            loss_type=OmegaConf.to_container(model_config.loss_type),
            loss_kwargs=OmegaConf.to_container(model_config.loss_kwargs),
            shared_encoder=model_config.shared_encoder,
        )
        model = model_cls(config)

    return model


def load_data(data_config: HypencoderDataConfig):
    cache_dir = os.environ.get("HF_HOME", DEFAULT_CACHE_DIR)

    if (data_config.training_data_jsonl is None) == (
        data_config.training_huggingface_dataset is None
    ):
        raise ValueError(
            "Must specify either training_data_jsonl or"
            " training_huggingface_dataset"
        )

    if (
        data_config.validation_data_jsonl is not None
        and data_config.validation_huggingface_dataset is not None
    ):
        raise ValueError(
            "Cannot specify both validation_data_jsonl and"
            " validation_huggingface_dataset"
        )

    if data_config.training_huggingface_dataset is not None:
        training_data = load_dataset(
            data_config.training_huggingface_dataset,
            split=data_config.training_data_split,
            cache_dir=cache_dir,
        )
    elif data_config.training_data_jsonl is not None:
        training_data = load_dataset(
            "json",
            data_files=data_config.training_data_jsonl,
            split=data_config.training_data_split,
            cache_dir=cache_dir,
        )

    validation_data = None
    if data_config.validation_huggingface_dataset is not None:
        training_data = load_dataset(
            data_config.validation_huggingface_dataset,
            split=data_config.validation_data_split,
            cache_dir=cache_dir,
        )
    elif data_config.validation_data_jsonl is not None:
        training_data = load_dataset(
            "json",
            data_files=data_config.validation_data_jsonl,
            split=data_config.validation_data_split,
            cache_dir=cache_dir,
        )

    return training_data, validation_data


def get_collator(
    data_config: HypencoderDataConfig,
    trainer_config: HypencoderTrainerConfig,
    tokenizer,
):
    return GeneralDualEncoderCollator(
        tokenizer=tokenizer,
        num_negatives_to_sample=data_config.num_negatives_to_sample,
        positive_filter=data_config.positive_filter_type,
        positive_filter_kwargs=data_config.positive_filter_kwargs,
        positive_sampler="random",
        negative_sampler="random",
        num_positives_to_sample=data_config.num_positives_to_sample,
        label_key=data_config.label_key,
        query_padding_mode="longest",
    )


def load_tokenizer(model_config: HypencoderModelConfig):
    return AutoTokenizer.from_pretrained(
        model_config.tokenizer_pretrained_model_name_or_path
    )


def train_model(cfg: HypencoderTrainingConfig):
    print(cfg)
    resume_from_checkpoint = cfg.trainer_config.resume_from_checkpoint

    training_data, validation_data = load_data(cfg.data_config)
    tokenizer = load_tokenizer(cfg.model_config)
    model = load_model(cfg.model_config)
    collator = get_collator(cfg.data_config, cfg.trainer_config, tokenizer)

    train_arguments_kwargs = None
    hf_trainer_config = cfg.trainer_config.hf_trainer_config

    if OmegaConf.is_config(hf_trainer_config):
        train_arguments_kwargs = OmegaConf.to_container(hf_trainer_config)
    else:
        train_arguments_kwargs = hf_trainer_config.__dict__

    training_args = TrainingArguments(
        **train_arguments_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=validation_data,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("Starting training")
    if resume_from_checkpoint is True:
        if not os.path.exists(training_args.output_dir) or not any(
            [
                "checkpoint" in name
                for name in os.listdir(training_args.output_dir)
            ]
        ):
            resume_from_checkpoint = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def run_training(config_path: Optional[str] = None) -> None:
    schema = OmegaConf.structured(HypencoderTrainingConfig)

    if config_path is not None:
        config = OmegaConf.load(config_path)
        config = OmegaConf.merge(schema, config)
    else:
        config = schema

    train_model(config)


if __name__ == "__main__":
    fire.Fire(run_training)