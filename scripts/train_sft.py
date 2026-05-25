import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_bnb_config(model_cfg):
    if model_cfg.get("load_in_8bit", False):
        return BitsAndBytesConfig(load_in_8bit=True)

    if model_cfg.get("load_in_4bit", False):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16
            if model_cfg.get("bnb_4bit_compute_dtype", "bfloat16") == "bfloat16"
            else torch.float16,
        )

    return None


def build_lora_config(model_cfg):
    lora_cfg = model_cfg["lora"]
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )


def build_sft_config(train_cfg):
    allowed_keys = {
        "output_dir",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "optim",
        "save_steps",
        "logging_steps",
        "learning_rate",
        "num_train_epochs",
        "max_steps",
        "max_grad_norm",
        "lr_scheduler_type",
        "weight_decay",
        "warmup_ratio",
        "fp16",
        "bf16",
        "packing",
        "max_length",
        "assistant_only_loss",
        "dataset_text_field",
        "report_to",
        "seed",
        "save_total_limit",
    }
    kwargs = {k: v for k, v in train_cfg.items() if k in allowed_keys and v is not None}
    return SFTConfig(**kwargs)


def inspect_first_batch(trainer, tokenizer):
    batch = next(iter(trainer.get_train_dataloader()))
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    print("Loss target text:")
    print(tokenizer.decode(input_ids[labels != -100]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", default="configs/train/sft_lora.yaml")
    parser.add_argument("--model-config", default="configs/model/qwen2_5_1_5b_lora.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_cfg = load_yaml(args.train_config)
    model_cfg = load_yaml(args.model_config)

    model_path = train_cfg["model_name_or_path"]
    dataset_path = train_cfg["dataset_path"]
    output_dir = train_cfg["output_dir"]

    #加载 tokenizer
    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token

    #加载 processed dataset
    print(f"Loading SFT dataset from {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(dataset)
    print(dataset[0])

    #加载 base model
    print("Loading base model")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=build_bnb_config(model_cfg),
        device_map="auto",
        trust_remote_code=True,
    )

    if model_cfg.get("load_in_8bit", False) or model_cfg.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)

    lora_config = build_lora_config(model_cfg)
    training_args = build_sft_config(train_cfg)

    #挂 LoRA
    #初始化 SFTTrainer
    print("Initializing SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.model.print_trainable_parameters()
    inspect_first_batch(trainer, tokenizer)

    if args.dry_run:
        print("Dry run finished.")
        return

    #正式训练
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("Starting training")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training finished. Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
