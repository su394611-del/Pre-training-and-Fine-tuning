"""Qwen2.5-0.5B-Instruct LoRA fine-tuning on Alien: Romulus question/answer examples.

The JSONL dataset must contain one JSON object per line,
with nonempty "instruction" and "output" string fields.

Examples:
    python qwen_alien_romulus_finetune_fixed.py --validate-data
    python qwen_alien_romulus_finetune_fixed.py
    python qwen_alien_romulus_finetune_fixed.py --epochs 1 --max-length 128

A CUDA GPU is optional but strongly recommended. A 4 GB GPU can still run out of memory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATA = BASE_DIR / "data" / "alien_romulus_qa.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "qwen_alien_romulus_finetuned"


def validate_jsonl(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    count = 0
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL on line {line_number}: {exc}") from exc
            if not isinstance(record, dict) or not all(
                isinstance(record.get(key), str) and record[key].strip()
                for key in ("instruction", "output")
            ):
                raise ValueError(f"Line {line_number}: expected nonempty instruction/output strings")
            count += 1
    if not count:
        raise ValueError("Dataset has no examples")
    print(f"Validated {count} Q&A examples in {path}")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--validate-data", action="store_true", help="Check JSONL without loading ML packages")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-lora", action="store_true", help="Full fine-tuning; needs substantially more memory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (very slow)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.data = args.data.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    validate_jsonl(args.data)
    if args.validate_data:
        return
    if args.epochs <= 0 or args.max_length < 8 or args.batch_size < 1:
        raise ValueError("epochs must be > 0; max-length >= 8; batch-size >= 1")

    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, TaskType, get_peft_model

    use_cuda = bool(torch.cuda.is_available() and not args.cpu)
    use_lora = not args.no_lora
    print(f"Device: {'cuda' if use_cuda else 'cpu'} | model: {args.model} | LoRA: {use_lora}")
    if use_cuda:
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = load_dataset("json", data_files=str(args.data), split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must contain a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    def format_sample(example):
        return tokenizer.apply_chat_template(
            [
                {"role": "user", "content": str(example["instruction"])},
                {"role": "assistant", "content": str(example["output"])},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    dataset = dataset.map(lambda example: {"text": format_sample(example)})

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
            add_special_tokens=False,
        )

    train_data = dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16 if use_cuda else torch.float32,
    )
    model.config.use_cache = False
    if use_lora:
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
            lora_dropout=0.05, bias="none", target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
        if use_cuda:
            model.enable_input_require_grads()

    # Transformers 5.x uses fractional warmup_steps instead of warmup_ratio.
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4 if use_lora else 2e-5,
        weight_decay=0.01,
        warmup_steps=0.03,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        use_cpu=not use_cuda,
        fp16=use_cuda,
        bf16=False,
        gradient_checkpointing=use_cuda,
        dataloader_num_workers=0,
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    result = trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    trainer.save_metrics("train", result.metrics)
    print("Model outputs saved at:", args.output)
    if use_lora:
        print("LoRA weights are adapters; load with the original base model.")


if __name__ == "__main__":
    main()
