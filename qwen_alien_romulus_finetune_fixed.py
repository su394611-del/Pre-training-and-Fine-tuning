"""Fine-tune Qwen2.5-0.5B-Instruct on Alien: Romulus QA examples.

Place data/alien_romulus_qa.jsonl next to this script. Each line must be
JSON with an 'instruction' question and an 'output' answer.

LoRA is enabled by default to reduce GPU memory use. Set USE_LORA=False
for full-parameter fine-tuning on hardware with adequate memory.
"""

from pathlib import Path

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


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "data" / "alien_romulus_qa.jsonl"
TUNED_MODEL_DIR = SCRIPT_DIR / "qwen_alien_romulus_finetuned"

USE_CUDA = torch.cuda.is_available()
USE_LORA = True     # Recommended for a GPU with 4 GB VRAM
MAX_LENGTH = 256


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_PATH}\n"
            "Create the 'data' folder and put alien_romulus_qa.jsonl there."
        )

    print("CUDA available:", USE_CUDA)
    if USE_CUDA:
        print("GPU:", torch.cuda.get_device_name(0))

    # Load the JSONL dataset (one JSON object per line).
    dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")
    if len(dataset) == 0:
        raise ValueError("The training dataset is empty.")

    required = {"instruction", "output"}
    if not required.issubset(dataset.column_names):
        raise ValueError(
            f"Expected columns {sorted(required)}; found {dataset.column_names}."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer lacks both pad and EOS tokens.")
        tokenizer.pad_token = tokenizer.eos_token

    # Use Qwen's conversation format rather than ad-hoc 'Question/Answer' tags.
    def format_sample(sample):
        messages = [
            {"role": "user", "content": str(sample["instruction"])},
            {"role": "assistant", "content": str(sample["output"])},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    dataset = dataset.map(lambda sample: {"text": format_sample(sample)})

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,            # Let the collator pad each batch
            add_special_tokens=False, # Chat template already adds them
        )

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Load weights in float16 on NVIDIA GPU to conserve VRAM.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if USE_CUDA else torch.float32,
    )
    model.config.use_cache = False

    # No new vocabulary tokens were added, so no embedding resize is needed.
    if USE_LORA:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        if USE_CUDA:
            # Ensure gradient checkpointing and PEFT work together.
            model.enable_input_require_grads()

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=str(TUNED_MODEL_DIR),
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4 if USE_LORA else 2e-5,
        weight_decay=0.01,
        warmup_steps=0.03,  # In Transformers 5.x, float means fraction of total steps
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        use_cpu=not USE_CUDA,
        fp16=USE_CUDA,
        bf16=False,
        gradient_checkpointing=USE_CUDA,
        dataloader_num_workers=0,  # Safer for Windows
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )

    result = trainer.train()
    trainer.save_model(str(TUNED_MODEL_DIR))
    tokenizer.save_pretrained(str(TUNED_MODEL_DIR))
    trainer.save_metrics("train", result.metrics)
    print(f"Training complete. Files saved to: {TUNED_MODEL_DIR}")
    if USE_LORA:
        print("The folder contains LoRA adapter weights, not a standalone full model.")


if __name__ == "__main__":
    main()
