# LLM Fine-Tuning Examples: Qwen LoRA, SFT, DPO and ORPO

Two  PyTorch / Hugging Face LLM post-training workflows, accompanied by standalone Python programs, interactive Jupyter notebooks and example data:

1. **Qwen2.5-0.5B-Instruct + LoRA**: fine-tune a causal language model on 49 *Alien: Romulus* instruction/answer examples.
2. **SFT → DPO or ORPO**: demonstrate instruction fine-tuning and preference optimization using a separate, explicitly version-pinned TRL 0.8.6 stack. The default model is `HuggingFaceTB/SmolLM-135M`.

> **:** The examples use very small datasets. Training loss or example generations cannot demonstrate the model's general quality. The project does **not** implement PPO, even though it is preference optimization.

## Project contents

```text
.
├── README.md
├── requirements.txt                       # Modern Qwen / Transformers 5.x environment
├── requirements_legacy_llm_v2.txt         # Separate, legacy SFT/DPO/ORPO environment
├── qwen_alien_romulus_finetune.py         # Qwen training + JSONL validation
├── qwen_alien_romulus_finetune.ipynb      # Equivalent interactive Qwen implementation
├── LLM_Training_Fixed_v2.py              # User-supplied legacy SFT/DPO/ORPO CLI
├── LLM_Training_Fixed_v2.ipynb           # Equivalent interactive legacy implementation
├── data/
│   └── alien_romulus_qa.jsonl            # 49 valid JSONL lines
├── original_sources/
│   ├── qwen_alien_romulus_finetune_original.py
│   └── alien_romulus_qa_pretty_not_jsonl.txt
└── .gitignore
```

The source files are preserved in `original_sources/` for reference. The top-level Qwen script has small usability updates: a standalone data-validation mode, adjustable training arguments, and Transformers 5. x-compatible `warmup_steps=0.03` in place of the older `warmup_ratio` field. The original uploaded dataset was a sequence of **multi-line JSON objects**, not actual JSONL; the tracked `data/alien_romulus_qa.jsonl` is normalized to **one object per line**.

The additional uploaded `data.zip` includes other datasets/corpora not consumed by either training program. It is intentionally not republished in this minimal GitHub project; add other data only after checking its licensing and provenance.

## Requirements and supported environments

- **Python:** 3.12 recommended, especially on Windows 11.
- **CPU:** Data validation works without ML libraries. Training on CPU is possible but potentially very slow.
- **GPU:** An NVIDIA CUDA GPU substantially accelerates training. A 4 GB GTX 1650-class GPU may still run out of memory with Qwen LoRA; reduce sequence length or use Colab/cloud GPU if necessary.
- **Network:** First training run downloads the base model and tokenizer from Hugging Face.
- **Separate environments are mandatory.** `requirements.txt` uses Transformers 5.x; `requirements_legacy_llm_v2.txt` pins Transformers 4.40.0 and TRL 0.8.6. **Do not combine the two dependency sets.**

If PyTorch reports `+cpu` or `torch.cuda.is_available() == False`, use an appropriate CUDA-enabled PyTorch wheel from the [official PyTorch installer](https://pytorch.org/get-started/locally/) in the selected environment. `nvidia-smi` showing a GPU does not mean the installed PyTorch has CUDA support.

## Workflow A — Qwen LoRA (modern environment)

### Windows PowerShell

```powershell
py -3.12 -m venv .venv-qwen
.\.venv-qwen\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check

# Fast validation; this does not download model weights
python qwen_alien_romulus_finetune.py --validate-data

# Full training run (5 epochs by default)
python qwen_alien_romulus_finetune.py

# Smaller resource-saving run
python qwen_alien_romulus_finetune.py --epochs 1 --max-length 128 --batch-size 1
```

If PowerShell blocks activation, use the environment's Python directly: ` .\.venv-qwen\Scripts\python.exe qwen_alien_romulus_finetune.py --validate-data`.

### Dataset format

`data/alien_romulus_qa.jsonl` contains exactly one JSON object per line:

```json
{"instruction": "What is Alien: Romulus about?","output":"Alien: Romulus is a science fiction horror film in the Alien franchise. A group of young colonists explore a derelict space station and encounter the deadly Xenomorph species."}
```

Both fields are required, and their values must be nonempty strings. The validator identifies malformed JSON and invalid records with line numbers.

### Qwen training configuration

| Setting | Default |
| --- | --- |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Adaptation | LoRA on `q_proj` and `v_proj` |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Sequence length | 256 tokens |
| Epochs | 5 |
| Per-device batch size | 1 |
| Gradient accumulation | 4 |
| Learning rate (LoRA) | `2e-4` |
| Mixed precision | FP16 on CUDA, FP32 on CPU |
| Output folder | `qwen_alien_romulus_finetuned/` |

**Important:** The Qwen example uses `DataCollatorForLanguageModeling` and computes causal language-model loss over both prompt and assistant tokens. It is a LoRA fine-tuning example, **not** a completion-only-masked SFT pipeline. With LoRA enabled, the saved folder contains adapter weights, which require the base Qwen checkpoint for inference. The example does not evaluate on a held-out validation set.

## Workflow B — Legacy SFT → DPO or ORPO

**Create an independent environment.** The `LLM_Training_Fixed_v2.py` script requires exact versions and exits if they differ.

```powershell
# From the project root; first deactivate any active environment
deactivate
py -3.12 -m venv .venv-legacy
.\.venv-legacy\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements_legacy_llm_v2.txt
python -m pip check

# Validate bundled examples: does not download models
python LLM_Training_Fixed_v2.py --mode validate

# Check exact dependency versions and trainer imports
python LLM_Training_Fixed_v2.py --mode check

# Option A: supervised fine-tuning followed by preference optimization
python LLM_Training_Fixed_v2.py --mode sft-dpo

# Option B: separate, single-stage ORPO experiment
python LLM_Training_Fixed_v2.py --mode orpo
```

This second workflow embeds **6 SFT examples and 5 preference pairs** directly in the Python source. It uses a textual `Instruct: ... Output:` template, a completion-only SFT collator, LoRA adapters, and a frozen SFT adapter as the DPO reference. ORPO must be run as a **separate experiment** rather than after SFT → DPO.

The optional `microsoft/phi-2` path is substantially more memory-intensive. A compatible 4-bit quantization backend and significantly more GPU RAM are required; see `--help` and the script's hardware checks. Install `bitsandbytes` separately **only** if your chosen CUDA/GPU/platform supports it. With the tiny demonstration datasets, neither DPO nor ORPO outputs represent reliable alignment benchmarks.

## Running the Jupyter notebooks

Both `.ipynb` files contain the corresponding Python logic directly rather than relying only on `%run` wrappers. They deliberately **do not start model training by default**; each has an opt-in `RUN_TRAINING` flag.

1. In VS Code or JupyterLab, open `qwen_alien_romulus_finetune.ipynb` and choose the **`.venv-qwen`** kernel. It validates the JSONL dataset first. Set `RUN_TRAINING = True` in the training cell to download Qwen and begin training.
2. Open `LLM_Training_Fixed_v2.ipynb` with the **`.venv-legacy`** kernel. Run the dataset-validation cells first. Set `RUN_TRAINING = True` in its final cell and choose `TRAINING_MODE = "sft-dpo"` or `"orpo"` to train.

To register environments as notebook kernels:

```powershell
python -m ipykernel install --user --name qwen-lora --display-name "Python (Qwen LoRA)"
```

Run that command in the Qwen environment, then repeat in the legacy environment with `--name legacy-trl --display-name "Python (Legacy TRL)"`.

**Google Colab:** The Qwen notebook is the better choice for a recent hosted GPU runtime. Copy the project files into your Colab working directory, install `requirements.txt,` and select a GPU runtime. The legacy TRL 0.8.6 stack may require an older compatible runtime and is not guaranteed to run against Colab's default preinstalled packages.

## Troubleshooting

- **`JSONDecodeError` or `ArrowInvalid`:** Make sure you use the corrected `data/alien_romulus_qa.jsonl`, not `original_sources/alien_romulus_qa_pretty_not_jsonl.txt`.
- **`warmup_ratio` unexpected keyword:** Use the corrected Qwen script and Transformers 5.x requirements; it specifies fractional `warmup_steps` instead.
- **`trl`/`peft` import failures:** Avoid mixing the legacy stack and the Qwen stack in the same virtual environment.
- **CUDA unavailable:** Check `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` and install a platform-appropriate CUDA-enabled PyTorch build if supported.
- **CUDA out of memory:** Try `--max-length 128 --batch-size 1`, close GPU-intensive applications, or use a larger GPU.
- **`ipykernel_launcher.py` argparse errors:** Run cells from the notebooks; don't feed Jupyter's own kernel arguments into the CLI scripts.
- **PowerShell says `from` is unsupported:** Python statements go in a Python interpreter (`python`), notebook, or `.py` file, not directly at a `PS>` prompt.

## Quality and validation status

- Python syntax checks: completed for both top-level programs.
- Data normalization and schema validation: passed for 49 Alien: Romulus records.
- Legacy script's `--mode validate`: passed for the embedded 6 SFT and 5 preference examples.
- Jupyter notebook structure and Python compilation: checked before packaging.
- **Full model training and GPU execution: not run in the packaging environment.**

## Further reading

- [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [PEFT / LoRA](https://huggingface.co/docs/peft/)
- [TRL documentation](https://huggingface.co/docs/trl/)
- [PyTorch installation](https://pytorch.org/get-started/locally/)

## Data, model licenses and publication

The included movie-related question/answer examples are illustrative, and factual statements have not been independently fact-checked. Review rights and licenses for any training data before redistribution. Hugging Face pretrained models have their own licenses; see each model card. This repository does not grant rights to third-party model weights or movie-related intellectual property.

This project is licensed under the Apache License, Version 2.0.© 2026 Sudhanshu Shukla · Licensed under the Apache License, Version 2.0
