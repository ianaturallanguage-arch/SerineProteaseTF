# ProGen2 Fine-tuning Guide

## Using hugohrban/ProGen2-finetuning Repository

This guide explains how to fine-tune the ProGen2 model from the [hugohrban/ProGen2-finetuning](https://github.com/hugohrban/ProGen2-finetuning) repository for serine protease sequence generation.

## Model Information

- **Repository**: https://github.com/hugohrban/ProGen2-finetuning
- **HuggingFace Model**: `hugohrban/progen2-small`
- **Model Size**: ~151M parameters
- **Paper**: [Protein Family Sequence Generation through ProGen2 Fine-Tuning](https://ieeexplore.ieee.org/document/10821712)

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies:
- `transformers>=4.30.0`
- `tokenizers>=0.13.0` (required for this model)
- `torch>=2.0.0`

## Data Format

The repository uses a specific sequence format:

```
[family_token]1[sequence]2
```

Where:
- `family_token`: Optional protein family token (e.g., `<|pf03668|>`)
- `1`: Start token
- `sequence`: Amino acid sequence
- `2`: End token

### Example

Without family token:
```
1MEVVIVTGMSGAGK2
```

With family token:
```
<|pf00089|>1MEVVIVTGMSGAGK2
```

## Fine-tuning

### Basic Usage

```python
from progen_finetune import finetune_progen

model, tokenizer, train_losses, val_losses = finetune_progen(
    data_dir="/kaggle/input/serine-proteases",
    model_name="hugohrban/progen2-small",
    batch_size=4,
    learning_rate=1e-4,
    num_epochs=3,
    patience=3,
    gradient_accumulation_steps=2,
    use_progen_format=True,  # Use 1/2 token format
    family_token=None  # Optional: add if you have a Pfam family code
)
```

### With Family Token

If you know the Pfam family code for serine proteases, you can add it:

```python
model, tokenizer, train_losses, val_losses = finetune_progen(
    data_dir="/kaggle/input/serine-proteases",
    model_name="hugohrban/progen2-small",
    batch_size=4,
    learning_rate=1e-4,
    num_epochs=3,
    use_progen_format=True,
    family_token="<|pf00089|>"  # Serine protease family token (example)
)
```

### Command Line

```bash
python progen_finetune.py
```

## Model Loading

The model uses the `tokenizers` library (not `AutoTokenizer`):

```python
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "hugohrban/progen2-small",
    trust_remote_code=True
)

# Load tokenizer
tokenizer = Tokenizer.from_pretrained("hugohrban/progen2-small")
tokenizer.no_padding()
```

## Sequence Generation

After fine-tuning, generate sequences:

```python
from progen_generate import generate_sequences_progen

sequences = generate_sequences_progen(
    model,
    tokenizer,
    num_sequences=100,
    device=device,
    max_length=300,
    temperature=1.0
)
```

### With Prompt

You can use prompts in the format expected by the model:

```python
prompt = "<|pf00089|>1MEVVIVTGMSGAGK"  # Family token + start + partial sequence
sequences = generate_sequences_progen(
    model,
    tokenizer,
    num_sequences=10,
    prompt=prompt
)
```

## Key Differences from Standard ProGen

1. **Tokenizer**: Uses `tokenizers` library instead of `AutoTokenizer`
2. **Sequence Format**: Requires `1` and `2` tokens at start/end
3. **Family Tokens**: Supports optional family tokens like `<|pf00089|>`
4. **No Padding**: Tokenizer has padding disabled by default

## Troubleshooting

### Tokenizer Issues

If you get errors about the tokenizer:

```bash
pip install tokenizers
```

The model requires the `tokenizers` library, not just `transformers`.

### Model Loading Errors

If the model fails to load:

1. Check internet connection (model downloads from HuggingFace)
2. Verify model name: `hugohrban/progen2-small`
3. Ensure `trust_remote_code=True` is set
4. Check repository: https://github.com/hugohrban/ProGen2-finetuning

### Memory Issues

For P100 GPU (16GB):

- Use `batch_size=4` with `gradient_accumulation_steps=2`
- Enable gradient checkpointing (automatic)
- Use float16 (automatic on CUDA)

## References

- **Repository**: https://github.com/hugohrban/ProGen2-finetuning
- **HuggingFace**: https://huggingface.co/hugohrban/progen2-small
- **Paper**: Hrbáň, H., & Hoksza, D. (2024). Protein Family Sequence Generation through ProGen2 Fine-Tuning. 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM).

## Citation

If you use this model, please cite:

```bibtex
@INPROCEEDINGS{10821712,
  author={Hrbáň, Hugo and Hoksza, David},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)}, 
  title={Protein Family Sequence Generation through ProGen2 Fine-Tuning}, 
  year={2024},
  pages={7037-7039},
  doi={10.1109/BIBM62325.2024.10821712}
}
```

