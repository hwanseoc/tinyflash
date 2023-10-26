# Tinyflash

Tinyflash is a minimal demo of the flash attention algorithm in PyTorch and Einops. The core of the algorithm is implemented in a mere few lines.

- `flash_attn_forward.py` contains the algorithm demo for inference.
- `flash_attn_backward.py` contains the algorithm demo for training.
- you can also look at `flash_attn_forward_min.py` and `flash_attn_backward_min.py` if you want a minimal version of the code with no comments.

## Dependencies
```bash
pip install torch einops
```
Tested on Python 3.10.12

## Usage

```bash
python3 flash_attn_forward.py
```

should print

```
All tensors are equal!
```

Which means the output tensors of the flash attention demo matches the non-flash output tensors.

## Credits

- Pseudocode from flash attention v1 + v2,
[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [Julia implementation of dense flash attention](https://github.com/nikopj/FlashAttention.jl/blob/main/src/dense.jl)
