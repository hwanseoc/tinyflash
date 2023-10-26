import math
import torch
from einops import rearrange, reduce, einsum

# note: when this code refers to attention,
# it means scaled-dot-product attention
# also, this code does not perform masking and dropout

b = 48              # batch size * number of heads
sq = 256            # query sequence length
skv = 128           # key value sequence length
d = 64              # embedding size per head

br = 16             # block size row
bc = 32             # block size col
tr = sq // br       # number of blocks row
tc = skv // bc      # number of blocks col

# sequence lengths must be divisible by block sizes
assert sq % br == 0
assert skv % bc == 0

q = torch.randn(b, sq, d)
k = torch.randn(b, skv, d)
v = torch.randn(b, skv, d)

def slow_attention(q, k, v):
    s = einsum(q, k, "b sq d, b skv d -> b sq skv") / math.sqrt(d)
    p = torch.softmax(s, dim=-1)
    o = einsum(p, v, "b sq skv, b skv d -> b sq d")
    return o

def flash_attention_forward(q, k, v):
    # split the qkvo tensor into gpu blocks
    q = rearrange(q, "b (tr br) d -> b tr br d", br=br)
    k = rearrange(k, "b (tc bc) d -> b tc bc d", bc=bc)
    v = rearrange(v, "b (tc bc) d -> b tc bc d", bc=bc)
    o = torch.zeros(b, tr, br, d)

    for i in range(tr): # "b" and "tr" is computed in parallel over blocks in a gpu
        q_tile = q[:, i, :, :] # shape (b, br, d)
        o_tile = torch.zeros(b, br, d)
        l = torch.zeros(b, br, 1) # running exp sum
        m = torch.full((b, br, 1), float("-inf")) # running max
        for j in range(tc): # also called kv mainloop
            k_tile = k[:, j, :, :] # shape (b, bc, d)
            v_tile = v[:, j, :, :] # shape (b, bc, d)
            s = einsum(q_tile, k_tile, "b br d, b bc d -> b br bc") / math.sqrt(d)
            m_new = torch.maximum(m, reduce(s, "b br bc -> b br 1", "max"))
            p = torch.exp(s - m_new) # broadcast subtract
            if j != 0:
                u = torch.exp(m - m_new) # scale to update l and o
            else:
                u = torch.zeros(b, br, 1)
            m = m_new
            l = u * l + reduce(p, "b br bc -> b br 1", "sum") # update l
            o_tile = o_tile * u # update o
            o_tile = o_tile + einsum(p, v_tile, "b br bc, b bc d -> b br d")
        o_tile = o_tile / l # final normalize o
        o[:, i, :, :] = o_tile

    # concatenate the o tensor blocks back
    o = rearrange(o, "b tr br d -> b (tr br) d")
    return o

o1 = slow_attention(q, k, v)
o2 = flash_attention_forward(q, k, v)
assert torch.allclose(o1, o2, rtol=1e-4, atol=1e-6)
print("All tensors are equal!")
