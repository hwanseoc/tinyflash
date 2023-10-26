import math
import torch
from einops import rearrange, reduce, einsum

b = 48
sq = 256
skv = 128
d = 64

br = 16
bc = 32
tr = sq // br
tc = skv // bc

q = torch.randn(b, sq, d)
k = torch.randn(b, skv, d)
v = torch.randn(b, skv, d)

def slow_attention(q, k, v):
    s = einsum(q, k, "b sq d, b skv d -> b sq skv") / math.sqrt(d)
    p = torch.softmax(s, dim=-1)
    o = einsum(p, v, "b sq skv, b skv d -> b sq d")
    return o

def flash_attention_forward(q, k, v):
    q = rearrange(q, "b (tr br) d -> b tr br d", br=br)
    k = rearrange(k, "b (tc bc) d -> b tc bc d", bc=bc)
    v = rearrange(v, "b (tc bc) d -> b tc bc d", bc=bc)
    o = torch.zeros(b, tr, br, d)

    for i in range(tr):
        q_tile = q[:, i, :, :]
        o_tile = torch.zeros(b, br, d)
        l = torch.zeros(b, br, 1)
        m = torch.full((b, br, 1), float("-inf"))
        for j in range(tc):
            k_tile = k[:, j, :, :]
            v_tile = v[:, j, :, :]
            s = einsum(q_tile, k_tile, "b br d, b bc d -> b br bc") / math.sqrt(d)
            m_new = torch.maximum(m, reduce(s, "b br bc -> b br 1", "max"))
            p = torch.exp(s - m_new)
            if j != 0:
                u = torch.exp(m - m_new)
            else:
                u = torch.zeros(b, br, 1)
            m = m_new
            l = u * l + reduce(p, "b br bc -> b br 1", "sum")
            o_tile = o_tile * u
            o_tile = o_tile + einsum(p, v_tile, "b br bc, b bc d -> b br d")
        o_tile = o_tile / l
        o[:, i, :, :] = o_tile

    o = rearrange(o, "b tr br d -> b (tr br) d")
    return o

o1 = slow_attention(q, k, v)
o2 = flash_attention_forward(q, k, v)
assert torch.allclose(o1, o2, rtol=1e-4, atol=1e-6)
print("All tensors are equal!")
