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
dO = torch.randn(b, sq, d) * 0.1

def slow_attention(q, k, v, is_stats=False):
    s = einsum(q, k, "b sq d, b skv d -> b sq skv") / math.sqrt(d)
    p = torch.softmax(s, dim=-1)
    o = einsum(p, v, "b sq skv, b skv d -> b sq d")

    if is_stats:
        m = reduce(s, "b sq skv -> b sq 1", "max")
        l = reduce(torch.exp(s - m), "b sq skv -> b sq 1", "sum")
        stats = m + torch.log(l)
        return o, stats

    return o

def flash_attention_forward_stats(q, k, v):
    q = rearrange(q, "b (tr br) d -> b tr br d", br=br)
    k = rearrange(k, "b (tc bc) d -> b tc bc d", bc=bc)
    v = rearrange(v, "b (tc bc) d -> b tc bc d", bc=bc)
    o = torch.zeros(b, tr, br, d)
    stats = torch.empty(b, tr, br, 1)

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
        stats_tile = m + torch.log(l)
        o[:, i, :, :] = o_tile
        stats[:, i, :, 0] = rearrange(stats_tile, "b br 1 -> b br")

    o = rearrange(o, "b tr br d -> b (tr br) d")
    stats = rearrange(stats, "b tr br 1 -> b (tr br) 1")
    return o, stats

def flash_attention_backward(q, k, v, o, dO, stats):
    q = rearrange(q, "b (tr br) d -> b tr br d", br=br)
    k = rearrange(k, "b (tc bc) d -> b tc bc d", bc=bc)
    v = rearrange(v, "b (tc bc) d -> b tc bc d", bc=bc)
    o = rearrange(o, "b (tr br) d -> b tr br d", br=br)
    dO = rearrange(dO, "b (tr br) d -> b tr br d", br=br)
    stats = rearrange(stats, "b (tr br) 1 -> b tr br 1", br=br)

    dQ = torch.zeros(b, tr, br, d)
    dK = torch.empty(b, tc, bc, d)
    dV = torch.empty(b, tc, bc, d)
    dot_o_dO = reduce(dO * o, "b tr br d -> b tr br 1", "sum")

    for j in range(tc):
        k_tile = k[:, j, :, :]
        v_tile = v[:, j, :, :]

        dK_tile = torch.zeros(b, bc, d)
        dV_tile = torch.zeros(b, bc, d)

        for i in range(tr):
            q_tile = q[:, i, :, :]
            dO_tile = dO[:, i, :, :]
            stats_tile = stats[:, i, :, :]
            dot_o_dO_tile = dot_o_dO[:, i, :, :]

            s = einsum(q_tile, k_tile, "b br d, b bc d -> b br bc") / math.sqrt(d)
            p = torch.exp(s - stats_tile)
            dV_tile = dV_tile + einsum(p, dO_tile, "b br bc, b br d -> b bc d")

            dP = einsum(dO_tile, v_tile, "b br d, b bc d -> b br bc")
            dS = p * (dP - dot_o_dO_tile) / math.sqrt(d)
            dQ[:, i, :, :] = dQ[:, i, :, :] + einsum(dS, k_tile, "b br bc, b bc d -> b br d")
            dK_tile = dK_tile + einsum(dS, q_tile, "b br bc, b br d -> b bc d")

        dK[:, j, :, :] = dK_tile
        dV[:, j, :, :] = dV_tile

    dQ = rearrange(dQ, "b tr br d -> b (tr br) d")
    dK = rearrange(dK, "b tc bc d -> b (tc bc) d")
    dV = rearrange(dV, "b tc bc d -> b (tc bc) d")
    return dQ, dK, dV

o1, stats1 = slow_attention(q, k, v, is_stats=True)
o2, stats2 = flash_attention_forward_stats(q, k, v)
assert torch.allclose(o1, o2, rtol=1e-4, atol=1e-6)
assert torch.allclose(stats1, stats2, rtol=1e-4, atol=1e-6)

q.requires_grad = k.requires_grad = v.requires_grad = True
dQ1, dK1, dV1 = torch.autograd.grad(outputs=slow_attention(q, k, v), inputs=(q, k, v), grad_outputs=dO)
dQ2, dK2, dV2 = flash_attention_backward(q, k, v, o2, dO, stats2)
assert torch.allclose(dQ1, dQ2, rtol=1e-4, atol=1e-6)
assert torch.allclose(dK1, dK2, rtol=1e-4, atol=1e-6)
assert torch.allclose(dV1, dV2, rtol=1e-4, atol=1e-6)
print("All tensors are equal!")
