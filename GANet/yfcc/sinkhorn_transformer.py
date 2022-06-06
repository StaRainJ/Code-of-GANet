import math
import torch
from torch import nn
from operator import mul
from math import gcd
import torch.nn.functional as F
from inspect import isfunction
from functools import partial, wraps, reduce

from local_attention import LocalAttention
from axial_positional_embedding import AxialPositionalEmbedding
from product_key_memory import PKM
from sinkhorn_transformer.reversible import ReversibleSequence, SequentialSequence

# helper functions

def identity(x, *args, **kwargs): return x

def default(x, d):
    if x is None:
        return d if not isfunction(d) else d()
    return x

def cast_tuple(x):
    return x if isinstance(x, tuple) else (x,)

def divisible_by(num, divisor):
    return num % divisor == 0

def lcm(*numbers):
    return int(reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))

def all_none(*arr):
    return all(el is None for el in arr)

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def rotate_left(t, n, dim=0):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(n, None))
    r = (*pre_slices, slice(0, n))
    return torch.cat((t[l], t[r]), dim=dim)

def rotate_right(t, n, dim=0):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(-n, None))
    r = (*pre_slices, slice(None, -n))
    return torch.cat((t[l], t[r]), dim=dim)

def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

def merge_heads(h, v):
    b, t, d = v.shape
    return v.view(b, t, h, -1).transpose(1, 2).reshape(b, h, t, -1)

def split_heads(h, v):
    *_, t, d = v.shape
    return v.view(-1, h, t, d).transpose(1, 2).reshape(-1, t, d * h)

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

def bucket(buckets, t, dim=1):
    shape = list(t.shape)
    shape[dim:dim+1] = [buckets, -1]
    return t.reshape(*shape)

def unbucket(t, dim=1):
    shape = list(t.shape)
    shape[dim:dim+2] = [-1]
    return t.reshape(*shape)

def sample_gumbel(shape, device, dtype, eps=1e-6):
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(0, 1)
    return -log(-log(u, eps), eps)

def sinkhorn_sorting_operator(r, n_iters=8):
    n = r.shape[1]
    for _ in range(n_iters):
        r = r - torch.logsumexp(r, dim=2, keepdim=True)
        r = r - torch.logsumexp(r, dim=1, keepdim=True)
    return torch.exp(r)

def gumbel_sinkhorn(r, n_iters=8, temperature=0.7):
    r = log(r)
    gumbel = sample_gumbel(r.shape, r.device, r.dtype)
    r = (r + gumbel) / temperature
    return sinkhorn_sorting_operator(r, n_iters)

def reorder_buckets(t, r):
    return torch.einsum('buv,bvtd->butd', r, t)

def log(t, eps = 1e-6):
    return torch.log(t + eps)

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def cumavg(t, dim):
    r = torch.arange(1, t.shape[dim] + 1, device=t.device, dtype=t.dtype)
    expand_slice = [None] * len(t.shape)
    expand_slice[dim] = slice(None, None)
    return t.cumsum(dim=dim) / r[tuple(expand_slice)]

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

def expand_dim(t, dim, k):
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def expand_batch_and_merge_head(b, t):
    shape = list(t.squeeze(0).shape)
    t = expand_dim(t, 0, b)
    shape[0] = shape[0] * b
    return t.reshape(*shape)

def differentiable_topk(x, k, temperature=1.):
    *_, n, dim = x.shape
    topk_tensors = []

    for i in range(k):
        is_last = i == (k - 1)
        values, indices = (x / temperature).softmax(dim=-1).topk(1, dim=-1)
        topks = torch.zeros_like(x).scatter_(-1, indices, values)
        topk_tensors.append(topks)
        if not is_last:
            x.scatter_(-1, indices, float('-inf'))

    topks = torch.cat(topk_tensors, dim=-1)
    return topks.reshape(*_, k * n, dim)

# helper classes

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x):
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c) for c in chunks], dim = self.dim)

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(1))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g

class PreNorm(nn.Module):
    def __init__(self, norm_class, dim, fn):
        super().__init__()
        self.norm = norm_class(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class ProjectInOut(nn.Module):
    def __init__(self, fn, dim_in, dim_out, project_out = True):
        super().__init__()
        self.fn = fn
        self.project_in = nn.Linear(dim_in, dim_out)
        self.project_out = nn.Linear(dim_out, dim_in) if project_out else identity

    def forward(self, x, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, **kwargs)
        x = self.project_out(x)
        return x

# non-causal sortnet and sinkhorn attention

class SimpleSortNet(nn.Module):
    def __init__(self, heads, bucket_size, max_buckets, dim, non_permutative, temperature, sinkhorn_iter):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.max_buckets = max_buckets
        self.bucket_size = bucket_size
        self.non_permutative = non_permutative
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter
        self.linear = nn.Parameter(torch.randn(1, heads, dim, max_buckets))
        self.act = nn.ReLU()

    def forward(self, q, k, topk=1):
        bh, t, _ = q.shape
        b = bh // self.heads
        buckets = t // self.bucket_size

        b_q, b_k = bucket(buckets, q), bucket(buckets, k)
        x = torch.cat((b_q.sum(dim=2), b_k.sum(dim=2)), dim=-1)

        W = expand_batch_and_merge_head(b, self.linear)
        R = self.act(x @ W)

        return differentiable_topk(R, k=topk, temperature=self.temperature) if self.non_permutative else gumbel_sinkhorn(R, self.sinkhorn_iter, self.temperature)

class AttentionSortNet(nn.Module):
    def __init__(self, heads, bucket_size, kv_bucket_size, dim, non_permutative, temperature, sinkhorn_iter, n_sortcut = 0):
        super().__init__()
        self.heads = heads
        self.bucket_size = bucket_size
        self.kv_bucket_size = kv_bucket_size
        self.dim = dim
        self.non_permutative = non_permutative
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter
        self.n_sortcut = n_sortcut

    def forward(self, q, k, topk=1):
        bh, *_, bucket_size, kv_bucket_size, device, dtype, dim = *q.shape, self.bucket_size, self.kv_bucket_size, q.device, q.dtype, self.dim
        b = bh // self.heads

        buckets = q.shape[1] // bucket_size
        kv_buckets = k.shape[1] // kv_bucket_size

        b_q = bucket(buckets, q) if self.n_sortcut == 0 else bucket(1, q)
        b_k = bucket(kv_buckets, k)

        sq = b_q.mean(dim=2)
        sk = b_k.mean(dim=2)

        R = torch.einsum('bie,bje->bij', sq, sk).to(q) * (dim ** -0.5)

        if self.non_permutative:
            k = topk if self.n_sortcut == 0 else self.n_sortcut
            return differentiable_topk(R, k=k)

        return gumbel_sinkhorn(F.relu(R), self.sinkhorn_iter, self.temperature)

class SinkhornAttention(nn.Module):
    def __init__(self, bucket_size, dim, dim_heads, heads, max_seq_len, temperature = 0.75, non_permutative = True, sinkhorn_iter = 7, n_sortcut = 0, dropout = 0., kv_bucket_size = None, use_simple_sort_net = False, n_top_buckets = 1):
        super().__init__()
        self.bucket_size = bucket_size
        self.kv_bucket_size = default(kv_bucket_size, bucket_size)

        self.dim = dim
        self.heads = heads
        self.temperature = temperature
        self.non_permutative = non_permutative
        self.sinkhorn_iter = sinkhorn_iter
        self.n_sortcut = n_sortcut

        if use_simple_sort_net:
            self.sort_net = SimpleSortNet(heads, self.kv_bucket_size, max_seq_len // self.kv_bucket_size, dim_heads * 2, non_permutative = non_permutative, temperature = temperature, sinkhorn_iter = sinkhorn_iter)
        else:
            self.sort_net = AttentionSortNet(heads, self.bucket_size, self.kv_bucket_size, dim_heads, non_permutative = non_permutative, temperature = temperature, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut)

        self.n_top_buckets = n_top_buckets
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, q_mask = None, kv_mask = None):
        b, h, t, d_h, n_top, d, heads, temperature, bucket_size, kv_bucket_size, device = *q.shape, self.n_top_buckets, self.dim, self.heads, self.temperature, self.bucket_size, self.kv_bucket_size, q.device

        bh = b * h
        buckets = q.shape[2] // bucket_size
        kv_buckets = k.shape[2] // kv_bucket_size
        n_top = min(n_top, kv_buckets)

        merge_batch_head = partial(merge_dims, 0, 1)
        q, k, v = map(merge_batch_head, (q, k, v))

        # bucket query, key, values

        b_q = bucket(buckets, q)
        b_k, b_v = map(partial(bucket, kv_buckets), (k, v))

        bsz = b_k.shape[2]

        # calculate reordering matrix R with simple sort net

        R = self.sort_net(q, k, topk=n_top)
        R = R.type_as(q).to(q)

        # concatenate reordered buckets

        b_k_r = reorder_buckets(b_k, R)
        b_v_r = reorder_buckets(b_v, R)

        # choose the top n ranked buckets for all query buckets

        if self.n_sortcut > 0:
            b_k_r = b_k_r[:, 0:self.n_sortcut].reshape(bh, 1, -1, d_h)
            b_v_r = b_v_r[:, 0:self.n_sortcut].reshape(bh, 1, -1, d_h)
            b_k_r = expand_dim(b_k_r, 1, buckets)
            b_v_r = expand_dim(b_v_r, 1, buckets)
        else:
            b_k_r = b_k_r.reshape(bh, buckets, -1, d_h)
            b_v_r = b_k_r.reshape(bh, buckets, -1, d_h)

        b_k = torch.cat((b_k_r, b_k), dim=2) if buckets == kv_buckets else b_k_r
        b_v = torch.cat((b_v_r, b_v), dim=2) if buckets == kv_buckets else b_v_r

        dots = torch.einsum('buie,buje->buij', b_q, b_k) * (d_h ** -0.5)

        # mask 
        mask_value = max_neg_value(dots)

        if not all_none(q_mask, kv_mask):
            q_mask = default(q_mask, lambda: torch.ones((b, t), device=device).bool())
            kv_mask = default(kv_mask, q_mask)
            mq, mk = bucket(buckets, q_mask), bucket(kv_buckets, kv_mask)
            expand_head_and_merge_into_batch = lambda x: merge_dims(0, 1, expand_dim(x.unsqueeze(1), 1, h))
            mq, mk = map(expand_head_and_merge_into_batch, (mq, mk))

            mk_r = batched_index_select(mk, R.abs().argmax(dim=-1))

            if self.n_sortcut > 0:
                mk_r = mk_r[:, 0:self.n_sortcut].reshape(-1, 1, bsz * self.n_sortcut)
                mk_r = expand_dim(mk_r, 1, buckets)
            else:
                mk_r = mk_r.reshape(bh, buckets, -1)

            mk = torch.cat((mk_r, mk), dim=2) if buckets == kv_buckets else mk_r
            mask = mq[:, :, :, None] * mk[:, :, None, :]
            dots.masked_fill_(~mask, mask_value)
            del mask            

        # attention
        dots = dots.softmax(dim=-1)
        dots = self.dropout(dots)

        out = torch.einsum('buij,buje->buie', dots, b_v)
        out = unbucket(out)

        out = out.reshape(b, h, t, d_h)
        return out

class SinkhornSelfAttention(nn.Module):
    def __init__(self, dim, bucket_size, max_seq_len, heads = 8, dim_head = None, kv_bucket_size = None, causal = False, non_permutative = True, sinkhorn_iter = 5, n_sortcut = 0, temperature = 0.75, attn_dropout = 0., dropout = 0., context_only = False, use_simple_sort_net = False, n_local_attn_heads = 0, n_top_buckets = 1):
        super().__init__()
        assert dim_head or divisible_by(dim, heads), f'If dim_head is None, dimension {dim} must be divisible by the number of heads {heads}'
        assert not (causal and n_sortcut > 0), 'sortcut can only be used for non causal attention'
        assert not (causal and context_only), 'context only self attention layer cannot be causal'
        assert n_local_attn_heads <= heads, 'number of local attention heads cannot exceed total heads'

        dim_head = default(dim_head, dim // heads)
        dim_heads = dim_head * heads
        self.dim_head = dim_head

        self.heads = heads
        self.bucket_size = bucket_size
        self.kv_bucket_size = default(kv_bucket_size, bucket_size)

        self.context_only = context_only
        self.to_q = nn.Linear(dim, dim_heads, bias=False)
        self.to_kv = nn.Linear(dim, dim_heads * 2, bias=False) if not context_only else None

        self.to_out = nn.Linear(dim_heads, dim)

        self.n_local_attn_heads = n_local_attn_heads
        self.local_attention = LocalAttention(bucket_size, causal, dropout = attn_dropout, look_forward=(1 if not causal else 0))

        sink_heads = heads - n_local_attn_heads

        if causal:
            attn = SinkhornCausalAttention(bucket_size, dim, dim_head, sink_heads, max_seq_len, dropout = attn_dropout, kv_bucket_size = kv_bucket_size, use_simple_sort_net = use_simple_sort_net, n_top_buckets = n_top_buckets, temperature = temperature)
        else:
            attn = SinkhornAttention(bucket_size, dim, dim_head, sink_heads, max_seq_len, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature, dropout = attn_dropout, kv_bucket_size = kv_bucket_size, use_simple_sort_net = use_simple_sort_net, n_top_buckets = n_top_buckets)

        self.sinkhorn_attention = attn

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, input_mask = None, context = None, context_mask = None):
        b, t, d, h, dh, l_h = *x.shape, self.heads, self.dim_head, self.n_local_attn_heads
        assert divisible_by(t, self.bucket_size), f'sequence {t} needs to be divisible by bucket size {self.bucket_size}'
        assert not (self.context_only and context is None), 'context key / values must be supplied if context self attention layer'
        assert not (context is not None and (context.shape[0], context.shape[2]) !=  (b, d)), 'contextual key / values must have the same batch and dimensions as the decoder'

        q = self.to_q(x)

        kv = self.to_kv(x).chunk(2, dim=-1) if not self.context_only else (context, context)
        kv_mask = input_mask if not self.context_only else context_mask

        assert divisible_by(kv[0].shape[1], self.kv_bucket_size), 'key/value sequences need to be divisible by key/value bucket size'

        qkv = (q, *kv)
        merge_heads_fn = partial(merge_heads, h)
        q, k, v = map(merge_heads_fn, qkv)

        split_index_fn = partial(split_at_index, 1, l_h)
        (lq, q), (lk, k), (lv, v) = map(split_index_fn, (q, k, v))
        has_local, has_sinkhorn = map(lambda x: x.shape[1] > 0, (lq, q))

        out = []

        if has_local > 0:
            out.append(self.local_attention(lq, lk, lv, input_mask = input_mask))

        if has_sinkhorn > 0:
            out.append(self.sinkhorn_attention(q, k, v, q_mask = input_mask, kv_mask = kv_mask))

        out = torch.cat(out, dim=1)
        out = split_heads(h, out)
        out = self.to_out(out)
        out = self.dropout(out)
        return out


