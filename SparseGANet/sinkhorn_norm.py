import math
import torch
from torch import nn
from operator import mul
from math import gcd
import torch.nn.functional as F
from inspect import isfunction
from functools import partial, wraps, reduce

def sample_gumbel(shape, device, dtype, eps=1e-6):
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(0, 1)
    return -torch.log(-torch.log(u + eps) + eps)

def sinkhorn_sorting_operator(r, n_iters=8):
    n = r.shape[1]
    for _ in range(n_iters):
        r = r - torch.logsumexp(r, dim=2, keepdim=True)
        r = r - torch.logsumexp(r, dim=1, keepdim=True)
    return torch.exp(r)

def gumbel_sinkhorn(r, n_iters=8, temperature=0.7):
    r = torch.log(r)
    gumbel = sample_gumbel(r.shape, r.device, r.dtype)
    r = (r + gumbel) / temperature
    return sinkhorn_sorting_operator(r, n_iters)
    
class AttentionPermMatrix(nn.Module):
    def __init__(self, blocks, temperature, sinkhorn_iter):
        super().__init__()
        self.blocks = blocks
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter

    def forward(self, b_q, b_k):
        sq = b_q.mean(dim=1).transpose(1,2)          # B * C * H_block * block  -->  B * block * H_block
        sk = b_k.mean(dim=1).transpose(1,2)          # B * C * H_block * block  -->  B * block * H_block

        R = torch.einsum('bie,bje->bij', sq, sk).to(b_q) * (b_q.shape[1] ** -0.5)

        return gumbel_sinkhorn(F.relu(R), self.sinkhorn_iter, self.temperature)
        
class SparseAttention(nn.Module):
    def __init__(self, blocks, temperature, sinkhorn_iter, inchannel):
        super().__init__()
        self.blocks = blocks
        self.temperature = temperature
        self.perm_mat = AttentionPermMatrix(blocks, temperature, sinkhorn_iter, inchannel)
        
    def forward(self, q, k, v):
        b_q = torch.cat(q.chunk(self.blocks, dim=2)[:-1], dim=-1)        # B * C * H * 1  -->  B * C * H_block * block  
        b_k = torch.cat(k.chunk(self.blocks, dim=2)[:-1], dim=-1)        # B * C * H * 1  -->  B * C * H_block * block
        b_v = torch.cat(v.chunk(self.blocks, dim=2)[:-1], dim=-1)        # B * C * H * 1  -->  B * C * H_block * block
        
        perm = self.perm_mat(b_q, b_k).unsqueeze(1).repeat(1, q.shape[1], 1, 1)           # B * C * block * block
        k_sort = torch.matmul(b_k, perm)                                                  # B * C * H_block * block
        v_sort = torch.matmul(b_v, perm)                                                  # B * C * H_block * block
        
        b_q = b_q.transpose(1, 3)                                                         # B * block * H_block * C
        b_k_sort = k_sort.transpose(1, 3)                                                 # B * block * H_block * C
        b_v_sort = v_sort.transpose(1, 3)                                                 # B * block * H_block * C
        
        attn_logits = torch.matmul(b_q, b_k_sort.transpose(2,3)) / self.temperature       # B * block * H_block * H_block
        attn = F.softmax(attn_logits, dim=-1)
        value = torch.matmul(attn, b_v_sort)                                                # B * block * H_block * C
        
        org_perm_val = torch.matmul(perm, value.transpose(1,3).transpose(2,3)).transpose(2,3).transpose(1,3)            # B * block * H_block * C
        b_v_last = v.chunk(self.blocks, dim=2)[-1].transpose(1,3)
        
        return torch.cat([torch.cat(org_perm_val.chunk(self.blocks, dim=1), dim=2), b_v_last], dim =2)
        
class SparseCutAttention(nn.Module):
    def __init__(self, blocks, temperature, sinkhorn_iter, inchannel, cut_length):
        super().__init__()
        self.blocks = blocks
        self.temperature = temperature
        self.cut_length = cut_length
        self.perm_mat = AttentionPermMatrix(blocks, temperature, sinkhorn_iter)
        self.norm = nn.InstanceNorm2d(inchannel) 
        
    def forward(self, q, k, v):
        b_q = torch.cat(q.chunk(self.blocks, dim=2)[:-1], dim=-1)        # B * C * H * 1  -->  B * C * H_block * block-1  
        b_k = torch.cat(k.chunk(self.blocks, dim=2)[:-1], dim=-1)        # B * C * H * 1  -->  B * C * H_block * block-1
        b_v = torch.cat(v.chunk(self.blocks, dim=2)[:-1], dim=-1)        # B * C * H * 1  -->  B * C * H_block * block-1
        
        perm = self.perm_mat(b_q, b_k).unsqueeze(1).repeat(1, q.shape[1], 1, 1)           # B * C * block-1 * block-1
        k_sort = torch.matmul(b_k, perm)                                                  # B * C * H_block * block-1
        v_sort = torch.matmul(b_v, perm)                                                  # B * C * H_block * block-1
        
        q = q.transpose(1, 3)                                                             # B * 1 * H * C
        k_sort = torch.cat(k_sort.chunk(self.blocks, dim=-1)[:self.cut_length], dim=-2).transpose(1, 3)   # B * 1 * H_cut * C
        v_sort = torch.cat(v_sort.chunk(self.blocks, dim=-1)[:self.cut_length], dim=-2).transpose(1, 3)   # B * 1 * H_cut * C
        
        #for i in range(cut_length):
        attn_logits = torch.matmul(q, k_sort.transpose(2,3)) / self.temperature            # B * 1 * H * H_cut
        attn = F.softmax(attn_logits, dim=-1)
        val = torch.matmul(attn, v_sort)                                                   # B * 1 * H * C
        val = self.norm(val)
        
        return val
        
        '''
        b_v = torch.cat(val.transpose(1,3).chunk(self.blocks, dim=2)[:-1], dim=-1)         # B * C * H * 1  -->  B * C * H_block * block-1
        
        org_perm_val = torch.matmul(perm, b_v.transpose(2,3))                              # B * C * block-1 * H_block
        b_v_last = v.chunk(self.blocks, dim=2)[-1].transpose(1,3)
        
        return torch.cat([torch.cat(org_perm_val.chunk(self.blocks-1, dim=1), dim=2), b_v_last], dim =2)
        '''
        
