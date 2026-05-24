"""DC attention prefill.

Key insight: QK^T is per-head independent — just like standard MHA.
Only the mixing step needs cross-head communication. So:

  1. QK^T: standard batched matmul [B,N,T,D]@[B,N,D,T] -> [B,N,T,T]  (tensor core)
  2. Pre-mix: [B*T, N, N] @ [B*T, N, T] -> [B*T, N, T]  (tensor core)
  3. Causal mask + softmax
  4. Post-mix: same shape as pre-mix  (tensor core)
  5. P@V: [B,N,T,T] @ [B,N,T,D] -> [B,N,T,D]  (tensor core)

This materializes the full [B,N,T,T] attention matrix (O(T^2) memory),
which is fine for T <= 2048-4096 on 80GB cards.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

## full DC
def dc_attention_window_chunked_residual(
    q: torch.Tensor,       # [B, T, N, D]
    k: torch.Tensor,       # [B, T, N, D]
    v: torch.Tensor,       # [B, T, N, D]
    dc_weights: tuple[torch.Tensor, torch.Tensor],   # [B, T, N]
    scaling: float,
    window: int=None,
    seq_lens: torch.Tensor=None,  # [B]
    chunk_size: int=None,
) -> torch.Tensor:
    # all shape is [B, T, N]
    pre_w1, pre_w2, pre_dd,post_w1, post_w2, post_dd = dc_weights
    B, T, N, D = q.shape
    if window is None:
        window = 256
    if seq_lens is None:
        seq_lens = torch.full((B,), T, device=q.device, dtype=torch.int32)
    if chunk_size is None:
        chunk_size = 256

    out_chunks = []
    for q_start in range(0, T, chunk_size):
        q_end = min(q_start + chunk_size, T)
        k_start = max(q_start - window, 0)
        k_end = q_end

        q_chunk = q[:, q_start:q_end] # btnd
        k_chunk = k[:, k_start:k_end]
        v_chunk = v[:, k_start:k_end]
        
        logits_chunk = torch.einsum('btnd,bsnd->bnts', q_chunk, k_chunk) * scaling # btns

        pre_dd_chunk = pre_dd[:, q_start:q_end] # btn
        logits_pre_dd_chunk = torch.einsum('bnts,btn->bnts', logits_chunk, pre_dd_chunk)

        pre_w1_chunk = pre_w1[:, q_start:q_end] # btn
        pre_w1_logits_chunk = torch.einsum('bnts,btn->bts', logits_chunk, pre_w1_chunk) # bts
        pre_w2_chunk = pre_w2[:, q_start:q_end]
        pre_w2_logits_chunk = torch.einsum('bts,btn->bnts', pre_w1_logits_chunk, pre_w2_chunk)
        
        logits_chunk = logits_chunk + pre_w2_logits_chunk + logits_pre_dd_chunk
        
        # mask and softmax
        q_idx = torch.arange(q_start, q_end, device=q.device)
        k_idx = torch.arange(k_start, k_end, device=q.device)
        causal = k_idx[None, :] <= q_idx[:, None]
        win_mask = (q_idx[:, None] - k_idx[None, :]) < window
        seq_mask = k_idx[None, :] < seq_lens[:, None]
        mask = causal[None, None, :, :] & win_mask[None, None, :, :]
        mask = mask & seq_mask[:, None, None, :] # b1ts
        logits_chunk = logits_chunk.masked_fill(~mask, float('-inf'))
        probs_chunk = torch.softmax(logits_chunk, dim=-1)
        probs_chunk = torch.nan_to_num(probs_chunk, nan=0.0)

        post_dd_chunk = post_dd[:, q_start:q_end] # btn
        post_dd_probs_chunk = torch.einsum('bnts,btn->bnts', probs_chunk, post_dd_chunk)

        post_w1_chunk = post_w1[:, q_start:q_end] # btn
        post_w1_probs_chunk = torch.einsum('bnts,btn->bts', probs_chunk, post_w1_chunk) # bts
        post_w2_chunk = post_w2[:, q_start:q_end]
        post_w2_probs_chunk = torch.einsum('bts,btn->bnts', post_w1_probs_chunk, post_w2_chunk)

        probs_chunk = probs_chunk + post_w2_probs_chunk + post_dd_probs_chunk

        post_w2_probs_chunk = torch.einsum('bnts,bsnd->bntd', probs_chunk, v_chunk)
        
        out_chunks.append(post_w2_probs_chunk)

    return torch.cat(out_chunks, dim=2)


# post-only and remove dd, also is only post w1 and w2
def dc_attention_postonly_nodd(
    q: torch.Tensor,       # [B, T, N, D]
    k: torch.Tensor,       # [B, T, N, D]
    v: torch.Tensor,       # [B, T, N, D]
    dc_weights: tuple[torch.Tensor, torch.Tensor],   # [B, T, N]
    scaling: float,
    window: int=None,
    seq_lens: torch.Tensor=None,  # [B] lengths or packed cu_seqlens for B=1
    chunk_size: int=None,
    max_seq_len: int=None,
) -> torch.Tensor:
    # all shape is [B, T, N]
    post_w1, post_w2 = dc_weights
    B, T, N, D = q.shape
    if window is None:
        window = 128
    if seq_lens is None:
        seq_lens = torch.full((B,), T, device=q.device, dtype=torch.int32)
    if chunk_size is None:
        chunk_size = 256
    if seq_lens.numel() != B:
        assert B == 1, "packed cu_seqlens path expects B == 1"
        if max_seq_len is None:
            max_seq_len = int((seq_lens[1:] - seq_lens[:-1]).max().item())
        return dc_attention_postonly_nodd_padded(
            q, k, v, (post_w1, post_w2), scaling, window, seq_lens,
            max_seq_len=max_seq_len,
            query_block_size=chunk_size,
        )

    pv_chunks = []
    for q_start in range(0, T, chunk_size):
        q_end = min(q_start + chunk_size, T)
        k_start = max(q_start - window, 0)
        k_end = q_end

        q_chunk = q[:, q_start:q_end] # btnd
        k_chunk = k[:, k_start:k_end]
        v_chunk = v[:, k_start:k_end]
        
        logits_chunk = torch.einsum('btnd,bsnd->bnts', q_chunk, k_chunk) * scaling # btns
      
        # mask and softmax
        q_idx = torch.arange(q_start, q_end, device=q.device)
        k_idx = torch.arange(k_start, k_end, device=q.device)
        causal = k_idx[None, :] <= q_idx[:, None]
        win_mask = (q_idx[:, None] - k_idx[None, :]) < window
        if seq_lens.numel() == B:
            seq_mask = k_idx[None, :] < seq_lens[:, None]
            mask = causal[None, None, :, :] & win_mask[None, None, :, :]
            mask = mask & seq_mask[:, None, None, :] # b1ts
        else:
            assert B == 1, "packed cu_seqlens path expects B == 1"
            doc_ids = torch.searchsorted(seq_lens, q_idx, right=True) - 1
            doc_starts = seq_lens[doc_ids]
            doc_ends = seq_lens[doc_ids + 1]
            doc_mask = (k_idx[None, :] >= doc_starts[:, None]) & (k_idx[None, :] < doc_ends[:, None])
            mask = (causal & win_mask & doc_mask)[None, None, :, :]
        logits_chunk = logits_chunk.masked_fill(~mask, float('-inf'))
        probs_chunk = torch.softmax(logits_chunk, dim=-1)
        probs_chunk = torch.nan_to_num(probs_chunk, nan=0.0)

        post_w1_chunk = post_w1[:, q_start:q_end] # btn
        post_w1_probs_chunk = torch.einsum('bnts,btn->bts', probs_chunk, post_w1_chunk) # bts

        post_w2_chunk = post_w2[:, q_start:q_end]
        post_w2_probs_chunk = torch.einsum('bts,btn->bnts', post_w1_probs_chunk, post_w2_chunk)

        probs_chunk = probs_chunk + post_w2_probs_chunk
        pv_chunk = torch.einsum('bnts,bsnd->bntd', probs_chunk, v_chunk)
        
        pv_chunks.append(pv_chunk)

    return torch.cat(pv_chunks, dim=2)


def dc_attention_postonly_nodd_padded(
    q: torch.Tensor,       # [1, T, N, D], packed documents
    k: torch.Tensor,       # [1, T, N, D]
    v: torch.Tensor,       # [1, T, N, D]
    dc_weights: tuple[torch.Tensor, torch.Tensor],   # [1, T, N]
    scaling: float,
    window: int,
    cu_seqlens: torch.Tensor,
    max_seq_len: int,
    query_block_size: int,
) -> torch.Tensor:
    post_w1, post_w2 = dc_weights
    B, T, N, D = q.shape
    assert B == 1

    S = int(max_seq_len)
    C = max(1, min(int(query_block_size), S))
    starts = cu_seqlens[:-1].to(torch.long)
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).clamp(min=0, max=S).to(torch.long)
    Bdoc = starts.numel()

    out_seq = q.new_zeros(T, N, D)
    for q_start in range(0, S, C):
        q_end = min(q_start + C, S)
        k_start = max(q_start - window + 1, 0)
        k_end = q_end

        q_pos = torch.arange(q_start, q_end, device=q.device)
        k_pos = torch.arange(k_start, k_end, device=q.device)
        q_valid = q_pos[None, :] < lengths[:, None]
        k_valid = k_pos[None, :] < lengths[:, None]

        q_idx = starts[:, None] + q_pos[None, :]
        k_idx = starts[:, None] + k_pos[None, :]
        safe_q_idx = torch.where(q_valid, q_idx, torch.zeros_like(q_idx))
        safe_k_idx = torch.where(k_valid, k_idx, torch.zeros_like(k_idx))

        q_blk = q[0].index_select(0, safe_q_idx.reshape(-1)).reshape(Bdoc, q_end - q_start, N, D).transpose(1, 2)
        k_blk = k[0].index_select(0, safe_k_idx.reshape(-1)).reshape(Bdoc, k_end - k_start, N, D).transpose(1, 2)
        v_blk = v[0].index_select(0, safe_k_idx.reshape(-1)).reshape(Bdoc, k_end - k_start, N, D).transpose(1, 2)

        logits = torch.matmul(q_blk, k_blk.transpose(-1, -2)) * scaling

        causal = k_pos[None, :] <= q_pos[:, None]
        win_mask = (q_pos[:, None] - k_pos[None, :]) < window
        mask = (causal & win_mask)[None, None, :, :]
        mask = mask & q_valid[:, None, :, None] & k_valid[:, None, None, :]

        probs = logits.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)

        pw1 = post_w1[0].index_select(0, safe_q_idx.reshape(-1)).reshape(Bdoc, q_end - q_start, N)
        pw1 = pw1 * q_valid[..., None].to(dtype=pw1.dtype)
        post_hidden = torch.einsum("bnck,bcn->bck", probs, pw1)
        pw2 = post_w2[0].index_select(0, safe_q_idx.reshape(-1)).reshape(Bdoc, q_end - q_start, N)
        pw2 = pw2 * q_valid[..., None].to(dtype=pw2.dtype)
        probs = probs + torch.einsum("bck,bcn->bnck", post_hidden, pw2)

        out_blk = torch.matmul(probs, v_blk).transpose(1, 2)
        out_blk = out_blk * q_valid[..., None, None].to(dtype=out_blk.dtype)
        out_seq.index_add_(0, safe_q_idx.reshape(-1), out_blk.reshape(Bdoc * (q_end - q_start), N, D))

    return out_seq.transpose(0, 1).unsqueeze(0)