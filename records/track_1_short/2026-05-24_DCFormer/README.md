# Post-only DC correction

Add a lightweight DC-inspired correction path to layer 10 attention, based on DCFormer paper:  
**[Improving Transformers with Dynamically Composable Multi-Head Attention](https://arxiv.org/abs/2405.08553)**.

The original DCFormer applies dynamic composition inside attention itself: DC parameters modify attention logits before softmax (`pre-DC`) and attention probabilities after softmax (`post-DC`). That means the normal attention kernel has to be replaced by a DC-aware attention kernel.

In early experiments, the full DC attention replacement did improve loss, but the overhead was too large. Even after writing custom Triton kernels, the H100 overhead was still around `6x` vs the FA3 window-attention path. Adding DC on layer 10 gave roughly `0.01` loss improvement, but the extra attention cost was too high to recover just by cutting training steps.

This PR keeps FA3 as the base attention and adds a much cheaper **post-only no-DD DC correction**:

```python
out = fa3_base_attention(q, k, v) + dc_correction(q, k, v, post_w1, post_w2)
```

The correction keeps the useful post-DC `w1/w2` path, drops `pre-DC` and `DD`, and is implemented with a specialized Triton forward/backward kernel.

# Result

```bash
                 Runs  Steps   Time μ   Time σ  Time +/-   Time p   Loss μ   Loss σ  Loss +/-   Loss p
  baseline-1415    10   1415  83.2430   0.0676    0.0000      nan   3.2785   0.0011    0.0000    0.0034
  baseline-1380    10   1380  81.2611   0.9264   -0.9819   0.0000   3.2823   0.0009   +0.0038    1.0000
     this_pr_v1    10   1380  82.9008   0.1009   -0.3472   0.0000   3.2783   0.0013   -0.0002    0.0013

```

## Formulation

Standard attention:

```python
logits = q @ k.transpose(-1, -2) * scale
probs = softmax(mask(logits))
out = probs @ v
```

The original DCformer-style formulation modifies the attention itself:

```python
logits = q @ k.transpose(-1, -2) * scale

# pre-DC: dynamic correction before softmax
logits = logits + dc_pre(q, k)

probs = softmax(mask(logits))

# post-DC / DD: dynamic composition after softmax
probs = dc_post(probs, q)

out = probs @ v
```

This is expressive, but expensive because the whole attention path becomes custom.

This PR instead leaves the base attention unchanged:

```python
base = flash_attn_varlen_func(
    q, k, v,
    causal=True,
    window_size=(window, 0),
    softmax_scale=scale,
)
```

Then computes only a post-only correction. For each query token, each source head first forms its own local attention distribution:

```python
p_h = softmax(q_h @ k_h.transpose(-1, -2) * scale)
```

The learned `post_w1` mixes the source-head probability maps into one shared correction map:

```python
a = sum(post_w1[..., h] * p_h for h in range(num_heads))
```

Then each output head applies the shared correction to its own value head and scales it with `post_w2`:

```python
corr_h = post_w2[..., h] * (a @ v_h)
out_h = base_h + corr_h
```

So the effective attention path is:

```python
out_h = FA3(q_h, k_h, v_h) + post_w2_h * (
    sum_h2(post_w1_h2 * softmax(q_h2 @ k_h2.T * scale)) @ v_h
)
```

This is not a full DCformer reproduction. It is a distilled DC correction path adapted to the FA3-window training setup. For related torch information, please see dc_torch_reference.py

## Implementation

- **Only layer 10 uses DC correction.** Earlier layers keep the existing FA3 attention path.
- **FA3 remains the base attention.** The Triton kernel only computes the additive correction.
- **Post-only no-DD.** The implementation removes `pre-DC` and `DD`, keeping only `post_w1` and `post_w2`.
- **Document boundaries are respected.** The correction receives `cu_seqlens` and masks out cross-document attention.
- **Specialized Triton forward/backward.** The kernel is specialized for the validated path: `H=6`, `D=128`, `BM=16`, and the current DC window.
- **No padding over documents.** Like FA3 varlen attention, the correction uses document boundaries instead of padding all documents into a dense batch.

## Why base + correction?

The full DC path replaces attention:

```python
out = DC_attention(q, k, v)
```

That gave useful loss, but made the most expensive part of the model custom and much slower.

The current path keeps the optimized attention kernel:

```python
base = FA3_attention(q, k, v)
corr = DC_postonly_correction(q, k, v)
out = base + corr
```

This keeps FA3 on the critical base path and makes DC a smaller learned residual correction. Empirically this kept most of the useful loss improvement while making the overhead tractable.

## Optimization history

1. Started from the DCformer formulation with both `pre-DC` and `post-DC`.
2. Replaced attention with a full DC attention path.
3. Wrote Triton kernels for the full DC path.
4. Found that full replacement still had around `6x` H100 overhead vs FA3-window attention.
5. Switched to `FA3 base + DC correction`.
6. Removed `pre-DC`.`DD,`kept only post-only `w1/w2`.
7. Specialized the Triton correction forward/backward for the validated layer/window/head shape.

The current Triton implementation may still be far from optimal. I am not especially experienced with Triton kernel optimization, so there may be better ways to reduce the DC correction overhead beyond the optimizations tried here. If the correction kernel can be made cheaper, it may also be worth revisiting fuller DC variants, since the early full-DC experiments showed a larger raw loss improvement before overhead became the bottleneck.

## Notes

Timing was validated on H100.