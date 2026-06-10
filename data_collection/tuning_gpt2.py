#!/usr/bin/env python3

"""
Static GPT-2-style model for TVM MetaSchedule tuning.

This script does NOT load the real Hugging Face GPT-2 model or pretrained weights.
Instead, it locally constructs a GPT-2-like decoder-only Transformer using TVM Relax frontend.

Purpose:
- Generate GPT-2-style workloads
- Run TVM MetaSchedule tuning
- Save tuning logs for later latency / power / frequency dataset extraction

Important:
- The weights are placeholder parameters.
- For compiler tuning and power measurement, the tensor shapes and operator structure matter more than real pretrained values.
"""

import argparse
import os

import tvm
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--work-dir", type=str, default="tuning_logs_gpt2")
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -mtriple=x86_64-linux-gnu -mcpu=skylake-avx512 -num-cores=16",
    )

    parser.add_argument("--total_trials", type=int, default=20000)

    # GPT-2-small-like default configuration
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--ffn-size", type=int, default=3072)

    # On CPU, float32 is usually safer.
    # float16 can be tested, but CPU float16 may not always be faster.
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
    )

    parser.add_argument("--skip-build", action="store_true")

    return parser.parse_args()


# ============================================================
# Activation
# ============================================================

def gpt2_activation(x: Tensor):
    """
    GPT-2 normally uses GELU.
    Some TVM versions may not expose op.gelu in the Relax frontend.
    If GELU is not available, use SiLU as a fallback so tuning can still run.
    """
    if hasattr(op, "gelu"):
        return op.gelu(x)
    return op.silu(x)


# ============================================================
# GPT-2 Transformer Block
# ============================================================

class GPT2Block(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_size: int):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # GPT-2 uses LayerNorm.
        # IMPORTANT:
        # Do NOT pass bias=True here.
        # Your TVM version's nn.LayerNorm does not support the bias keyword.
        self.ln1 = nn.LayerNorm(hidden_size, -1, 1e-5)
        self.ln2 = nn.LayerNorm(hidden_size, -1, 1e-5)

        # GPT-2-style projections.
        # Linear supports bias in your TVM frontend, unlike LayerNorm.
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # MLP: hidden -> 4 hidden -> hidden
        self.fc1 = nn.Linear(hidden_size, ffn_size, bias=True)
        self.fc2 = nn.Linear(ffn_size, hidden_size, bias=True)

    def forward(self, x: Tensor):
        b, s, d = x.shape

        # -------------------------------
        # Self-attention block
        # -------------------------------
        h = self.ln1(x)

        qkv = self.qkv(h)
        qkv = op.reshape(qkv, (b, s, 3, self.num_heads, self.head_dim))

        q, k, v = op.split(qkv, 3, axis=2)

        q = op.squeeze(q, axis=2)
        k = op.squeeze(k, axis=2)
        v = op.squeeze(v, axis=2)

        # [B, S, H, D] -> [B, H, S, D]
        q = op.permute_dims(q, axes=[0, 2, 1, 3])
        k = op.permute_dims(k, axes=[0, 2, 1, 3])
        v = op.permute_dims(v, axes=[0, 2, 1, 3])

        # [B, H, S, D] x [B, H, D, S] -> [B, H, S, S]
        kt = op.permute_dims(k, axes=[0, 1, 3, 2])

        scale = self.head_dim ** 0.5
        attn = op.matmul(q, kt) / scale

        # For compiler tuning, we omit causal masking.
        # The major GPT-2 compute kernels are still represented:
        # QKV projection, attention matmul, softmax, attention-value matmul, MLP.
        attn = op.softmax(attn, axis=-1)

        out = op.matmul(attn, v)

        # [B, H, S, D] -> [B, S, H, D] -> [B, S, hidden]
        out = op.permute_dims(out, axes=[0, 2, 1, 3])
        out = op.reshape(out, (b, s, d))

        x = x + self.out_proj(out)

        # -------------------------------
        # MLP block
        # -------------------------------
        h = self.ln2(x)
        h = self.fc1(h)
        h = gpt2_activation(h)
        h = self.fc2(h)

        x = x + h

        return x


# ============================================================
# Static GPT-2 Model
# ============================================================

class StaticGPT2(nn.Module):
    def __init__(self, args):
        super().__init__()

        if args.hidden_size % args.num_heads != 0:
            raise ValueError("hidden-size must be divisible by num-heads")

        self.token_embed = nn.Embedding(args.vocab_size, args.hidden_size)
        self.pos_embed = nn.Embedding(args.seq_len, args.hidden_size)

        self.layers = nn.ModuleList(
            [
                GPT2Block(
                    hidden_size=args.hidden_size,
                    num_heads=args.num_heads,
                    ffn_size=args.ffn_size,
                )
                for _ in range(args.num_layers)
            ]
        )

        # Final GPT-2 LayerNorm.
        # Again, no bias=True because this TVM LayerNorm API does not support it.
        self.ln_f = nn.LayerNorm(args.hidden_size, -1, 1e-5)

        # Language modeling head.
        # This is the expensive final projection:
        # [B, S, hidden] x [hidden, vocab]
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def main(self, input_ids: Tensor, position_ids: Tensor):
        tok = self.token_embed(input_ids)
        pos = self.pos_embed(position_ids)

        x = tok + pos

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def forward(self, input_ids: Tensor, position_ids: Tensor):
        return self.main(input_ids, position_ids)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    target = tvm.target.Target(args.target)
    is_in_ci = os.getenv("CI", "") == "true"

    print("=" * 70)
    print("[INFO] Static GPT-2 TVM tuning")
    print("=" * 70)
    print(f"[INFO] work_dir     = {args.work_dir}")
    print(f"[INFO] target       = {args.target}")
    print(f"[INFO] total_trials = {args.total_trials}")
    print(f"[INFO] batch_size   = {args.batch_size}")
    print(f"[INFO] seq_len      = {args.seq_len}")
    print(f"[INFO] hidden_size  = {args.hidden_size}")
    print(f"[INFO] num_heads    = {args.num_heads}")
    print(f"[INFO] num_layers   = {args.num_layers}")
    print(f"[INFO] vocab_size   = {args.vocab_size}")
    print(f"[INFO] ffn_size     = {args.ffn_size}")
    print(f"[INFO] dtype        = {args.dtype}")
    print("=" * 70)

    model = StaticGPT2(args)
    model.to(args.dtype)

    spec = nn.spec.ModuleSpec.from_raw(
        {
            "main": {
                "input_ids": nn.spec.Tensor(
                    [args.batch_size, args.seq_len],
                    "int32",
                ),
                "position_ids": nn.spec.Tensor(
                    [args.batch_size, args.seq_len],
                    "int32",
                ),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            }
        },
        model,
    )

    mod, params = model.export_tvm(spec=spec)

    print("[INFO] Exported GPT-2-style Relax IR:")
    mod["main"].show()

    if not is_in_ci:
        print("[INFO] Starting MetaSchedule tuning...")

        mod = relax.get_pipeline(
            "static_shape_tuning",
            target=target,
            total_trials=args.total_trials,
            work_dir=args.work_dir,
        )(mod)

        print("[INFO] Tuning done:", args.work_dir)
        print("[INFO] Tuned Relax IR:")
        mod["main"].show()

    if args.skip_build:
        print("[INFO] --skip-build set. Exiting after tuning.")
        return

    if not is_in_ci:
        print("[INFO] Building tuned module...")

        ex = relax.build(mod, target=target)
        vm = relax.VirtualMachine(ex, tvm.cpu())

        print("[INFO] VM ready")
        print("[INFO] Number of detached params:", len(params))


if __name__ == "__main__":
    main()