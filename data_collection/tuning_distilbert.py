#!/usr/bin/env python3

"""
Static DistilBERT-style text-classification workload for TVM MetaSchedule tuning.

This script does NOT download Hugging Face DistilBERT weights or run a tokenizer.
Instead, it locally constructs a DistilBERT-like encoder using TVM Relax frontend.

Purpose:
- Generate a lightweight encoder-only Transformer workload
- Tune the workload with TVM MetaSchedule
- Save tuning logs for later latency / power / frequency dataset extraction

Default architecture:
- 6 Transformer encoder layers
- hidden size: 768
- attention heads: 12
- FFN size: 3072
- vocabulary size: 30522
- text-classification head using the first token representation

This is best described as a "DistilBERT-style classification workload",
not as an exact pretrained Hugging Face checkpoint.
"""

import argparse
import os

import tvm
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--work-dir", type=str, default="tuning_logs_distilbert")
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -mtriple=x86_64-linux-gnu -mcpu=skylake-avx512 -num-cores=16",
    )
    parser.add_argument("--total_trials", type=int, default=20000)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=30522)
    parser.add_argument("--ffn-size", type=int, default=3072)
    parser.add_argument("--num-labels", type=int, default=2)

    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
    )

    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def distilbert_ffn_activation(x: Tensor):
    """DistilBERT normally uses GELU; SiLU is a fallback for older TVM builds."""
    if hasattr(op, "gelu"):
        return op.gelu(x)
    return op.silu(x)


def classification_activation(x: Tensor):
    """DistilBERT classification head normally uses ReLU."""
    if hasattr(op, "relu"):
        return op.relu(x)
    return op.silu(x)


class DistilBERTEncoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_size: int):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.ffn1 = nn.Linear(hidden_size, ffn_size, bias=True)
        self.ffn2 = nn.Linear(ffn_size, hidden_size, bias=True)

        # Do not pass bias=True: your TVM LayerNorm API does not support it.
        self.attn_ln = nn.LayerNorm(hidden_size, -1, 1e-5)
        self.ffn_ln = nn.LayerNorm(hidden_size, -1, 1e-5)

    def forward(self, x: Tensor):
        b, s, d = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = op.reshape(q, (b, s, self.num_heads, self.head_dim))
        k = op.reshape(k, (b, s, self.num_heads, self.head_dim))
        v = op.reshape(v, (b, s, self.num_heads, self.head_dim))

        q = op.permute_dims(q, axes=[0, 2, 1, 3])
        k = op.permute_dims(k, axes=[0, 2, 1, 3])
        v = op.permute_dims(v, axes=[0, 2, 1, 3])

        kt = op.permute_dims(k, axes=[0, 1, 3, 2])

        scale = self.head_dim ** 0.5
        attn = op.matmul(q, kt) / scale
        attn = op.softmax(attn, axis=-1)

        out = op.matmul(attn, v)
        out = op.permute_dims(out, axes=[0, 2, 1, 3])
        out = op.reshape(out, (b, s, d))

        x = self.attn_ln(x + self.out_proj(out))

        h = self.ffn1(x)
        h = distilbert_ffn_activation(h)
        h = self.ffn2(h)

        return self.ffn_ln(x + h)


class StaticDistilBERT(nn.Module):
    def __init__(self, args):
        super().__init__()

        if args.hidden_size % args.num_heads != 0:
            raise ValueError("hidden-size must be divisible by num-heads")

        self.token_embed = nn.Embedding(args.vocab_size, args.hidden_size)
        self.pos_embed = nn.Embedding(args.seq_len, args.hidden_size)

        # DistilBERT does not use token-type embeddings.
        self.embed_ln = nn.LayerNorm(args.hidden_size, -1, 1e-5)

        self.layers = nn.ModuleList(
            [
                DistilBERTEncoderBlock(
                    hidden_size=args.hidden_size,
                    num_heads=args.num_heads,
                    ffn_size=args.ffn_size,
                )
                for _ in range(args.num_layers)
            ]
        )

        self.pre_classifier = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
        self.classifier = nn.Linear(args.hidden_size, args.num_labels, bias=True)

    def encode(self, input_ids: Tensor, position_ids: Tensor):
        x = self.embed_ln(
            self.token_embed(input_ids) + self.pos_embed(position_ids)
        )

        for layer in self.layers:
            x = layer(x)

        return x

    def main(self, input_ids: Tensor, position_ids: Tensor):
        x = self.encode(input_ids, position_ids)

        if not hasattr(op, "strided_slice"):
            raise RuntimeError(
                "op.strided_slice is required for DistilBERT classification mode"
            )

        cls = op.strided_slice(x, axes=[1], begin=[0], end=[1])
        cls = op.reshape(cls, (x.shape[0], x.shape[2]))

        h = self.pre_classifier(cls)
        h = classification_activation(h)
        return self.classifier(h)

    def forward(self, input_ids: Tensor, position_ids: Tensor):
        return self.main(input_ids, position_ids)


def main():
    args = parse_args()

    target = tvm.target.Target(args.target)
    is_in_ci = os.getenv("CI", "") == "true"

    print("=" * 70)
    print("[INFO] Static DistilBERT-style TVM tuning")
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
    print(f"[INFO] num_labels   = {args.num_labels}")
    print(f"[INFO] dtype        = {args.dtype}")
    print("=" * 70)

    model = StaticDistilBERT(args)
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

    print("[INFO] Exported DistilBERT-style Relax IR:")
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
