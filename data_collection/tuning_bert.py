#!/usr/bin/env python3

"""
Static BERT-style model for TVM MetaSchedule tuning.

This script does NOT load the real Hugging Face BERT model or pretrained weights.
Instead, it locally constructs a BERT-like encoder-only Transformer using TVM Relax frontend.

Purpose:
- Generate BERT-style workloads
- Run TVM MetaSchedule tuning
- Save tuning logs for later latency / power / frequency dataset extraction

Important:
- The weights are placeholder parameters.
- For compiler tuning and power measurement, tensor shapes and operator structure matter more than real pretrained values.
- This is closer to BERT pretraining / masked-language-model inference than supervised fine-tuning with a dataset.
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

    parser.add_argument("--work-dir", type=str, default="tuning_logs_bert")
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -mtriple=x86_64-linux-gnu -mcpu=skylake-avx512 -num-cores=16",
    )

    parser.add_argument("--total_trials", type=int, default=20000)

    # BERT-base-like default configuration
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=30522)
    parser.add_argument("--ffn-size", type=int, default=3072)
    parser.add_argument("--type-vocab-size", type=int, default=2)

    # For BERT fine-tuning classification head.
    # Set --task classification to use this instead of MLM output.
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument(
        "--task",
        type=str,
        default="mlm",
        choices=["mlm", "classification"],
        help="mlm: BERT masked-language-model style output; classification: [CLS] classification head",
    )

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

def bert_activation(x: Tensor):
    """
    BERT normally uses GELU.
    Some TVM versions may not expose op.gelu in the Relax frontend.
    If GELU is not available, use SiLU as a fallback so tuning can still run.
    """
    if hasattr(op, "gelu"):
        return op.gelu(x)
    return op.silu(x)


# ============================================================
# BERT Transformer Encoder Block
# ============================================================

class BERTEncoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_size: int):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # BERT uses LayerNorm after each residual path.
        # Do NOT pass bias=True here because some TVM nn.LayerNorm versions
        # do not support the bias keyword.
        self.attn_ln = nn.LayerNorm(hidden_size, -1, 1e-5)
        self.ffn_ln = nn.LayerNorm(hidden_size, -1, 1e-5)

        # BERT-style separate Q, K, V projections.
        # This differs from the GPT-2 script, which uses a fused QKV projection.
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # Feed-forward network: hidden -> 4 hidden -> hidden
        self.fc1 = nn.Linear(hidden_size, ffn_size, bias=True)
        self.fc2 = nn.Linear(ffn_size, hidden_size, bias=True)

    def forward(self, x: Tensor):
        b, s, d = x.shape

        # -------------------------------
        # Bidirectional self-attention
        # -------------------------------
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = op.reshape(q, (b, s, self.num_heads, self.head_dim))
        k = op.reshape(k, (b, s, self.num_heads, self.head_dim))
        v = op.reshape(v, (b, s, self.num_heads, self.head_dim))

        # [B, S, H, D] -> [B, H, S, D]
        q = op.permute_dims(q, axes=[0, 2, 1, 3])
        k = op.permute_dims(k, axes=[0, 2, 1, 3])
        v = op.permute_dims(v, axes=[0, 2, 1, 3])

        # [B, H, S, D] x [B, H, D, S] -> [B, H, S, S]
        kt = op.permute_dims(k, axes=[0, 1, 3, 2])

        scale = self.head_dim ** 0.5
        attn = op.matmul(q, kt) / scale

        # BERT is bidirectional, so there is no causal mask.
        # For compiler tuning, this keeps the main BERT kernels:
        # Q/K/V projections, attention matmul, softmax, attention-value matmul, FFN.
        attn = op.softmax(attn, axis=-1)

        out = op.matmul(attn, v)

        # [B, H, S, D] -> [B, S, H, D] -> [B, S, hidden]
        out = op.permute_dims(out, axes=[0, 2, 1, 3])
        out = op.reshape(out, (b, s, d))

        # BERT-style post-LN residual block
        x = self.attn_ln(x + self.out_proj(out))

        # -------------------------------
        # Feed-forward block
        # -------------------------------
        h = self.fc1(x)
        h = bert_activation(h)
        h = self.fc2(h)

        x = self.ffn_ln(x + h)

        return x


# ============================================================
# Static BERT Model
# ============================================================

class StaticBERT(nn.Module):
    def __init__(self, args):
        super().__init__()

        if args.hidden_size % args.num_heads != 0:
            raise ValueError("hidden-size must be divisible by num-heads")

        self.task = args.task
        self.hidden_size = args.hidden_size
        self.seq_len = args.seq_len

        self.token_embed = nn.Embedding(args.vocab_size, args.hidden_size)
        self.pos_embed = nn.Embedding(args.seq_len, args.hidden_size)
        self.type_embed = nn.Embedding(args.type_vocab_size, args.hidden_size)

        self.embed_ln = nn.LayerNorm(args.hidden_size, -1, 1e-5)

        self.layers = nn.ModuleList(
            [
                BERTEncoderBlock(
                    hidden_size=args.hidden_size,
                    num_heads=args.num_heads,
                    ffn_size=args.ffn_size,
                )
                for _ in range(args.num_layers)
            ]
        )

        # MLM head: hidden -> hidden -> vocab
        self.mlm_dense = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
        self.mlm_ln = nn.LayerNorm(args.hidden_size, -1, 1e-5)
        self.mlm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=True)

        # Classification head: [CLS] hidden -> labels
        # For BERT fine-tuning style workloads such as sentiment or intent classification.
        self.cls_head = nn.Linear(args.hidden_size, args.num_labels, bias=True)

    def encode(self, input_ids: Tensor, token_type_ids: Tensor, position_ids: Tensor):
        tok = self.token_embed(input_ids)
        typ = self.type_embed(token_type_ids)
        pos = self.pos_embed(position_ids)

        x = self.embed_ln(tok + typ + pos)

        for layer in self.layers:
            x = layer(x)

        return x

    def main(self, input_ids: Tensor, token_type_ids: Tensor, position_ids: Tensor):
        x = self.encode(input_ids, token_type_ids, position_ids)

        if self.task == "classification":
            # Approximate [CLS] extraction by taking the first sequence position.
            # Depending on your TVM frontend version, op.take may be unavailable,
            # so strided_slice is used when possible.
            if hasattr(op, "strided_slice"):
                cls = op.strided_slice(x, axes=[1], begin=[0], end=[1])
                cls = op.reshape(cls, (x.shape[0], x.shape[2]))
            else:
                # Fallback: use mean-like workload replacement by reshaping first token is not available.
                # This keeps the classification head path compilable in older frontends only if
                # strided_slice exists. Raise clearly otherwise.
                raise RuntimeError("op.strided_slice is required for classification mode")

            logits = self.cls_head(cls)
            return logits

        # Default: masked language modeling style output
        h = self.mlm_dense(x)
        h = bert_activation(h)
        h = self.mlm_ln(h)
        logits = self.mlm_head(h)
        return logits

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, position_ids: Tensor):
        return self.main(input_ids, token_type_ids, position_ids)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    target = tvm.target.Target(args.target)
    is_in_ci = os.getenv("CI", "") == "true"

    print("=" * 70)
    print("[INFO] Static BERT TVM tuning")
    print("=" * 70)
    print(f"[INFO] work_dir        = {args.work_dir}")
    print(f"[INFO] target          = {args.target}")
    print(f"[INFO] total_trials    = {args.total_trials}")
    print(f"[INFO] task            = {args.task}")
    print(f"[INFO] batch_size      = {args.batch_size}")
    print(f"[INFO] seq_len         = {args.seq_len}")
    print(f"[INFO] hidden_size     = {args.hidden_size}")
    print(f"[INFO] num_heads       = {args.num_heads}")
    print(f"[INFO] num_layers      = {args.num_layers}")
    print(f"[INFO] vocab_size      = {args.vocab_size}")
    print(f"[INFO] ffn_size        = {args.ffn_size}")
    print(f"[INFO] type_vocab_size = {args.type_vocab_size}")
    print(f"[INFO] num_labels      = {args.num_labels}")
    print(f"[INFO] dtype           = {args.dtype}")
    print("=" * 70)

    model = StaticBERT(args)
    model.to(args.dtype)

    spec = nn.spec.ModuleSpec.from_raw(
        {
            "main": {
                "input_ids": nn.spec.Tensor(
                    [args.batch_size, args.seq_len],
                    "int32",
                ),
                "token_type_ids": nn.spec.Tensor(
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

    print("[INFO] Exported BERT-style Relax IR:")
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
