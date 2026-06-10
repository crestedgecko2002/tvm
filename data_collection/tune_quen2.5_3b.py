

import argparse
import os

import tvm
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--work-dir", type=str, default="tuning_logs_qwen_static")

    parser.add_argument(
        "--target",
        type=str,
        default="llvm -mcpu=skylake-avx512 -num-cores=16",
    )

    parser.add_argument("--total_trials", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)

    # Qwen-style config
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=8)

    # 🔥 important differences
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--ffn-size", type=int, default=4096)

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16"],
    )

    parser.add_argument("--skip-build", action="store_true")

    return parser.parse_args()

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_size: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Qwen-style attention projection
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)

        # Qwen-style SwiGLU FFN
        self.gate_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.down_proj = nn.Linear(ffn_size, hidden_size, bias=False)

        # Qwen uses RMSNorm, usually eps = 1e-6
        self.norm1 = nn.RMSNorm(hidden_size, -1, 1e-6, bias=False)
        self.norm2 = nn.RMSNorm(hidden_size, -1, 1e-6, bias=False)

    def forward(self, x: Tensor):
        b, s, d = x.shape

        # =====================
        # Self-attention
        # =====================
        h = self.norm1(x)

        qkv = self.qkv(h)
        qkv = op.reshape(qkv, (b, s, 3, self.num_heads, self.head_dim))
        q, k, v = op.split(qkv, 3, axis=2)

        q = op.squeeze(q, axis=2)
        k = op.squeeze(k, axis=2)
        v = op.squeeze(v, axis=2)

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

        x = x + self.out(out)

        # =====================
        # Qwen-style SwiGLU FFN
        # =====================
        h = self.norm2(x)

        gate = op.silu(self.gate_proj(h))
        up = self.up_proj(h)

        h = gate * up
        h = self.down_proj(h)

        return x + h


class StaticQwen(nn.Module):
    def __init__(self, args):
        super().__init__()

        if args.hidden_size % args.num_heads != 0:
            raise ValueError("hidden-size must be divisible by num-heads")

        self.embed = nn.Embedding(args.vocab_size, args.hidden_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    args.hidden_size,
                    args.num_heads,
                    args.ffn_size,
                )
                for _ in range(args.num_layers)
            ]
        )

        self.norm = nn.RMSNorm(args.hidden_size, -1, 1e-6, bias=False)
        self.head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def main(self, input_ids: Tensor):
        x = self.embed(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.head(x)

        return logits

    def forward(self, input_ids: Tensor):
        return self.main(input_ids)


def main():
    args = parse_args()
    target = tvm.target.Target(args.target)
    is_in_ci = os.getenv("CI", "") == "true"

    model = StaticQwen(args)
    model.to(args.dtype)

    spec = nn.spec.ModuleSpec.from_raw(
        {
            "main": {
                "input_ids": nn.spec.Tensor([1, args.seq_len], "int32"),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            }
        },
        model,
    )

    mod, params = model.export_tvm(spec=spec)

    print("[INFO] Exported Relax IR:")
    mod["main"].show()

    if not is_in_ci:
        mod = relax.get_pipeline(
            "static_shape_tuning",
            target=target,
            total_trials=args.total_trials,
            work_dir=args.work_dir,
        )(mod)

        print("[INFO] Tuning done:", args.work_dir)
        mod["main"].show()

    if args.skip_build:
        print("[INFO] --skip-build set. Exiting after tuning.")
        return

    if not is_in_ci:
        ex = relax.build(mod, target=target)
        vm = relax.VirtualMachine(ex, tvm.cpu())
        print("[INFO] VM ready")
        print("[INFO] Number of detached params:", len(params))


if __name__ == "__main__":
    main()
