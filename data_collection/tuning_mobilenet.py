# ============================================================
# 0. Imports
# ============================================================
import os
import argparse
import numpy as np
import torch
import tvm

from torch.export import export
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# ============================================================
# 1. Argument Parsing
#    - 실험 설정값 정의
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    # tuning log 저장 폴더
    parser.add_argument("--work-dir", type=str, default="tuning_logs_mobilenet")

    # target device / compilation target
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -mtriple=x86_64-linux-gnu -mcpu=skylake-avx512 -num-cores=16",
    )

    # tuning trial 수
    parser.add_argument("--total_trials", type=int, default=128)

    # batch size
    parser.add_argument("--batch-size", type=int, default=1)

    # image size (MobileNet 기본은 224)
    parser.add_argument("--image-size", type=int, default=224)

 

    return parser.parse_args()


# ============================================================
# 2. Load PyTorch MobileNetV2
#    - pretrained MobileNetV2 모델 로드
# ============================================================
def load_model():
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
    print("Loaded PyTorch MobileNetV2 (pretrained)")
    return model


# ============================================================
# 3. Create Example Input
#    - torch.export에 사용할 example input 생성
# ============================================================
def make_example_input(batch_size, image_size):
    x = torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32)
    print("Example input shape:", tuple(x.shape))
    return (x,)


# ============================================================
# 4. Convert PyTorch -> ExportedProgram -> Relax IR
#    - MobileNet을 TVM Relax IRModule로 변환
# ============================================================
def convert_to_relax(torch_model, example_args):
    with torch.no_grad():
        exported_program = export(torch_model, example_args)
        mod = from_exported_program(
            exported_program,
            keep_params_as_input=True,
        )

    # params를 mod에서 분리
    mod, params = relax.frontend.detach_params(mod)

    print("\n=== Imported Relax IR ===")
    mod["main"].show()

    return mod, params


# ============================================================
# 5. Apply TVM Tuning Pipeline
#    - static_shape_tuning pipeline 적용
# ============================================================
def apply_tuning(mod, target, total_trials):
    print("\n=== Running Tuning Pipeline ===")

    tuned_mod = relax.get_pipeline(
        "static_shape_tuning",
        target=target,
        total_trials=total_trials,
        work_dir=work_dir,
    )(mod)


    print("\n=== Tuned Relax IR ===")
    tuned_mod["main"].show()

    return tuned_mod


# ============================================================
# 6. Build TVM Module
#    - Relax IRModule을 실행 가능한 module로 build
# ============================================================
def build_module(mod, target):
    print("\n=== Building Module ===")
    ex = relax.build(mod, target=target)
    return ex


# ============================================================
# 7. Make Random Input for Runtime
#    - TVM runtime에서 사용할 랜덤 입력 생성
# ============================================================
def make_runtime_input(batch_size, image_size, dev):
    data = np.random.rand(batch_size, 3, image_size, image_size).astype("float32")
    return tvm.nd.array(data, dev)


# ============================================================
# 8. Prepare Parameters
#    - detach_params 이후 params["main"]을 TVM NDArray로 변환
# ============================================================
def prepare_params(params, dev):
    print("params type:", type(params))

    if hasattr(params, "keys"):
        print("params keys:", list(params.keys()))

    if not isinstance(params, dict):
        raise TypeError(f"Unexpected params type: {type(params)}")

    if "main" not in params:
        raise ValueError(f"Unexpected params structure. keys={list(params.keys())}")

    tvm_params = [tvm.nd.array(p, dev) for p in params["main"]]
    return tvm_params


# ============================================================
# 9. Run Inference
#    - VM 생성 후 실행
# ============================================================
def run_inference(ex, params, batch_size, image_size):
    print("\n=== Running Inference ===")

    dev = tvm.device("cpu", 0)
    vm = relax.VirtualMachine(ex, dev)

    cpu_data = make_runtime_input(batch_size, image_size, dev)
    cpu_params = prepare_params(params, dev)

    res = vm["main"](cpu_data, *cpu_params)

    print("result type:", type(res))

    if isinstance(res, tvm.ir.container.Array):
        print("num outputs:", len(res))
        for i, x in enumerate(res):
            try:
                print(f"output[{i}] shape =", x.numpy().shape)
            except Exception:
                print(f"output[{i}] type =", type(x))
    else:
        print("output shape:", res.numpy().shape)

    # output이 Array면 첫 번째 output 추출
    if isinstance(res, tvm.ir.container.Array):
        res = res[0]

    out = res.numpy()
    print("final output shape:", out.shape)


# ============================================================
# 10. Main Pipeline
#     전체 실행 흐름
# ============================================================
def main():
    args = parse_args()

    target = tvm.target.Target(args.target)
    is_in_ci = os.getenv("CI", "") == "true"

    # 1) PyTorch model load
    torch_model = load_model()

    # 2) Example input 생성
    example_args = make_example_input(args.batch_size, args.image_size)

    # 3) PyTorch -> Relax 변환
    mod, params = convert_to_relax(torch_model, example_args)

    # 4) Tuning
    if not is_in_ci:
        mod = apply_tuning(mod, target, args.total_trials)

    # 5) Build + Run
    if not is_in_ci:
        ex = build_module(mod, target)
        run_inference(ex, params, args.batch_size, args.image_size)


# ============================================================
# 11. Entry Point
# ============================================================
if __name__ == "__main__":
    main()