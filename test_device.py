import torch
import os

print("=== 设备检测测试 ===")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 设备数量: {torch.cuda.device_count()}")

# 检测 ROCm
print(f"\n=== ROCm 检测 ===")
print(f"是否有 is_rocm_available 属性: {hasattr(torch, 'is_rocm_available')}")
if hasattr(torch, 'is_rocm_available'):
    print(f"ROCm 可用: {torch.is_rocm_available()}")

# 检测 MPS
print(f"\n=== MPS 检测 ===")
print(f"是否有 mps 属性: {hasattr(torch, 'mps')}")
if hasattr(torch, 'mps'):
    print(f"MPS 可用: {torch.mps.is_available()}")

# 打印环境变量
print(f"\n=== 环境变量 ===")
env_vars = ['HSA_OVERRIDE_GFX_VERSION', 'PYTORCH_ROCM_ARCH', 'CUDA_VISIBLE_DEVICES']
for var in env_vars:
    print(f"{var}: {os.environ.get(var, '未设置')}")

print(f"\n=== 设备列表 ===")
try:
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
except Exception as e:
    print(f"获取设备名称失败: {e}")

print(f"\n=== 最终设备选择 ===")
def get_available_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch, 'is_rocm_available') and torch.is_rocm_available():
        return 'cuda'
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

device = get_available_device()
print(f"选择的设备: {device}")
