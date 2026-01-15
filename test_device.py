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

# 测试MPS是否可用
print("\n=== MPS 检测 ===")
print(f"是否有 mps 属性: {hasattr(torch, 'mps')}")
if hasattr(torch, 'mps'):
    # 检查mps是否有is_available方法
    if hasattr(torch.mps, 'is_available'):
        print(f"MPS 可用: {torch.mps.is_available()}")
    else:
        print(f"MPS 可用: False (is_available方法不存在)")

# 测试DirectML是否可用
print("\n=== DirectML 检测 ===")
directml_available = False
directml_device = None
try:
    import torch_directml as dml
    print("DirectML 模块已安装")
    # 修复：直接使用device_count()，不进行比较
    device_count = dml.device_count()
    print(f"DirectML 设备数量: {device_count}")
    for i in range(device_count):
        dev = dml.device(i)
        print(f"设备 {i}: {dev}")
    # 测试创建DirectML设备
    directml_device = dml.device(0)
    print(f"默认DirectML设备: {directml_device}")
    directml_available = True
    # 测试简单计算
    x = torch.tensor([1.0, 2.0, 3.0], device=directml_device)
    y = x * 2
    print(f"DirectML 计算测试: {y}")
except ImportError:
    print("DirectML 模块未安装")
except Exception as e:
    print(f"DirectML 检测失败: {e}")

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
    elif hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available():
        return 'mps'
    elif directml_available:
        return directml_device
    else:
        return 'cpu'

device = get_available_device()
print(f"选择的设备: {device}")

# 测试设备使用
try:
    # 在所选设备上执行简单计算
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = x * 2
    print(f"设备计算测试: {y}")
    print("设备使用成功!")
except Exception as e:
    print(f"设备使用失败: {e}")
