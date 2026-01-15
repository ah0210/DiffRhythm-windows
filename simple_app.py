import os
import sys
import torch

print("=" * 60)
print("🎵 DiffRhythm 简化测试")
print("=" * 60)
print(f"Python版本: {sys.version}")
print(f"Python路径: {sys.path}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

# 测试Gradio导入
print("\n测试Gradio导入...")
try:
    import gradio as gr
    print(f"Gradio版本: {gr.__version__}")
    print("✓ Gradio导入成功")
except Exception as e:
    print(f"✗ Gradio导入失败: {e}")
    sys.exit(1)

# 测试模型导入
print("\n测试模型导入...")
try:
    from model import DiT, CFM
    print("✓ 模型导入成功")
except Exception as e:
    print(f"✗ 模型导入失败: {e}")
    sys.exit(1)

# 测试配置文件读取
print("\n测试配置文件读取...")
try:
    import json
    with open("./config/diffrhythm-1b.json") as f:
        model_config = json.load(f)
    print(f"✓ 配置文件读取成功")
except Exception as e:
    print(f"✗ 配置文件读取失败: {e}")
    sys.exit(1)

# 测试HF Hub导入
print("\n测试HF Hub导入...")
try:
    from huggingface_hub import hf_hub_download
    print("✓ HF Hub导入成功")
except Exception as e:
    print(f"✗ HF Hub导入失败: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有测试通过，环境正常！")
print("=" * 60)
