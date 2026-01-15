import torch
import sys
import os
from infer.infer_utils import prepare_model, get_style_prompt, get_lrc_token, decode_audio, get_negative_style_prompt, get_reference_latent
from model import CFM
import json
import time

# 添加当前目录到路径
sys.path.append(os.getcwd())

print("=== 测试推理流程 ===")
print(f"Python 版本: {sys.version}")
print(f"PyTorch 版本: {torch.__version__}")

# 检测可用设备
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
print(f"选择设备: {device}")

# 测试参数
max_frames = 2048  # 最小参数
steps = 10  # 最小扩散步数
repo_id = "ASLP-lab/DiffRhythm-base"

print(f"\n=== 加载模型 ===")
try:
    # 加载模型
    cfm, tokenizer, muq, vae = prepare_model(max_frames, device, repo_id=repo_id)
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n=== 测试风格提示生成 ===")
try:
    # 使用文本提示生成风格嵌入
    text_prompt = "Electronic Dance Music"
    style_prompt = get_style_prompt(muq, prompt=text_prompt)
    print(f"✅ 风格提示生成成功，形状: {style_prompt.shape}")
except Exception as e:
    print(f"❌ 风格提示生成失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n=== 测试歌词处理 ===")
try:
    # 简单的歌词测试
    test_lrc = "[00:04.34]Tell me that I'm special\n[00:06.57]Tell me I look pretty"
    lrc_emb, start_time = get_lrc_token(max_frames, test_lrc, tokenizer, device)
    print(f"✅ 歌词处理成功，形状: {lrc_emb.shape}")
except Exception as e:
    print(f"❌ 歌词处理失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n=== 测试参考潜在空间生成 ===")
try:
    latent_prompt = get_reference_latent(device, max_frames)
    print(f"✅ 参考潜在空间生成成功，形状: {latent_prompt.shape}")
except Exception as e:
    print(f"❌ 参考潜在空间生成失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n=== 测试负向风格提示生成 ===")
try:
    negative_style_prompt = get_negative_style_prompt(device)
    print(f"✅ 负向风格提示生成成功，形状: {negative_style_prompt.shape}")
except Exception as e:
    print(f"❌ 负向风格提示生成失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n=== 测试模型推理（简化版） ===")
try:
    # 设置简化的推理参数
    duration = max_frames
    
    # 测试 CFM 模型的 sample 方法
    print(f"🔄 开始 CFM 模型采样，步骤数: {steps}")
    sample_start_time = time.time()
    
    # 使用简化的进度回调
    def simple_progress_callback(step, total_steps, progress_ratio):
        print(f"   进度: {step}/{total_steps} ({progress_ratio*100:.1f}%)")
    
    # 执行采样
    generated, _ = cfm.sample(
        cond=latent_prompt,
        text=lrc_emb,
        duration=duration,
        style_prompt=style_prompt,
        negative_style_prompt=negative_style_prompt,
        steps=steps,
        cfg_strength=4.0,
        start_time=start_time,
        progress_callback=simple_progress_callback,
    )
    
    sample_time = time.time() - sample_start_time
    print(f"✅ CFM 模型采样成功，耗时: {sample_time:.2f} 秒")
    print(f"   生成结果形状: {generated.shape}")
except Exception as e:
    print(f"❌ CFM 模型采样失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n=== 测试音频解码 ===")
try:
    # 转换数据类型
    generated = generated.to(torch.float32)
    latent = generated.transpose(1, 2)  # [b d t]
    
    # 测试解码
    decode_start_time = time.time()
    output = decode_audio(latent, vae, chunked=True)
    decode_time = time.time() - decode_start_time
    print(f"✅ 音频解码成功，耗时: {decode_time:.2f} 秒")
    print(f"   输出形状: {output.shape}")
except Exception as e:
    print(f"❌ 音频解码失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n=== 测试完成 ===")
print("🎉 所有测试通过！项目逻辑正常")
