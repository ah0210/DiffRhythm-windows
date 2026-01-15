# -*- coding: utf-8 -*-
"""
DiffRhythm 模型推理模块
"""

import os
import time
import torch
import torchaudio
from einops import rearrange

from infer.infer_utils import (
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
)


class DiffRhythmGenerator:
    """
    DiffRhythm 模型生成器类
    """
    
    def __init__(self, device='cpu'):
        """
        初始化 DiffRhythm 生成器
        
        Args:
            device: 计算设备，默认为 'cpu'
        """
        self.device = device
        self.cfm = None
        self.tokenizer = None
        self.muq = None
        self.vae = None
        self.max_frames = None
    
    def load_model(self, max_frames=2048, repo_id="ASLP-lab/DiffRhythm-base"):
        """
        加载 DiffRhythm 模型
        
        Args:
            max_frames: 最大帧数，95秒对应2048，285秒对应6144
            repo_id: 模型仓库ID
        """
        print(f"📦 正在加载 DiffRhythm 模型: {repo_id}")
        print(f"   • 最大帧数: {max_frames}")
        print(f"   • 设备: {self.device}")
        
        self.max_frames = max_frames
        self.cfm, self.tokenizer, self.muq, self.vae = prepare_model(
            max_frames, self.device, repo_id=repo_id
        )
        
        print("✅ DiffRhythm 模型加载完成")
        return self.cfm, self.vae
    
    def generate(self, lrc="", ref_prompt=None, ref_audio_path=None, chunked=False):
        """
        使用 DiffRhythm 模型生成音乐
        
        Args:
            lrc: LRC格式的歌词
            ref_prompt: 文本风格提示
            ref_audio_path: 音频风格提示路径
            chunked: 是否使用分块解码
        
        Returns:
            生成的音频数据
        """
        if self.cfm is None or self.vae is None:
            self.load_model()
        
        print("🎵 使用 DiffRhythm 生成音乐")
        print(f"   • 设备: {self.device}")
        print(f"   • 歌词长度: {len(lrc)} 字符")
        print(f"   • 风格提示: {'文本提示' if ref_prompt else '音频提示'}")
        
        # 获取LRC token
        lrc_prompt, start_time = get_lrc_token(
            self.max_frames, lrc, self.tokenizer, self.device
        )
        print(f"   • 开始时间: {start_time.item():.2f} 秒")
        
        # 获取风格提示
        if ref_audio_path:
            style_prompt = get_style_prompt(self.muq, ref_audio_path)
        else:
            style_prompt = get_style_prompt(self.muq, prompt=ref_prompt)
        
        # 获取负风格提示
        negative_style_prompt = get_negative_style_prompt(self.device)
        
        # 获取参考潜变量
        latent_prompt = get_reference_latent(self.device, self.max_frames)
        
        # 执行推理
        with torch.inference_mode():
            generated, _ = self.cfm.sample(
                cond=latent_prompt,
                text=lrc_prompt,
                duration=self.max_frames,
                style_prompt=style_prompt,
                negative_style_prompt=negative_style_prompt,
                steps=32,
                cfg_strength=4.0,
                start_time=start_time,
            )
            
            generated = generated.to(torch.float32)
            latent = generated.transpose(1, 2)  # [b d t]
            
            output = decode_audio(latent, self.vae, chunked=chunked)
            
            # 重排音频批次为单个序列
            output = rearrange(output, "b d n -> d (b n)")
            # 峰值归一化，裁剪，转换为int16
            output = (
                output.to(torch.float32)
                .div(torch.max(torch.abs(output)))
                .clamp(-1, 1)
                .mul(32767)
                .to(torch.int16)
                .cpu()
            )
        
        print(f"✅ DiffRhythm 生成完成")
        return output
    
    def generate_with_progress(self, lrc="", ref_prompt=None, ref_audio_path=None, chunked=False,
                              num_inference_steps=32, temperature=0.7, top_p=0.9):
        """
        带进度的生成方法，用于Gradio界面
        
        Args:
            lrc: LRC格式的歌词
            ref_prompt: 文本风格提示
            ref_audio_path: 音频风格提示路径
            chunked: 是否使用分块解码
            num_inference_steps: 推理步数
            temperature: 温度参数
            top_p: 核采样参数
        
        Returns:
            生成的音频数据
        """
        # 进度回调函数
        def diffusion_progress_callback(step, total_steps, progress_ratio):
            """扩散过程的进度回调"""
            if step == 0:
                print(f"🚀 开始扩散过程，共 {total_steps} 步...")
                print(f"   • 设备: {self.device}")
                if self.device == 'cuda':
                    print(f"   • 内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                else:
                    print(f"   • 使用 CPU 进行推理")
            elif step == total_steps:
                print(f"✅ 扩散过程完成，共 {total_steps} 步")
            elif step % 5 == 0:
                print(f"   • 已完成: {step}/{total_steps} ({progress_ratio:.1%})")
        
        # 设置进度回调
        if hasattr(self.cfm, 'set_progress_callback'):
            self.cfm.set_progress_callback(diffusion_progress_callback)
        
        # 调用生成方法
        return self.generate(lrc, ref_prompt, ref_audio_path, chunked)
