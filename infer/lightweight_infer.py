# Copyright (c) 2025 ASLP-LAB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import requests
import json
import time
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class LightweightMusicGenerator:
    """轻量级音乐生成器，支持本地轻量模型和AI API"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.lightweight_model = None
        self.api_key = os.getenv('AI_API_KEY', '')
        self.api_base = os.getenv('AI_API_BASE', '')
        self.pipes = {}  # MusicGen模型管道字典，支持多个模型
        self.current_model = None  # 当前使用的模型名称
        
    def load_lightweight_model(self, model_type="musicgen-small"):
        """加载轻量级音乐生成模型
        
        Args:
            model_type: 模型类型，可选值：musicgen-small, musicgen-melody
        """
        print(f"📦 正在加载轻量级音乐生成模型: {model_type}")
        
        try:
            # 尝试加载MusicGen模型
            from transformers import MusicgenForConditionalGeneration, AutoProcessor
            
            # 根据模型类型选择模型名称
            model_mapping = {
                "musicgen-small": "facebook/musicgen-small",
                "musicgen-melody": "facebook/musicgen-melody"
            }
            
            model_name = model_mapping.get(model_type, "facebook/musicgen-small")
            print(f"   • 正在加载模型: {model_name}")
            
            # 如果模型已经加载，直接返回
            if model_type in self.pipes:
                print(f"✅ {model_type} 模型已加载，直接使用")
                self.current_model = model_type
                return self.pipes[model_type]
            
            # 加载处理器（所有MusicGen模型使用相同的处理器）
            if not hasattr(self, 'processor'):
                self.processor = AutoProcessor.from_pretrained(
                    "facebook/musicgen-small", 
                    cache_dir="./pretrained"
                )
            
            # 加载模型
            pipe = MusicgenForConditionalGeneration.from_pretrained(
                model_name, 
                cache_dir="./pretrained",
                torch_dtype=torch.float32 if self.device == 'cpu' else torch.float16
            ).to(self.device)
            
            # 优化模型加载
            pipe.eval()
            
            # 保存到模型字典
            self.pipes[model_type] = pipe
            self.current_model = model_type
            
            print(f"✅ {model_type} 模型加载完成")
            return pipe
        except Exception as e:
            print(f"❌ {model_type} 模型加载失败: {e}")
            print("⚠️  回退到简单模型")
            
            # 示例：使用一个简单的线性模型作为轻量级模型
            class SimpleMusicModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = torch.nn.Linear(512, 1024)
                    
                def forward(self, x):
                    return self.fc(x)
            
            self.lightweight_model = SimpleMusicModel().to(self.device)
            self.current_model = "simple"
            print("✅ 简单模型加载完成")
            return self.lightweight_model
    
    def generate_with_lightweight_model(self, text_prompt=None, wav_path=None, duration=10, steps=10, model_type="musicgen-small"):
        """使用轻量级模型生成音乐
        
        Args:
            text_prompt: 文本提示
            wav_path: 音频提示路径
            duration: 生成时长
            steps: 生成步数（未使用）
            model_type: 模型类型，可选值：musicgen-small, musicgen-melody
        """
        # 加载指定类型的模型
        current_pipe = self.load_lightweight_model(model_type)
        
        # 确定使用哪种提示
        if text_prompt and text_prompt.strip():
            print(f"🎵 使用{model_type}生成音乐，文本提示: {text_prompt}")
            is_text_prompt = True
        elif wav_path:
            print(f"🎵 使用{model_type}生成音乐，音频提示: {wav_path}")
            is_text_prompt = False
        else:
            raise ValueError("请提供有效的文本提示或音频提示")
        
        print(f"   • 请求时长: {duration}秒")
        print(f"   • 设备: {self.device}")
        print(f"   • 当前模型: {self.current_model}")
        
        # 使用MusicGen模型生成音乐
        if current_pipe is not None and is_text_prompt:
            try:
                # 检查模型的最大位置嵌入限制
                # 使用try-except块处理不同的配置结构
                try:
                    # 尝试从decoder_config获取
                    max_position_embeddings = current_pipe.config.decoder_config.max_position_embeddings
                    print(f"   • 模型最大位置嵌入: {max_position_embeddings}")
                except AttributeError:
                    # 尝试从直接配置获取
                    try:
                        max_position_embeddings = current_pipe.config.max_position_embeddings
                        print(f"   • 模型最大位置嵌入: {max_position_embeddings}")
                    except AttributeError:
                        # 设置默认安全值，MusicGen-small的max_position_embeddings为2048
                        max_position_embeddings = 2048
                        print(f"   • 使用默认最大位置嵌入: {max_position_embeddings}")
                
                # MusicGen的采样率是32000 Hz
                sample_rate = 32000
                
                # 进一步限制最大生成时长，MusicGen-small适合短音频生成
                # 限制为最大30秒，避免内存和计算问题
                actual_duration = min(duration, 30)
                print(f"   • 调整后生成时长: {actual_duration}秒")
                
                # 计算每个token对应的音频长度（MusicGen的token步长）
                # MusicGen使用编解码器，每个token对应~20ms音频
                token_step_ms = 20
                tokens_per_second = 1000 / token_step_ms
                
                # 计算需要的token数量
                required_tokens = int(actual_duration * tokens_per_second)
                
                # 使用保守的生成参数，确保不超过模型限制
                # 1. 基于调整后的时长计算
                # 2. 不超过模型最大位置嵌入的1/4
                # 3. 不超过8000个token（保守限制）
                max_new_tokens = min(
                    required_tokens,
                    max_position_embeddings // 4,  # 使用1/4的最大位置嵌入，确保安全
                    8000  # 最大不超过8000个token
                )
                
                print(f"   • 生成token数量: {max_new_tokens}")
                print(f"   • 预计生成时长: {max_new_tokens / tokens_per_second:.2f}秒")
                
                # 生成音乐输入
                inputs = self.processor(
                    text=[text_prompt],
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # 使用model.generate方法生成音频
                print("   • 正在生成音频...")
                with torch.no_grad():
                    # 使用优化的生成参数，提高音频质量
                    audio_values = current_pipe.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.6,  # 降低随机性，提高音频质量
                        top_p=0.90,  # 合理的概率质量保留
                        guidance_scale=1.0,  # 对于MusicGen-small，1.0是更稳定的指导权重
                        use_cache=True,  # 启用缓存，提高生成速度
                        num_return_sequences=1  # 生成一个序列
                    )
                
                # 生成输出文件路径
                output_path = f"output/lightweight_{int(time.time())}.wav"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # 保存音频
                from scipy.io.wavfile import write
                import numpy as np
                
                # 获取音频数据并转换为numpy数组
                audio_values = audio_values.cpu().numpy()[0, 0]
                
                # 调试：检查音频数据的统计信息
                print(f"   • 音频数据范围: {audio_values.min():.6f} 到 {audio_values.max():.6f}")
                print(f"   • 音频数据平均值: {audio_values.mean():.6f}")
                print(f"   • 音频数据标准差: {audio_values.std():.6f}")
                
                # 检查是否有明显的音频信号
                audio_energy = np.sum(audio_values ** 2) / len(audio_values)
                print(f"   • 音频能量: {audio_energy:.6f}")
                
                # 音频后处理：改进音频质量
                import scipy.signal as signal
                
                # 1. 应用低通滤波器，减少高频噪音
                # 设计一个简单的低通滤波器，截止频率为8kHz
                sos = signal.butter(10, 8000, 'low', fs=sample_rate, output='sos')
                audio_values = signal.sosfilt(sos, audio_values)
                print(f"   • 应用低通滤波器，减少高频噪音")
                
                # 2. 归一化音频数据，确保有足够的音量
                # 计算音频的峰值
                peak_value = np.max(np.abs(audio_values))
                if peak_value > 0:
                    # 归一化到合适的范围
                    target_peak = 0.8  # 目标峰值，避免削波
                    audio_values = audio_values * (target_peak / peak_value)
                    print(f"   • 归一化音频，峰值从 {peak_value:.6f} 调整到 {target_peak}")
                
                # 3. 如果音频能量仍然过低，进行增益处理
                audio_energy = np.sum(audio_values ** 2) / len(audio_values)
                if audio_energy < 0.01:
                    print(f"   • 音频能量过低，应用增益处理")
                    # 计算增益因子，将音频能量提升到0.05左右
                    gain_factor = np.sqrt(0.05 / max(audio_energy, 1e-8))
                    # 限制最大增益，避免噪音过度放大
                    gain_factor = min(gain_factor, 5.0)  # 最大增益不超过5倍
                    audio_values = audio_values * gain_factor
                    audio_values = np.clip(audio_values, -1.0, 1.0)
                    print(f"   • 应用增益因子: {gain_factor:.2f}")
                    print(f"   • 增益后音频数据范围: {audio_values.min():.6f} 到 {audio_values.max():.6f}")
                
                # MusicGen生成的音频是float32类型，范围在[-1, 1]
                # 需要转换为int16格式才能保存为标准WAV文件
                audio_values = audio_values * 32767  # 将范围从[-1, 1]转换为[-32767, 32767]
                audio_values = np.clip(audio_values, -32767, 32767)  # 确保在有效范围内
                audio_values = audio_values.astype(np.int16)  # 转换为int16格式
                
                # 保存为WAV文件
                write(output_path, sample_rate, audio_values)
                
                print(f"✅ MusicGen模型生成完成，输出文件: {output_path}")
                print(f"   • 采样率: {sample_rate} Hz")
                print(f"   • 实际音频长度: {len(audio_values) / sample_rate:.2f} 秒")
                return output_path
            except Exception as e:
                print(f"❌ MusicGen模型生成失败: {e}")
                import traceback
                traceback.print_exc()
                print("⚠️  回退到简单波形生成")
        
        # 回退到简单波形生成
        output_path = f"output/lightweight_{int(time.time())}.wav"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 生成有效的WAV文件
        self._generate_valid_wav(output_path, duration, text_prompt or "default")
        
        print(f"✅ 简单波形生成完成，输出文件: {output_path}")
        return output_path
    
    def _generate_valid_wav(self, output_path, duration, text_prompt):
        """生成有效的WAV文件（作为回退方案）"""
        import numpy as np
        import scipy.io.wavfile
        
        # 生成简单的音频波形
        sample_rate = 44100
        
        # 生成基础音调
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # 根据文本提示生成不同的波形
        if "piano" in text_prompt.lower() or "classical" in text_prompt.lower():
            # 钢琴风格：单音
            frequency = 440  # A4音
            audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        elif "electronic" in text_prompt.lower() or "dance" in text_prompt.lower():
            # 电子舞曲风格：和弦
            freq1 = 440  # A4
            freq2 = 554.37  # C5
            freq3 = 659.25  # E5
            audio_data = 0.3 * np.sin(2 * np.pi * freq1 * t) + \
                        0.3 * np.sin(2 * np.pi * freq2 * t) + \
                        0.3 * np.sin(2 * np.pi * freq3 * t)
        elif "guitar" in text_prompt.lower() or "folk" in text_prompt.lower():
            # 吉他风格：带有衰减的波形
            frequency = 392  # G4
            envelope = np.exp(-t * 0.5)  # 指数衰减
            audio_data = 0.5 * envelope * np.sin(2 * np.pi * frequency * t)
        else:
            # 默认风格：简单的正弦波
            frequency = 440  # A4音
            audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # 添加一些变化，使音频更有趣
        modulator = 0.5 * np.sin(2 * np.pi * 0.5 * t)  # 低频调制
        audio_data = audio_data * (1 + 0.1 * modulator)
        
        # 确保音频在合理范围内
        audio_data = np.clip(audio_data, -0.9, 0.9)
        
        # 转换为16位整数
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # 保存为WAV文件
        scipy.io.wavfile.write(output_path, sample_rate, audio_data)
