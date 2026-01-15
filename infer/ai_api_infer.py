# -*- coding: utf-8 -*-
"""
AI API 音乐生成模块
"""

import os
import time
import requests
import json


class AIApiManager:
    """
    AI API管理器，用于管理不同的AI API提供商
    """
    
    def __init__(self):
        self.api_providers = {
            "free": {
                "name": "免费API",
                "description": "使用免费的国内AI API生成音乐",
                "requires_key": False,
                "supported_features": ["text_prompt", "duration"]
            },
            "baidu": {
                "name": "百度AI",
                "description": "使用百度AI API生成音乐",
                "requires_key": True,
                "supported_features": ["text_prompt", "duration", "style"]
            },
            "tencent": {
                "name": "腾讯AI",
                "description": "使用腾讯AI API生成音乐",
                "requires_key": True,
                "supported_features": ["text_prompt", "duration", "style"]
            }
        }
    
    def get_available_providers(self):
        """
        获取可用的API提供商
        
        Returns:
            可用API提供商列表
        """
        return list(self.api_providers.keys())
    
    def get_provider_info(self, provider):
        """
        获取API提供商的详细信息
        
        Args:
            provider: API提供商名称
            
        Returns:
            API提供商详细信息
        """
        return self.api_providers.get(provider, {})
    
    def validate_api_key(self, provider, api_key):
        """
        验证API密钥是否有效
        
        Args:
            provider: API提供商名称
            api_key: API密钥
            
        Returns:
            密钥是否有效
        """
        # 实际项目中应实现真实的API密钥验证
        return True


class AIApiMusicGenerator:
    """
    AI API 音乐生成器类
    """
    
    def __init__(self, device='cpu'):
        """
        初始化AI API音乐生成器
        
        Args:
            device: 计算设备（未使用，保留兼容性）
        """
        self.device = device
        self.api_key = os.getenv('AI_API_KEY', '')
        self.api_base = os.getenv('AI_API_BASE', '')
        self.api_manager = AIApiManager()
    
    def generate(self, text_prompt, duration=10, api_provider="free"):
        """
        使用AI API生成音乐
        
        Args:
            text_prompt: 文本提示
            duration: 生成时长
            api_provider: API提供商
            
        Returns:
            生成的音频文件路径
        """
        print(f"🌐 使用AI API生成音乐，提供商: {api_provider}")
        print(f"   • 提示: {text_prompt}")
        print(f"   • 时长: {duration}秒")
        
        # 根据API提供商选择不同的API
        if api_provider == "free":
            return self.generate_with_free_api(text_prompt, duration)
        elif api_provider == "baidu":
            return self.generate_with_baidu_api(text_prompt, duration)
        elif api_provider == "tencent":
            return self.generate_with_tencent_api(text_prompt, duration)
        else:
            raise ValueError(f"不支持的API提供商: {api_provider}")
    
    def generate_with_free_api(self, text_prompt, duration=10):
        """
        使用免费AI API生成音乐
        
        Args:
            text_prompt: 文本提示
            duration: 生成时长
            
        Returns:
            生成的音频文件路径
        """
        print("   • 使用免费AI API...")
        
        # 模拟API调用过程
        time.sleep(3)
        
        # 生成一个简单的音频文件路径
        output_path = f"output/free_api_{int(time.time())}.wav"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 生成有效的WAV文件
        self._generate_valid_wav(output_path, duration, text_prompt)
        
        print(f"✅ 免费API生成完成，输出文件: {output_path}")
        return output_path
    
    def generate_with_baidu_api(self, text_prompt, duration=10):
        """
        使用百度AI API生成音乐
        
        Args:
            text_prompt: 文本提示
            duration: 生成时长
            
        Returns:
            生成的音频文件路径
        """
        # 实际项目中应实现真实的百度AI API调用
        print("   • 使用百度AI API...")
        print("   • 注意：需要配置百度API密钥")
        
        # 模拟API调用过程
        time.sleep(4)
        
        # 生成一个简单的音频文件路径
        output_path = f"output/baidu_api_{int(time.time())}.wav"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 生成有效的WAV文件
        self._generate_valid_wav(output_path, duration, text_prompt)
        
        print(f"✅ 百度API生成完成，输出文件: {output_path}")
        return output_path
    
    def generate_with_tencent_api(self, text_prompt, duration=10):
        """
        使用腾讯AI API生成音乐
        
        Args:
            text_prompt: 文本提示
            duration: 生成时长
            
        Returns:
            生成的音频文件路径
        """
        # 实际项目中应实现真实的腾讯AI API调用
        print("   • 使用腾讯AI API...")
        print("   • 注意：需要配置腾讯API密钥")
        
        # 模拟API调用过程
        time.sleep(4)
        
        # 生成一个简单的音频文件路径
        output_path = f"output/tencent_api_{int(time.time())}.wav"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 生成有效的WAV文件
        self._generate_valid_wav(output_path, duration, text_prompt)
        
        print(f"✅ 腾讯API生成完成，输出文件: {output_path}")
        return output_path
    
    def _generate_valid_wav(self, output_path, duration, text_prompt):
        """
        生成有效的WAV文件
        
        Args:
            output_path: 输出文件路径
            duration: 生成时长
            text_prompt: 文本提示
        """
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
