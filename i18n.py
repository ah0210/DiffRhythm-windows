# -*- coding: utf-8 -*-
"""
多语言支持模块
"""

# 多语言支持配置
LANGUAGES = {
    "en": {
        "title": "Di♪♪Rhythm (谛韵)",
        "tab_music_generate": "Music Generate",
        "label_lyrics": "Lyrics",
        "placeholder_lyrics": "Input the full lyrics in LRC format",
        "tab_audio_prompt": "Audio Prompt",
        "tab_text_prompt": "Text Prompt",
        "label_text_prompt": "Text Prompt",
        "placeholder_text_prompt": "Enter the Text Prompt, eg: emotional piano pop",
        "accordion_advanced": "Advanced Settings",
        "label_seed": "Seed",
        "label_randomize_seed": "Randomize seed",
        "label_diffusion_steps": "Diffusion Steps",
        "label_cfg_strength": "CFG Strength",
        "label_ode_solver": "ODE Solver",
        "label_output_format": "Output Format",
        "accordion_best_practices": "Best Practices Guide",
        "label_music_duration": "Music Duration",
        "btn_generate": "Generate",
        "label_audio_result": "Audio Result",
        "label_audio_examples": "Audio Examples",
        "label_text_examples": "Text Examples",
        "label_lrc_examples": "Lrc Examples",
        "markdown_best_practices": """
        1. **Lyrics Format Requirements**
            - Each line must follow: `[mm:ss.xx]Lyric content`
            - Example of valid format:
            ``` 
            [00:10.00]Moonlight spills through broken blinds
            [00:13.20]Your shadow dances on the dashboard shrine
            ```
        2. **Audio Prompt Requirements**
            - Reference audio should be ≥ 1 second, audio >10 seconds will be randomly clipped into 10 seconds
            - For optimal results, the 10-second clips should be carefully selected
            - Shorter clips may lead to incoherent generation
        3. **Supported Languages**
            - **Chinese and English**
            - More languages comming soon

        4. **Others** 
            - If loading audio result is slow, you can select Output Format as mp3 in Advanced Settings.
        """,
        "language_selector": "Language / 语言",
        "label_model_type": "Model Type",
        "option_model_diffrhythm": "DiffRhythm",
        "option_model_lightweight": "Lightweight",
        "option_model_ai_api": "AI API",
        "label_api_provider": "API Provider",
        "option_api_free": "Free API",
        "option_api_baidu": "Baidu AI",
        "option_api_tencent": "Tencent AI",
        "label_lightweight_model": "Lightweight Model",
        "option_lightweight_musicgen_small": "MusicGen-small",
        "option_lightweight_musicgen_melody": "MusicGen-melody"
    },
    "zh": {
        "title": "Di♪♪Rhythm (谛韵)",
        "tab_music_generate": "音乐生成",
        "label_lyrics": "歌词",
        "placeholder_lyrics": "输入完整的歌词，使用 LRC 格式",
        "tab_audio_prompt": "音频提示",
        "tab_text_prompt": "文本提示",
        "label_text_prompt": "文本提示",
        "placeholder_text_prompt": "输入文本提示，例如：情感钢琴流行音乐",
        "accordion_advanced": "高级设置",
        "label_seed": "随机种子",
        "label_randomize_seed": "随机化种子",
        "label_diffusion_steps": "扩散步数",
        "label_cfg_strength": "CFG 强度",
        "label_ode_solver": "ODE 求解器",
        "label_output_format": "输出格式",
        "accordion_best_practices": "最佳实践指南",
        "label_music_duration": "音乐时长",
        "btn_generate": "生成",
        "label_audio_result": "音频结果",
        "label_audio_examples": "音频示例",
        "label_text_examples": "文本示例",
        "label_lrc_examples": "歌词示例",
        "markdown_best_practices": """
        1. **歌词格式要求**
            - 每行必须遵循格式：`[mm:ss.xx]歌词内容`
            - 有效格式示例：
            ``` 
            [00:10.00]月光透过破碎的百叶窗洒落
            [00:13.20]你的影子在仪表盘神龛上舞动
            ```
        2. **音频提示要求**
            - 参考音频应 ≥ 1 秒，音频 > 10 秒将被随机裁剪为 10 秒
            - 为获得最佳效果，应仔细选择 10 秒片段
            - 较短的片段可能导致生成不连贯
        3. **支持的语言**
            - **中文和英文**
            - 更多语言即将推出

        4. **其他** 
            - 如果音频结果加载缓慢，您可以在高级设置中选择输出格式为 mp3。
        """,
        "language_selector": "语言 / Language",
        "label_model_type": "模型类型",
        "option_model_diffrhythm": "DiffRhythm",
        "option_model_lightweight": "轻量级模型",
        "option_model_ai_api": "AI API",
        "label_api_provider": "API提供商",
        "option_api_free": "免费API",
        "option_api_baidu": "百度AI",
        "option_api_tencent": "腾讯AI",
        "label_lightweight_model": "轻量级模型",
        "option_lightweight_musicgen_small": "MusicGen-small",
        "option_lightweight_musicgen_melody": "MusicGen-melody"
    }
}

# 默认语言
DEFAULT_LANGUAGE = "zh"


def get_translation(lang, key):
    """
    获取指定语言的翻译
    
    Args:
        lang: 语言代码，如 "en" 或 "zh"
        key: 翻译键
        
    Returns:
        翻译后的文本
    """
    return LANGUAGES.get(lang, LANGUAGES[DEFAULT_LANGUAGE]).get(key, key)


def get_available_languages():
    """
    获取可用的语言列表
    
    Returns:
        语言代码列表
    """
    return list(LANGUAGES.keys())


def get_language_names():
    """
    获取语言名称映射
    
    Returns:
        语言代码到语言名称的映射
    """
    return {
        "en": "English",
        "zh": "中文"
    }
