import os
import signal
import sys
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import torch
import torchaudio
from einops import rearrange
import random
import numpy as np

from transformers import AutoTokenizer
import time
from infer.infer_utils import (
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
)

# Global variables
MAX_SEED = np.iinfo(np.int32).max

# 检测可用设备
def get_available_device():
    """检测可用的计算设备，优先使用 GPU"""
    # 检测 CUDA
    if torch.cuda.is_available():
        return 'cuda'
    # 检测 AMD ROCm
    elif hasattr(torch, 'is_rocm_available') and torch.is_rocm_available():
        return 'cuda'  # ROCm 也使用 'cuda' 设备
    # 检测 MPS (Apple Silicon)
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

device = get_available_device()
print(f"🖥️  检测到计算设备: {device}")

# 如果是 AMD GPU，设置 ROCm 相关环境变量
if device == 'cuda' and hasattr(torch, 'is_rocm_available') and torch.is_rocm_available():
    print("🔴 检测到 AMD ROCm 支持，正在配置 AMD GPU 环境")
    # 设置 ROCm 相关环境变量
    os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '9.0.0')  # RX 580 对应的 GFX 版本
    os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx803')  # RX 580 的架构

# Multi-language support
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
        "language_selector": "Language / 语言"
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
        "language_selector": "语言 / Language"
    }
}

def inference_with_progress(
    cfm_model,
    vae_model,
    cond,
    text,
    duration,
    style_prompt,
    negative_style_prompt,
    start_time,
    chunked=True,
    num_inference_steps=32,
    temperature=0.7,
    top_p=0.9,
):
    """带进度条的推理函数"""
    print("📊 推理参数配置:")
    print(f"   • 扩散步数: {num_inference_steps}")
    print(f"   • CFG 强度: 4.0")
    print(f"   • 温度参数: {temperature}")
    print(f"   • Top-p 参数: {top_p}")
    print(f"   • 开始时间: {start_time.item():.2f} 秒")
    print(f"   • 分块解码: {'是' if chunked else '否'}")
    
    # 创建简单的文本进度条
    def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
        """打印文本进度条"""
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
        if iteration == total:
            print()
    
    # 进度回调函数
    def diffusion_progress_callback(step, total_steps, progress_ratio):
        """扩散过程的进度回调"""
        if step == 0:
            print(f"🚀 开始扩散过程，共 {total_steps} 步...")
            print(f"   • 设备: {device}")
            print(f"   • 内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB" if device == 'cuda' else "   • 使用 CPU 进行推理")
            print(f"   • 预计完成时间: {total_steps * 0.5:.1f} 秒")
        elif step == total_steps:
            print(f"✅ 扩散过程完成，共 {total_steps} 步")
        else:
            print_progress_bar(step, total_steps, prefix='扩散进度', suffix=f'步骤 {step}/{total_steps} (进度: {progress_ratio*100:.1f}%)')
    
    with torch.inference_mode():
        # 记录扩散开始时间
        diffusion_start_time = time.time()
        
        # 执行扩散采样，带进度监控
        generated, _ = cfm_model.sample(
            cond=cond,
            text=text,
            duration=duration,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            steps=num_inference_steps,
            cfg_strength=4.0,
            start_time=start_time,
            progress_callback=diffusion_progress_callback,
        )
        
        diffusion_time = time.time() - diffusion_start_time
        print(f"✅ 扩散过程完成，耗时 {diffusion_time:.2f} 秒")
        
        # 后处理阶段
        print("🔄 正在进行后处理...")
        
        # 转换数据类型
        generated = generated.to(torch.float32)
        latent = generated.transpose(1, 2)  # [b d t]
        
        # 解码音频
        print("🎵 正在解码音频...")
        output = decode_audio(latent, vae_model, chunked=chunked)
        
        # 重排音频批次
        output = rearrange(output, "b d n -> d (b n)")
        
        # 峰值归一化、裁剪、转换为 int16
        print("🔧 正在进行音频后处理...")
        output = (
            output.to(torch.float32)
            .div(torch.max(torch.abs(output)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )
        
        print("✅ 音频后处理完成")
        
        return output

# 保持原有函数名兼容性
inference = inference_with_progress

def infer_music(lrc, ref_audio_path, text_prompt, current_prompt_type, seed=42, randomize_seed=False, steps=32, cfg_strength=4.0, temperature=0.7, top_p=0.9, file_type='wav', odeint_method='euler', Music_Duration='95s'):
    """Main function to generate music from lyrics and prompts."""
    print("=" * 60)
    print("🎵 开始音乐生成流程")
    print("=" * 60)
    
    # 记录开始时间
    total_start_time = time.time()
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
        print(f"🔢 使用随机种子: {seed}")
    else:
        print(f"🔢 使用固定种子: {seed}")
    torch.manual_seed(seed)
    
    # Set up model parameters based on duration
    if Music_Duration == '95s':
        max_frames = 2048
        repo_id = "ASLP-lab/DiffRhythm-base"
        print("⏱️  选择 95秒 模型 (DiffRhythm-base)")
    else:  # '285s'
        max_frames = 6144
        repo_id = "ASLP-lab/DiffRhythm-full"
        print("⏱️  选择 285秒 模型 (DiffRhythm-full)")
    
    print(f"📊 模型参数: 最大帧数={max_frames}, 模型仓库={repo_id}")
    
    # Prepare models
    print("🔄 正在加载模型...")
    model_start_time = time.time()
    cfm, tokenizer, muq, vae = prepare_model(max_frames, device, repo_id=repo_id)
    model_load_time = time.time() - model_start_time
    print(f"✅ 模型加载完成，耗时 {model_load_time:.2f} 秒")

    try:
        # Process lyrics
        print("📝 正在处理歌词...")
        lrc_prompt, start_time = get_lrc_token(max_frames, lrc, tokenizer, device)
        print(f"✅ 歌词处理完成，开始时间: {start_time.item():.2f} 秒")
        
        # Get style prompt based on prompt type
        if current_prompt_type == 'audio':
            print("🎧 使用音频提示作为风格参考")
            style_prompt = get_style_prompt(muq, ref_audio_path)
        else:
            print(f"📋 使用文本提示作为风格参考: {text_prompt}")
            style_prompt = get_style_prompt(muq, prompt=text_prompt)
        print("✅ 风格提示生成完成")
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {str(e)}")
        raise gr.Error(f"Error: {str(e)}")
    
    # Get negative style prompt and reference latent
    print("🔄 正在生成负向风格提示和参考潜在空间...")
    negative_style_prompt = get_negative_style_prompt(device)
    latent_prompt = get_reference_latent(device, max_frames)
    print("✅ 负向风格提示和参考潜在空间生成完成")
    
    # Run inference
    print("🚀 开始音乐推理生成...")
    s_t = time.time()
    generated_song = inference_with_progress(
        cfm_model=cfm,
        vae_model=vae,
        cond=latent_prompt,
        text=lrc_prompt,
        duration=max_frames,
        style_prompt=style_prompt,
        negative_style_prompt=negative_style_prompt,
        start_time=start_time,
        chunked=True,
        num_inference_steps=steps,
        temperature=temperature,
        top_p=top_p,
    )
    e_t = time.time() - s_t
    print(f"✅ 推理生成完成，耗时 {e_t:.2f} 秒")
    
    # Save the generated song to a file
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_filename = f"diffrhythm_{timestamp}.{file_type}"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"💾 正在保存音频文件: {output_path}")
    torchaudio.save(output_path, generated_song, sample_rate=44100)
    
    # 计算总耗时
    total_time = time.time() - total_start_time
    
    print("=" * 60)
    print(f"🎉 音乐生成完成！")
    print(f"📊 总耗时: {total_time:.2f} 秒")
    print(f"📁 输出文件: {output_path}")
    print(f"🎵 音频格式: {file_type}")
    print(f"🔢 随机种子: {seed}")
    print(f"⏱️  音乐时长: {Music_Duration}")
    print("=" * 60)
    
    return output_path

# CSS styling for the UI
css = """
/* 固定文本域高度并强制滚动条 */
.lyrics-scroll-box textarea {
    height: 405px !important;  /* 固定高度 */
    max-height: 500px !important;  /* 最大高度 */
    overflow-y: auto !important;  /* 垂直滚动 */
    white-space: pre-wrap;  /* 保留换行 */
    line-height: 1.5;  /* 行高优化 */
}

.gr-examples {
    background: transparent !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 8px;
    margin: 1rem 0 !important;
    padding: 1rem !important;
}

"""

def create_language_tab(lang="en"):
    """Create a complete interface for a specific language."""
    lang_dict = LANGUAGES[lang]
    
    with gr.Tab(lang_dict["tab_music_generate"]):
        with gr.Row():
            with gr.Column():
                lrc = gr.Textbox(
                    label=lang_dict["label_lyrics"],
                    placeholder=lang_dict["placeholder_lyrics"],
                    lines=12,
                    max_lines=50,
                    elem_classes="lyrics-scroll-box",
                    value="""[00:04.34]Tell me that I'm special
[00:06.57]Tell me I look pretty
[00:08.46]Tell me I'm a little angel
[00:10.58]Sweetheart of your city
[00:13.64]Say what I'm dying to hear
[00:17.35]Cause I'm dying to hear you
[00:20.86]Tell me I'm that new thing
[00:22.93]Tell me that I'm relevant
[00:24.96]Tell me that I got a big heart
[00:27.04]Then back it up with evidence
[00:29.94]I need it and I don't know why
[00:34.28]This late at night
[00:36.32]Isn't it lonely
[00:39.24]I'd do anything to make you want me
[00:43.40]I'd give it all up if you told me
[00:47.42]That I'd be
[00:49.43]The number one girl in your eyes
[00:52.85]Your one and only
[00:55.74]So what's it gon' take for you to want me
[00:59.78]I'd give it all up if you told me
[01:03.89]That I'd be
[01:05.94]The number one girl in your eyes
[01:11.34]Tell me I'm going real big places
[01:14.32]Down to earth so friendly
[01:16.30]And even through all the phases
[01:18.46]Tell me you accept me
[01:21.56]Well that's all I'm dying to hear
[01:25.30]Yeah I'm dying to hear you
[01:28.91]Tell me that you need me
[01:30.85]Tell me that I'm loved
[01:32.90]Tell me that I'm worth it
[01:34.95]And that I'm enough
[01:37.91]I need it and I don't know why
[01:42.08]This late at night
[01:44.24]Isn't it lonely
[01:47.18]I'd do anything to make you want me
[01:51.30]I'd give it all up if you told me
[01:55.32]That I'd be
[01:57.35]The number one girl in your eyes
[02:00.72]Your one and only
[02:03.57]So what's it gon' take for you to want me
[02:07.78]I'd give it all up if you told me
[02:11.74]That I'd be
[02:13.86]The number one girl in your eyes
[02:17.03]The girl in your eyes
[02:21.05]The girl in your eyes
[02:26.30]Tell me I'm the number one girl
[02:28.44]I'm the number one girl in your eyes
[02:33.49]The girl in your eyes
[02:37.58]The girl in your eyes
[02:42.74]Tell me I'm the number one girl
[02:44.88]I'm the number one girl in your eyes"""
                )
                
                current_prompt_type = gr.State(value="audio")
                with gr.Tabs() as inside_tabs:
                    with gr.Tab(lang_dict["tab_audio_prompt"]):
                        audio_prompt = gr.Audio(label="Audio Prompt", type="filepath", value="./src/prompt/default.wav")
                    with gr.Tab(lang_dict["tab_text_prompt"]):
                        text_prompt = gr.Textbox(
                            label=lang_dict["label_text_prompt"],
                            placeholder=lang_dict["placeholder_text_prompt"],
                        )
                
                with gr.Accordion(lang_dict["accordion_advanced"], open=False):
                    seed = gr.Slider(
                        label=lang_dict["label_seed"],
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0,
                    )
                    randomize_seed = gr.Checkbox(label=lang_dict["label_randomize_seed"], value=True)
                    
                    steps = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=32,
                        step=1,
                        label=lang_dict["label_diffusion_steps"],
                        interactive=True,
                        elem_id="step_slider"
                    )
                    cfg_strength = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=4.0,
                        step=0.5,
                        label=lang_dict["label_cfg_strength"],
                        interactive=True,
                        elem_id="cfg_slider"
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="温度参数 (Temperature)" if lang == "zh" else "Temperature",
                        interactive=True,
                        elem_id="temperature_slider"
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-p 参数" if lang == "zh" else "Top-p",
                        interactive=True,
                        elem_id="top_p_slider"
                    )
                    odeint_method = gr.Radio(["euler", "midpoint", "rk4", "implicit_adams"], label=lang_dict["label_ode_solver"], value="euler")                        
                    file_type = gr.Dropdown(["wav", "mp3", "ogg"], label=lang_dict["label_output_format"], value="wav")

                    def update_prompt_type(evt: gr.SelectData):
                        return "audio" if evt.index == 0 else "text"

                    inside_tabs.select(
                        fn=update_prompt_type,
                        outputs=current_prompt_type
                    )
                
            with gr.Column():
                with gr.Accordion(lang_dict["accordion_best_practices"], open=True):
                    gr.Markdown(lang_dict["markdown_best_practices"])
                
                Music_Duration = gr.Radio(["95s", "285s"], label=lang_dict["label_music_duration"], value="95s")
                
                lyrics_btn = gr.Button(lang_dict["btn_generate"], variant="primary")
                audio_output = gr.Audio(label=lang_dict["label_audio_result"], type="filepath", elem_id="audio_output")
        
        # Examples for the current language
        gr.Examples(
            examples=[
                ["./src/prompt/pop_cn.wav"], 
                ["./src/prompt/default.wav"],
            ],
            inputs=[audio_prompt],  
            label=lang_dict["label_audio_examples"],
            examples_per_page=13,
            elem_id="audio-examples-container" 
        )
        
        gr.Examples(
            examples=[
                ["Pop Emotional Piano"],
                ["Electronic Dance Music"],
                ["Acoustic Folk Guitar"],
                ["Orchestral Cinematic"],
            ],
            inputs=[text_prompt],  
            label=lang_dict["label_text_examples"],
            examples_per_page=4,
            elem_id="text-examples-container" 
        )

        gr.Examples(
            examples=[
                ["""[00:04.34]I'm standing on the edge of tomorrow
[00:08.55]Looking out at a world I don't know
[00:12.67]The path ahead is filled with shadows
[00:16.83]But I know I can't let go"""],
                ["""[00:02.00]The morning sun breaks through the clouds
[00:06.50]As I walk along the shore
[00:10.75]The waves crash gently at my feet
[00:15.00]I've never felt so sure"""],
            ],
            inputs=[lrc],
            label=lang_dict["label_lrc_examples"],
            examples_per_page=3,
            elem_id="lrc-examples-container",
        )
    
    return lrc, audio_prompt, text_prompt, current_prompt_type, seed, randomize_seed, steps, cfg_strength, temperature, top_p, file_type, odeint_method, Music_Duration, lyrics_btn, audio_output

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.HTML(f"""
            <div style="display: flex; align-items: center;">
                <img src='https://raw.githubusercontent.com/ASLP-lab/DiffRhythm/refs/heads/main/src/DiffRhythm_logo.jpg' 
                    style='width: 200px; height: 40%; display: block; margin: 0 auto 20px;'>
            </div>
            
            <div style="flex: 1; text-align: center;">
                <div style="font-size: 2em; font-weight: bold; text-align: center; margin-bottom: 5px">
                    Di♪♪Rhythm (谛韵)
                </div>
                <div style="display:flex; justify-content: center; column-gap:4px;">
                    <a href="https://arxiv.org/abs/2503.01183">
                        <img src='https://img.shields.io/badge/Arxiv-Paper-blue'>
                    </a> 
                    <a href="https://github.com/ASLP-lab/DiffRhythm">
                        <img src='https://img.shields.io/badge/GitHub-Repo-green'>
                    </a> 
                    <a href="https://aslp-lab.github.io/DiffRhythm.github.io/">
                        <img src='https://img.shields.io/badge/Project-Page-brown'>
                    </a>
                </div>
            </div> 
            """)
    
    # Create tabs for each language
    with gr.Tabs():
        # English interface
        lrc_en, audio_prompt_en, text_prompt_en, current_prompt_type_en, seed_en, randomize_seed_en, steps_en, cfg_strength_en, temperature_en, top_p_en, file_type_en, odeint_method_en, Music_Duration_en, lyrics_btn_en, audio_output_en = create_language_tab("en")
        
        # Chinese interface
        lrc_zh, audio_prompt_zh, text_prompt_zh, current_prompt_type_zh, seed_zh, randomize_seed_zh, steps_zh, cfg_strength_zh, temperature_zh, top_p_zh, file_type_zh, odeint_method_zh, Music_Duration_zh, lyrics_btn_zh, audio_output_zh = create_language_tab("zh")
    
    # Connect the generate buttons to the inference function
    lyrics_btn_en.click(
        fn=infer_music,
        inputs=[lrc_en, audio_prompt_en, text_prompt_en, current_prompt_type_en, seed_en, randomize_seed_en, steps_en, cfg_strength_en, temperature_en, top_p_en, file_type_en, odeint_method_en, Music_Duration_en],
        outputs=audio_output_en
    )
    
    lyrics_btn_zh.click(
        fn=infer_music,
        inputs=[lrc_zh, audio_prompt_zh, text_prompt_zh, current_prompt_type_zh, seed_zh, randomize_seed_zh, steps_zh, cfg_strength_zh, temperature_zh, top_p_zh, file_type_zh, odeint_method_zh, Music_Duration_zh],
        outputs=audio_output_zh
    )

def signal_handler(signum, frame):
    """处理中断信号的函数"""
    print("\n" + "=" * 60)
    print("⚠️  接收到中断信号，正在优雅退出...")
    print("=" * 60)
    
    # 清理资源
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("✅ 资源清理完成，程序退出")
    sys.exit(0)

if __name__ == "__main__":
    # 注册信号处理器，支持 Ctrl+C 中断
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 60)
    print("🎵 DiffRhythm 音乐生成系统")
    print("=" * 60)
    print("📋 使用说明:")
    print("   • 按 Ctrl+C 可以中断程序运行")
    print("   • 音乐生成过程中可以随时中断")
    print("   • 中断后会自动清理 GPU 内存")
    print("=" * 60)
    
    try:
        demo.launch(css=css)
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("👋 用户主动中断程序，感谢使用！")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 程序运行出错: {str(e)}")
        sys.exit(1)