import os
import signal
import sys
import logging
from dotenv import load_dotenv
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 修复Windows终端编码问题
if sys.platform == 'win32':
    import codecs
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleOutputCP(65001)  # 设置控制台编码为UTF-8

# 导入多语言支持模块
from i18n import LANGUAGES, DEFAULT_LANGUAGE, get_translation, get_available_languages, get_language_names

logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("🎵 DiffRhythm 音乐生成系统")
logger.info("=" * 60)

import torch
import torchaudio
import gradio as gr
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
from infer.diffrhythm_infer import DiffRhythmGenerator
from infer.lightweight_infer import LightweightMusicGenerator
from infer.ai_api_infer import AIApiMusicGenerator, AIApiManager

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
    elif hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available():
        return 'mps'
    # 检测 DirectML (Windows上的AMD GPU)
    else:
        try:
            import torch_directml
            # 检查是否有可用的DirectML设备
            if torch_directml.device_count() > 0:
                return torch_directml.device(0)  # 使用第一个DirectML设备
        except ImportError:
            pass
        except Exception:
            pass
        return 'cpu'

device = get_available_device()
logger.info(f"🖥️  检测到计算设备: {device}")

# 如果是 AMD GPU，设置 ROCm 相关环境变量
if device == 'cuda' and hasattr(torch, 'is_rocm_available') and torch.is_rocm_available():
    logger.info("🔴 检测到 AMD ROCm 支持，正在配置 AMD GPU 环境")
    # 设置 ROCm 相关环境变量
    os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '9.0.0')  # RX 580 对应的 GFX 版本
    os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx803')  # RX 580 的架构

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
    logger.info("📊 推理参数配置:")
    logger.info(f"   • 扩散步数: {num_inference_steps}")
    logger.info(f"   • CFG 强度: 4.0")
    logger.info(f"   • 温度参数: {temperature}")
    logger.info(f"   • Top-p 参数: {top_p}")
    logger.info(f"   • 开始时间: {start_time.item():.2f} 秒")
    logger.info(f"   • 分块解码: {'是' if chunked else '否'}")
    
    # 进度回调函数
    def diffusion_progress_callback(step, total_steps, progress_ratio):
        """扩散过程的进度回调"""
        if step == 0:
            logger.info(f"🚀 开始扩散过程，共 {total_steps} 步...")
            logger.info(f"   • 设备: {device}")
            logger.info(f"   • 内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB" if device == 'cuda' else "   • 使用 CPU 进行推理")
        elif step == total_steps:
            logger.info(f"✅ 扩散过程完成，共 {total_steps} 步")
        elif step % 5 == 0:  # 每5步记录一次
            logger.info(f"   扩散进度: {step}/{total_steps} ({progress_ratio*100:.1f}%)")
    
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
        logger.info(f"✅ 扩散过程完成，耗时 {diffusion_time:.2f} 秒")
        
        # 后处理阶段
        logger.info("🔄 正在进行后处理...")
        
        # 转换数据类型
        generated = generated.to(torch.float32)
        latent = generated.transpose(1, 2)  # [b d t]
        
        # 解码音频
        logger.info("🎵 正在解码音频...")
        output = decode_audio(latent, vae_model, chunked=chunked)
        
        # 重排音频批次
        output = rearrange(output, "b d n -> d (b n)")
        
        # 峰值归一化、裁剪、转换为 int16
        logger.info("🔧 正在进行音频后处理...")
        output = (
            output.to(torch.float32)
            .div(torch.max(torch.abs(output)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )
        
        logger.info("✅ 音频后处理完成")
        
        return output

# 保持原有函数名兼容性
inference = inference_with_progress

def infer_music(lrc, ref_audio_path, text_prompt, current_prompt_type, seed=42, randomize_seed=False, steps=32, cfg_strength=4.0, temperature=0.7, top_p=0.9, file_type='wav', odeint_method='euler', Music_Duration='95s', model_type='lightweight', api_provider='free', lightweight_model_type='musicgen-small'):
    """Main function to generate music from lyrics and prompts."""
    logger.info("=" * 60)
    logger.info("🎵 开始音乐生成流程")
    logger.info("=" * 60)
    logger.info(f"📋 接收到的参数:")
    logger.info(f"   • 提示类型: {current_prompt_type}, 类型: {type(current_prompt_type)}")
    logger.info(f"   • 音频提示路径: {ref_audio_path}, 类型: {type(ref_audio_path)}")
    logger.info(f"   • 文本提示: {text_prompt}, 类型: {type(text_prompt)}")
    logger.info(f"   • 模型类型: {model_type}, 类型: {type(model_type)}")
    logger.info(f"   • API提供商: {api_provider}, 类型: {type(api_provider)}")
    
    # 记录开始时间
    total_start_time = time.time()
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
        logger.info(f"🔢 使用随机种子: {seed}")
    else:
        logger.info(f"🔢 使用固定种子: {seed}")
    torch.manual_seed(seed)
    
    # 检查是否有有效的提示
    if model_type in ['ai_api']:
        # AI API 只支持文本提示
        if not text_prompt or not text_prompt.strip():
            raise gr.Error("请提供有效的文本提示")
    elif model_type in ['lightweight', 'full']:
        # 轻量级模型和完整模型支持文本提示或音频提示
        if (not text_prompt or not text_prompt.strip()) and not ref_audio_path:
            raise gr.Error("请提供有效的音频提示或文本提示")
    
    # 根据模型类型选择生成方式
    if model_type == 'lightweight':
        # 使用轻量级模型生成
        logger.info(f"⚡ 使用轻量级模型生成音乐，模型类型: {lightweight_model_type}")
        lightweight_gen = LightweightMusicGenerator(device=device)
        output_path = lightweight_gen.generate_with_lightweight_model(
            text_prompt=text_prompt,
            wav_path=ref_audio_path,
            duration=int(Music_Duration.replace('s', '')),
            steps=steps,
            model_type=lightweight_model_type
        )
    elif model_type == 'ai_api':
        # 使用AI API生成
        logger.info(f"🌐 使用AI API生成音乐，提供商: {api_provider}")
        ai_gen = AIApiMusicGenerator(device=device)
        output_path = ai_gen.generate(
            text_prompt=text_prompt,
            duration=int(Music_Duration.replace('s', '')),
            api_provider=api_provider
        )
    else:
        # 使用DiffRhythm模型生成
        logger.info("📦 使用DiffRhythm模型生成音乐")
        # Set up model parameters based on duration
        if Music_Duration == '95s':
            max_frames = 2048
            repo_id = "ASLP-lab/DiffRhythm-base"
            logger.info("⏱️  选择 95秒 模型 (DiffRhythm-base)")
        else:  # '285s'
            max_frames = 6144
            repo_id = "ASLP-lab/DiffRhythm-full"
            logger.info("⏱️  选择 285秒 模型 (DiffRhythm-full)")
        
        logger.info(f"📊 模型参数: 最大帧数={max_frames}, 模型仓库={repo_id}")
        
        # 使用DiffRhythmGenerator生成器
        diffrhythm_gen = DiffRhythmGenerator(device=device)
        diffrhythm_gen.load_model(max_frames, repo_id)
        
        try:
            # 生成音乐
            generated_song = diffrhythm_gen.generate_with_progress(
                lrc=lrc,
                ref_prompt=text_prompt if text_prompt and text_prompt.strip() else None,
                ref_audio_path=ref_audio_path,
                chunked=True,
                num_inference_steps=steps,
                temperature=temperature,
                top_p=top_p
            )
        except Exception as e:
            logger.error(f"❌ DiffRhythm生成失败: {str(e)}")
            raise gr.Error(f"DiffRhythm生成失败: {str(e)}")
        
        # Save the generated song to a file
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_filename = f"diffrhythm_{timestamp}.{file_type}"
        output_path = os.path.join(output_dir, output_filename)
        
        logger.info(f"💾 正在保存音频文件: {output_path}")
        torchaudio.save(output_path, generated_song, sample_rate=44100)
    
    # 计算总耗时
    total_time = time.time() - total_start_time
    
    logger.info("=" * 60)
    logger.info(f"🎉 音乐生成完成！")
    logger.info(f"📊 总耗时: {total_time:.2f} 秒")
    logger.info(f"📁 输出文件: {output_path}")
    logger.info(f"🎵 音频格式: {file_type}")
    logger.info(f"🔢 随机种子: {seed}")
    logger.info(f"⏱️  音乐时长: {Music_Duration}")
    logger.info("=" * 60)
    
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

def create_language_tab(lang=DEFAULT_LANGUAGE):
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
                    # 模型类型选择
                    model_type = gr.Radio(
                        ["diffrhythm", "lightweight", "ai_api"], 
                        label="模型类型" if lang == "zh" else "Model Type",
                        value="lightweight",  # 将轻量级模型设为默认选项
                        interactive=True,
                        elem_id="model_type_radio"
                    )
                    
                    # 轻量级模型选择
                    lightweight_model_type = gr.Dropdown(
                        ["musicgen-small", "musicgen-melody"],
                        label="轻量级模型" if lang == "zh" else "Lightweight Model",
                        value="musicgen-small",
                        interactive=True,
                        elem_id="lightweight_model_dropdown"
                    )
                    
                    # API提供商选择
                    api_provider = gr.Dropdown(
                        ["free", "baidu", "tencent"],
                        label="API提供商" if lang == "zh" else "API Provider",
                        value="free",
                        interactive=True,
                        elem_id="api_provider_dropdown"
                    )
                    
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
    
    return lrc, audio_prompt, text_prompt, current_prompt_type, seed, randomize_seed, steps, cfg_strength, temperature, top_p, file_type, odeint_method, Music_Duration, model_type, api_provider, lightweight_model_type, lyrics_btn, audio_output

# Create the Gradio interface
try:
    logger.info("🔧 正在创建Gradio界面...")
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
            logger.info("   • 创建英文界面...")
            lrc_en, audio_prompt_en, text_prompt_en, current_prompt_type_en, seed_en, randomize_seed_en, steps_en, cfg_strength_en, temperature_en, top_p_en, file_type_en, odeint_method_en, Music_Duration_en, model_type_en, api_provider_en, lightweight_model_type_en, lyrics_btn_en, audio_output_en = create_language_tab("en")
            
            # Chinese interface
            logger.info("   • 创建中文界面...")
            lrc_zh, audio_prompt_zh, text_prompt_zh, current_prompt_type_zh, seed_zh, randomize_seed_zh, steps_zh, cfg_strength_zh, temperature_zh, top_p_zh, file_type_zh, odeint_method_zh, Music_Duration_zh, model_type_zh, api_provider_zh, lightweight_model_type_zh, lyrics_btn_zh, audio_output_zh = create_language_tab("zh")
        
        # Connect the generate buttons to the inference function
        logger.info("   • 连接生成按钮...")
        lyrics_btn_en.click(
            fn=infer_music,
            inputs=[lrc_en, audio_prompt_en, text_prompt_en, current_prompt_type_en, seed_en, randomize_seed_en, steps_en, cfg_strength_en, temperature_en, top_p_en, file_type_en, odeint_method_en, Music_Duration_en, model_type_en, api_provider_en, lightweight_model_type_en],
            outputs=audio_output_en
        )
        
        lyrics_btn_zh.click(
            fn=infer_music,
            inputs=[lrc_zh, audio_prompt_zh, text_prompt_zh, current_prompt_type_zh, seed_zh, randomize_seed_zh, steps_zh, cfg_strength_zh, temperature_zh, top_p_zh, file_type_zh, odeint_method_zh, Music_Duration_zh, model_type_zh, api_provider_zh, lightweight_model_type_zh],
            outputs=audio_output_zh
        )
    logger.info("✅ Gradio界面创建成功")
except Exception as e:
    logger.error(f"❌ Gradio界面创建失败: {e}", exc_info=True)
    sys.exit(1)

def signal_handler(signum, frame):
    """处理中断信号的函数"""
    logger.info("\n" + "=" * 60)
    logger.info("⚠️  接收到中断信号，正在优雅退出...")
    logger.info("=" * 60)
    
    # 清理资源
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("✅ 资源清理完成，程序退出")
    sys.exit(0)

if __name__ == "__main__":
    # 注册信号处理器，支持 Ctrl+C 中断
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("📋 使用说明:")
    logger.info("   • 按 Ctrl+C 可以中断程序运行")
    logger.info("   • 音乐生成过程中可以随时中断")
    logger.info("   • 中断后会自动清理 GPU 内存")
    logger.info("=" * 60)
    
    try:
        logger.info("🚀 正在启动Gradio服务...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            debug=True,
            share=False,
            css=css
        )
    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("👋 用户主动中断程序，感谢使用！")
        logger.info("=" * 60)
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ 程序运行出错: {str(e)}", exc_info=True)
        sys.exit(1)
