<p align="center">
    <img src="src/DiffRhythm_logo.jpg" width="400"/>
<p>

<p align="center">
   <h1>Di♪♪Rhythm：极速且简单的端到端</br>基于潜在扩散的全长歌曲生成</h1>
</p>

Ziqian Ning, Huakang Chen, Yuepeng Jiang, Chunbo Hao, Guobin Ma, Shuai Wang, Jixun Yao, Lei Xie†

<p align="center">
 <a href="https://huggingface.co/spaces/ASLP-lab/DiffRhythm"> Huggingface Space 演示</a> </a>&nbsp
<br>
📑 <a href="https://arxiv.org/abs/2503.01183">论文</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://aslp-lab.github.io/DiffRhythm.github.io/">演示</a> &nbsp&nbsp | &nbsp&nbsp 💬 <a href="src/contact.md">微信</a>&nbsp&nbsp
</p>

DiffRhythm（中文名：谛韵，Dì Yùn）是***首个***能够创作全长歌曲的开源基于扩散的音乐生成模型。该名称结合了"Diff"（引用其扩散架构）和"Rhythm"（突出其对音乐和歌曲创作的关注）。中文名谛韵（Dì Yùn）在发音上与"DiffRhythm"相呼应，其中"谛"（专注聆听）象征听觉感知，"韵"（旋律魅力）代表音乐性。



<p align="center">
    <img src="src/diffrhythm.jpg" width="90%"/>
<p>

## 新闻和更新

* 📌 加入我们的 Discord！[![Discord](https://dcbadge.limes.pink/api/server/https://discord.gg/vUD4zgTpJa)](https://discord.gg/vUD4zgTpJa)

* **2025.3.15 🔥** **DiffRhythm-full 正式发布：完整音乐生成！**  

    等待结束了——**285秒全长音乐生成**现已上线！  

    *交响乐在演进。你将创作出什么不可能的音乐？*

* **2025.3.11 💻** DiffRhythm 现在可以在 MacOS 上运行！ 

* **2025.3.9 🔥** **DiffRhythm 更新：文本到音乐和纯音乐生成！**  

    我们很高兴地宣布两个突破性功能现已上线：  

    🎯 **基于文本的风格提示**  
    用文字描述风格/场景（例如，`Jazzy Nightclub Vibe`、`Pop Emotional Piano` 或 `Indie folk ballad, coming-of-age themes, acoustic guitar picking with harmonica interludes`）——*无需音频参考！*  

    🎧 **纯音乐模式**  
    使用狂野的提示生成纯音乐，例如：  
    ```bash  
    "Arctic research station, theremin auroras dancing with geomagnetic storms"  
    ```

    ✨ 特别感谢社区贡献者 @Jourdelune 通过 #PR29 实现了这些功能！

    **完整发布说明**：查看 [src/update_alert.md](src/update_alert.md) 了解详细信息、演示和路线图。

    打破规则。创作不存在的音乐。

* **2025.3.7 🔥** **DiffRhythm** 现在正式采用 **Apache 2.0 许可证**！🎉 作为首个基于扩散的音乐生成模型，DiffRhythm 为 AI 驱动的音乐创意开辟了令人兴奋的新可能性。无论您是研究人员、开发者还是音乐爱好者，我们都邀请您探索、创新并在此基础上构建。

* **2025.3.6 🔥** 本地部署指南现已提供。

* **2025.3.4 🔥** 我们发布了 [DiffRhythm 论文](https://arxiv.org/abs/2503.01183) 和 [Huggingface Space 演示](https://huggingface.co/spaces/ASLP-lab/DiffRhythm)。

## 模型版本

|  模型   | 类型 | 描述 |
|  ----  | ---- | ---- |
| DiffRhythm-base (1分35秒)  | 完整 | 原始扩散模型，需要GPU |
| DiffRhythm-full (4分45秒)  | 完整 | 完整音乐生成模型 |
| DiffRhythm-vae  | VAE | 用于音乐编码的变分自编码器 |
| MusicGen-small | 轻量级 | 快速，CPU友好的音乐生成 |
| MusicGen-melody | 轻量级 | 专注于旋律的音乐生成 |
| AI API | API | 基于云的音乐生成 |

## 特性

### 🎵 核心特性
- **全长音乐生成**（最长可达4分45秒）
- **文本到音乐** 使用风格提示词生成
- **音频到音乐** 使用参考音频生成
- **轻量级模型** 支持CPU和低资源设备
- **AI API** 集成用于云端生成

### 🖥️ 设备支持
- **NVIDIA GPU**：CUDA支持
- **AMD GPU**：DirectML支持（Windows）/ ROCm支持（Linux）
- **Apple Silicon**：MPS支持
- **CPU**：完全兼容

### 🌐 多语言支持
- **中文**：中文界面和提示词支持
- **English**：英文界面和提示词支持

### 🛠️ 易于使用
- **Gradio Web UI**：用户友好的界面
- **模块化设计**：易于扩展和定制
- **自动设备检测**：自动使用最佳可用设备

## 安装

### 系统要求
- **操作系统**：Windows 10/11, macOS, Linux
- **Python**：3.10或更高版本
- **GPU**：可选，但推荐用于完整模型

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/ASLP-lab/DiffRhythm.git
   cd DiffRhythm
   ```

2. **创建虚拟环境**
   ```bash
   # 使用Python venv（推荐Windows）
   python -m venv venv
   
   # 在Windows上激活
   venv\Scripts\activate
   
   # 在macOS/Linux上激活
   source venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **Windows系统的额外依赖**
   - 安装 [eSpeak NG](https://github.com/espeak-ng/espeak-ng/releases) 用于语音转换
   - 设置环境变量：
     ```
     PHONEMIZER_ESPEAK_LIBRARY -> C:\Program Files\eSpeak NG\libespeak-ng.dll
     PHONEMIZER_ESPEAK_PATH -> C:\Program Files\eSpeak NG
     ```

### GPU支持配置

#### NVIDIA GPU (CUDA)
```bash
pip install -r requirements_cuda.txt
```

#### AMD GPU (DirectML, Windows)
```bash
# 创建DirectML特定环境
python -m venv venv_amd
venv_amd\Scripts\activate
pip install torch-directml
pip install -r requirements.txt
```

#### AMD GPU (ROCm, Linux)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
pip install -r requirements.txt
```

## 使用

### 🚀 快速开始

1. **启动Web UI**
   ```bash
   python app.py
   ```

2. **打开浏览器** 访问 `http://localhost:7860`

3. **选择模型类型**：
   - **DiffRhythm**：完整扩散模型（需要GPU）
   - **Lightweight**：快速，CPU友好的模型（MusicGen）
   - **AI API**：基于云的生成

4. **输入提示词** 并点击"生成"！

### 📝 示例提示词

#### 文本提示词
- `电子舞曲，带有强烈的节拍和合成器旋律`
- `古典钢琴独奏，带有情感化的旋律`
- `爵士四重奏，带有萨克斯风和低音提琴`

#### 音频提示词
- 上传参考音频文件来指导生成
- 模型将学习风格并生成类似的音乐

### ⚙️ 高级设置

- **时长**：调整生成音乐的长度
- **步数**：更多步数=更好质量，但生成更慢
- **温度**：控制随机性（越低=越确定）
- **Top-p**：控制多样性（越高=更多样化）

## 项目结构

```
DiffRhythm/
├── config/              # 配置文件
├── dataset/             # 训练数据集
├── g2p/                 # 音素转换
├── infer/               # 推理模块
│   ├── ai_api_infer.py      # AI API生成
│   ├── diffrhythm_infer.py  # DiffRhythm生成
│   ├── lightweight_infer.py  # 轻量级模型生成
│   └── infer_utils.py        # 推理工具
├── model/               # 模型架构
├── scripts/             # 工具脚本
├── src/                 # 源文件和资源
├── train/               # 训练代码
├── app.py               # 主应用
├── i18n.py              # 多语言支持
├── requirements.txt     # 依赖
└── test_device.py       # 设备检测测试
```

## 开发

### 模型加载

#### 轻量级模型
```python
from infer.lightweight_infer import LightweightMusicGenerator

# 加载模型
generator = LightweightMusicGenerator(device='cpu')
generator.load_lightweight_model("musicgen-small")

# 生成音乐
audio_path = generator.generate_with_lightweight_model(
    text_prompt="电子舞曲",
    duration=10,
    model_type="musicgen-small"
)
```

#### AI API
```python
from infer.ai_api_infer import AIApiMusicGenerator

# 初始化API生成器
generator = AIApiMusicGenerator()

# 使用API生成音乐
audio_path = generator.generate_with_api(
    text_prompt="流行音乐",
    duration=10,
    api_provider="free"
)
```

## 测试

### 设备检测
```bash
python test_device.py
```

### 轻量级模型
```bash
python test_lightweight_api.py
```

## 故障排除

### 常见问题


toolName: todo_write
            
status: success
          
Todos updated: 5 items
          

        
            
toolName: view_files
            
status: success
          
            
filePath: e:\mygit\2026\DiffRhythm-windows\infer\lightweight_infer.py
          

        
            
toolName: search_codebase
            
status: success
          
query: app.py 中 infer_music 对 lightweight 的 Music_Duration / 音频提示处理位置
          

        
            
toolName: apply_patch
            
status: failed
          
          

        
            
toolName: edit_file_update
            
status: success
          
            
filePath: e:/mygit/2026/DiffRhythm-windows/infer/lightweight_infer.py
          
changes: 
          

        
            
toolName: view_files
            
status: success
          
            
filePath: e:\mygit\2026\DiffRhythm-windows\app.py
          

        
            
toolName: apply_patch
            
status: failed
          
          

        
            
toolName: edit_file_update
            
status: success
          
            
filePath: e:/mygit/2026/DiffRhythm-windows/infer/lightweight_infer.py
          
changes: 
          

        
            
toolName: apply_patch
            
status: failed
          
          

        
            
toolName: edit_file_update
            
status: success
          
            
filePath: e:/mygit/2026/DiffRhythm-windows/app.py
          
changes: 
          

        
            
toolName: apply_patch
            
status: failed
          
          

        
            
toolName: edit_file_update
            
status: success
          
            
filePath: e:/mygit/2026/DiffRhythm-windows/test_lightweight_api.py
          
changes: 
          

        
            
toolName: GetDiagnostics
            
status: success
          
          

        
            
toolName: todo_write
            
status: success
          
Todos updated: 5 items
          

        
            
toolName: run_command
            
status: success
          
command: .\venv\Scripts\python.exe test_lightweight_api.py undefined
          

        
            
toolName: apply_patch
            
status: failed
          
          

        
            
toolName: edit_file_update
            
status: success
          
            
filePath: e:/mygit/2026/DiffRhythm-windows/infer/lightweight_infer.py
          
changes: 
          

        
            
toolName: GetDiagnostics
            
status: success
          
          

        
            
toolName: run_command
            
status: success
          
command: .\venv\Scripts\python.exe test_lightweight_api.py undefined
          

        
            
toolName: run_command
            
status: success
          
command: .\venv\Scripts\python.exe -c "import os,wave; from infer.lightweight_infer import LightweightMusicGenerator as G; g=G(device='cpu'); p=g.generate_with_lightweight_model(text_prompt='Orchestral cinematic, 120 bpm, strings, brass, drums, clear rhythm', duration=15, model_type='musicgen-small'); print('path',p); wf=wave.open(p,'rb'); print('sr',wf.getframerate(),'frames',wf.getnframes(),'sec',wf.getnframes()/wf.getframerate()); wf.close()" undefined
          

        
            
toolName: todo_write
            
status: success
          
Todos updated: 1 items
          

        
            
toolName: todo_write
            
status: success
          
Todos updated: 5 items
          

        
            
toolName: run_command
            
status: running
          
command: .\venv\Scripts\python.exe app.py undefined
          

        
            
toolName: check_command_status
            
status: success
          
          

        
            
toolName: check_command_status
            
status: success
          
          

        
            
toolName: open_preview
            
status: success
          
预览
          
有，而且我已经把最影响“只出呜声/时长不对/音频提示不生效”的几个点直接优化掉了。

**这次改动解决了什么**
- 轻量级模型（musicgen-small）之前把 token 上限卡在 `2048/4=512`，所以你输入 95s 最终只能得到 ~10s；现在改为按目标时长计算 token，并在 CPU 上允许到约 30s（例如 15s 能稳定生成 14.9s）。
- 轻量级“音频提示”之前基本等同于走回退波形（容易是“呜声”）；现在 `musicgen-melody` 支持真正用音频提示生成。
- 后处理增加了 DC 去除 + 高通/低通滤波，并在必要时裁剪到目标时长，能明显减少低频嗡声与“时长漂移”。
- Web UI 里如果 lightweight 选了 95s/285s，会自动限制到 30s；如果只给音频提示却选了 `musicgen-small`，会明确提示你切到 `musicgen-melody`（避免 silent fallback 出“呜声”）。

**修改位置**
- 轻量级生成逻辑与后处理：[lightweight_infer.py](file:///e:/mygit/2026/DiffRhythm-windows/infer/lightweight_infer.py)
- Web UI 对 lightweight 时长/音频提示的限制：[app.py](file:///e:/mygit/2026/DiffRhythm-windows/app.py)
- 测试用例同步（音频提示改为走 musicgen-melody）：[test_lightweight_api.py](file:///e:/mygit/2026/DiffRhythm-windows/test_lightweight_api.py)

**你现在怎么用（建议）**
- 想用“参考音频”引导：轻量级模型请选择 `musicgen-melody`，并最好同时给一个文本提示词。
- 想生成更长更像“整首歌”的：用 DiffRhythm（建议 CUDA GPU），lightweight 仍然更适合 10–30 秒片段。