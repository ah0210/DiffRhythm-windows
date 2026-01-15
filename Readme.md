
<p align="center">
   <h1>Di♪♪Rhythm: Blazingly Fast and Embarrassingly Simple</br>End-to-End Full-Length Song Generation with Latent Diffusion</h1>
</p>

Ziqian Ning, Huakang Chen, Yuepeng Jiang, Chunbo Hao, Guobin Ma, Shuai Wang, Jixun Yao, Lei Xie†

<p align="center">
 <a href="https://huggingface.co/spaces/ASLP-lab/DiffRhythm"> Huggingface Space Demo</a> </a>&nbsp
<br>
📑 <a href="https://arxiv.org/abs/2503.01183">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://aslp-lab.github.io/DiffRhythm.github.io/">Demo</a> &nbsp&nbsp | &nbsp&nbsp 💬 <a href="src/contact.md">WeChat (微信)</a>&nbsp&nbsp
</p>

DiffRhythm (Chinese: 谛韵, Dì Yùn) is the ***first*** open-sourced diffusion-based music generation model that is capable of creating full-length songs. The name combines "Diff" (referencing its diffusion architecture) with "Rhythm" (highlighting its focus on music and song creation). The Chinese name 谛韵 (Dì Yùn) phonetically mirrors "DiffRhythm", where "谛" (attentive listening) symbolizes auditory perception, and "韵" (melodic charm) represents musicality.


## News and Updates

* 📌 Join Us on Discord! [![Discord](https://dcbadge.limes.pink/api/server/https://discord.gg/vUD4zgTpJa)](https://discord.gg/vUD4zgTpJa)

* **2025.3.15 🔥** **DiffRhythm-full Official Release: Complete Music Generation!**  

    The wait is over - **285s full-length music generation** is now live!  

    *The symphony evolves. What impossible music will you compose next?*

* **2025.3.11 💻** DiffRhythm can now run on MacOS! 

* **2025.3.9 🔥** **DiffRhythm Update: Text-to-Music and Pure Music Generation!**  

    We're excited to announce two groundbreaking features now live in our open-source music model:  

    🎯 **Text-Based Style Prompts**  
    Describe styles/scenes in words (e.g., `Jazzy Nightclub Vibe`, `Pop Emotional Piano` or `Indie folk ballad, coming-of-age themes, acoustic guitar picking with harmonica interludes`) — *no audio reference needed!*  

    🎧 **Instrumental Mode**  
    Generate pure music with wild prompts like:  
    ```bash  
    "Arctic research station, theremin auroras dancing with geomagnetic storms"  
    ```

    ✨ Special Thanks to community contributor @Jourdelune for implementing these features via #PR29!

    **Full Release Notes**: See [src/update_alert.md](src/update_alert.md) for  details, demos, and roadmap.

    Break the rules. Make music that shouldn't exist.

* **2025.3.7 🔥** **DiffRhythm** is now officially licensed under the **Apache 2.0 License**! 🎉 As the first diffusion-based music generation model, DiffRhythm opens up exciting new possibilities for AI-driven creativity in music. Whether you're a researcher, developer, or music enthusiast, we invite you to explore, innovate, and build upon this foundation. 

* **2025.3.6 🔥** The local deployment guide is now available.

* **2025.3.4 🔥** We released the [DiffRhythm paper](https://arxiv.org/abs/2503.01183) and [Huggingface Space demo](https://huggingface.co/spaces/ASLP-lab/DiffRhythm).

## Model Versions

|  Model   | Type | Description |
|  ----  | ---- | ---- |
| DiffRhythm-base (1m35s)  | Full | Original diffusion model, requires GPU |
| DiffRhythm-full (4m45s)  | Full | Complete music generation model |
| DiffRhythm-vae  | VAE | Variational autoencoder for music encoding |
| MusicGen-small | Lightweight | Fast, CPU-friendly music generation |
| MusicGen-melody | Lightweight | Melody-focused music generation |
| AI API | API | Cloud-based music generation |

## Features

### 🎵 Core Features
- **Full-length music generation** (up to 4m45s)
- **Text-to-music** generation using style prompts
- **Audio-to-music** generation using reference audio
- **Lightweight model** support for CPU and low-resource devices
- **AI API** integration for cloud-based generation

### 🖥️ Device Support
- **NVIDIA GPU**: CUDA support
- **AMD GPU**: DirectML support (Windows) / ROCm support (Linux)
- **Apple Silicon**: MPS support
- **CPU**: Full compatibility

### 🌐 Multi-language Support
- **中文**: 中文界面和提示词支持
- **English**: English interface and prompt support

### 🛠️ Easy to Use
- **Gradio Web UI**: User-friendly interface
- **Modular design**: Easy to extend and customize
- **Auto device detection**: Automatically uses the best available device

## Installation

### System Requirements
- **Operating System**: Windows 10/11, macOS, Linux
- **Python**: 3.10 or higher
- **GPU**: Optional, but recommended for full model

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/ASLP-lab/DiffRhythm.git
   cd DiffRhythm
   ```

2. **Create a virtual environment**
   ```bash
   # Using Python venv (recommended for Windows)
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Additional dependencies for Windows**
   - Install [eSpeak NG](https://github.com/espeak-ng/espeak-ng/releases) for phonetic conversion
   - Set environment variables:
     ```
     PHONEMIZER_ESPEAK_LIBRARY -> C:\Program Files\eSpeak NG\libespeak-ng.dll
     PHONEMIZER_ESPEAK_PATH -> C:\Program Files\eSpeak NG
     ```

### GPU Support Configuration

#### NVIDIA GPU (CUDA)
```bash
pip install -r requirements_cuda.txt
```

#### AMD GPU (DirectML, Windows)
```bash
# Create a DirectML-specific environment
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

## Usage

### 🚀 Quick Start

1. **Start the Web UI**
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to `http://localhost:7860`

3. **Choose a model type**:
   - **DiffRhythm**: Full diffusion model (requires GPU)
   - **Lightweight**: Fast, CPU-friendly model (MusicGen)
   - **AI API**: Cloud-based generation

4. **Enter your prompt** and click "Generate"!

### 📝 Example Prompts

#### Text Prompts
- `Electronic Dance Music with strong beat and synthesizer melody`
- `Classical piano solo with emotional melody`
- `Jazz quartet with saxophone and double bass`

#### Audio Prompts
- Upload a reference audio file to guide the generation
- The model will learn the style and generate similar music

### ⚙️ Advanced Settings

- **Duration**: Adjust the length of the generated music
- **Steps**: More steps = better quality, but slower generation
- **Temperature**: Controls randomness (lower = more deterministic)
- **Top-p**: Controls diversity (higher = more diverse)

## Project Structure

```
DiffRhythm/
├── config/              # Configuration files
├── dataset/             # Training dataset
├── g2p/                 # Grapheme-to-phoneme conversion
├── infer/               # Inference modules
│   ├── ai_api_infer.py      # AI API generation
│   ├── diffrhythm_infer.py  # DiffRhythm generation
│   ├── lightweight_infer.py  # Lightweight model generation
│   └── infer_utils.py        # Inference utilities
├── model/               # Model architecture
├── scripts/             # Utility scripts
├── src/                 # Source files and resources
├── train/               # Training code
├── app.py               # Main application
├── i18n.py              # Multi-language support
├── requirements.txt     # Dependencies
└── test_device.py       # Device detection test
```

## Development

### Model Loading

#### Lightweight Model
```python
from infer.lightweight_infer import LightweightMusicGenerator

# Load the model
generator = LightweightMusicGenerator(device='cpu')
generator.load_lightweight_model("musicgen-small")

# Generate music
audio_path = generator.generate_with_lightweight_model(
    text_prompt="Electronic Dance Music",
    duration=10,
    model_type="musicgen-small"
)
```

#### AI API
```python
from infer.ai_api_infer import AIApiMusicGenerator

# Initialize API generator
generator = AIApiMusicGenerator()

# Generate music using API
audio_path = generator.generate_with_api(
    text_prompt="Pop Music",
    duration=10,
    api_provider="free"
)
```

## Testing

### Device Detection
```bash
python test_device.py
```

### Lightweight Model
```bash
python test_lightweight_api.py
```

## Troubleshooting

### Common Issues

1. **DirectML Device Not Detected**
   - Ensure `torch-directml` is installed
   - Check if your AMD GPU is DirectML-compatible
   - Run `test_device.py` to diagnose

2. **Music Sounds Like Noise**
   - Try a more specific text prompt
   - Adjust generation parameters (lower temperature)
   - Reduce the generation duration

3. **Model Loading Error**
   - Ensure you have enough disk space for model downloads
   - Check your internet connection for model downloads
   - Verify Python version compatibility (3.10+)

## License & Disclaimer

DiffRhythm (code and DiT weights) is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). This open-source license allows you to freely use, modify, and distribute the model, as long as you include the appropriate copyright notice and disclaimer.

We do not make any profit from this model. Our goal is to provide a high-quality base model for music generation, fostering innovation in AI music and contributing to the advancement of human creativity. We hope that DiffRhythm will serve as a foundation for further research and development in the field of AI-generated music.

DiffRhythm enables the creation of original music across diverse genres, supporting applications in artistic creation, education, and entertainment. While designed for positive use cases, potential risks include unintentional copyright infringement through stylistic similarities, inappropriate blending of cultural musical elements, and misuse for generating harmful content. To ensure responsible deployment, users must implement verification mechanisms to confirm musical originality, disclose AI involvement in generated works, and obtain permissions when adapting protected styles.

## Citation
```
@article{ning2025diffrhythm,
  title={{DiffRhythm}: Blazingly Fast and Embarrassingly Simple End-to-End Full-Length Song Generation with Latent Diffusion},
  author={Ziqian, Ning and Huakang, Chen and Yuepeng, Jiang and Chunbo, Hao and Guobin, Ma and Shuai, Wang and Jixun, Yao and Lei, Xie},
  journal={arXiv preprint arXiv:2503.01183},
  year={2025}
}
```

## Contact Us

If you are interested in leaving a message to our research team, feel free to email `nzqiann@gmail.com`.
<p align="center">
    <a href="http://www.nwpu-aslp.org/">
        <img src="src/ASLP.jpg" width="400"/>
    </a>
</p>