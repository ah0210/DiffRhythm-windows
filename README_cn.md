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

## 待办事项
- [ ] 动态长度控制
- [ ] 仅人声
- [ ] 歌曲扩展
- [ ] 支持 Colab。
- [ ] 支持 Docker。
- [x] 发布 DiffRhythm-full。
- [x] 发布训练代码。
- [x] 支持本地部署。
- [x] 发布论文到 Arxiv。
- [x] 在 Hugging Face Space 上在线服务。

## 模型版本

|  模型   | HuggingFace |
|  ----  | ----  |
| DiffRhythm-base (1分35秒)  | https://huggingface.co/ASLP-lab/DiffRhythm-base |
| DiffRhythm-full (4分45秒)  | https://huggingface.co/ASLP-lab/DiffRhythm-full |
| DiffRhythm-vae  | https://huggingface.co/ASLP-lab/DiffRhythm-vae |

## 推理

按照以下步骤克隆仓库并安装环境。

```bash 
# 克隆并进入仓库
git clone https://github.com/ASLP-lab/DiffRhythm.git
cd DiffRhythm

# 安装环境

## espeak-ng
# 对于 Debian-like 发行版（如 Ubuntu、Mint 等）
sudo apt-get install espeak-ng
# 对于 RedHat-like 发行版（如 CentOS、Fedora 等） 
sudo yum install espeak-ng
# 对于 MacOS
brew install espeak-ng
# 对于 Windows
# 请访问 https://github.com/espeak-ng/espeak-ng/releases 下载 .msi 安装程序


## 创建 Python 环境
conda create -n diffrhythm python=3.10
conda activate diffrhythm

## 或者您可以使用经典的 Python 虚拟环境代替 conda
python -m venv venv
# 在 Linux 上激活 venv
source venv/bin/activate
# 在 Windows 上激活 venv
venv\Scripts\activate
.\venv\Scripts\Activate.ps1

## 安装依赖
pip install -r requirements.txt
```

在 Linux 上，您现在可以简单地使用推理脚本：
```bash
# 使用参考 WAV 文件进行推理
bash scripts/infer_wav_ref.sh
# 使用文本提示参考进行推理
bash scripts/infer_prompt_ref.sh
```

但在 Windows 上运行推理之前，请确保设置了用户环境变量：\
`PHONEMIZER_ESPEAK_LIBRARY` -> `C:\Program Files\eSpeak NG\libespeak-ng.dll`\
`PHONEMIZER_ESPEAK_PATH` -> `C:\Program Files\eSpeak NG`\
将 `C:\Program Files\eSpeak NG` 更改为您的 eSpeak 安装目录，然后重启电脑以应用更改。

*在 Windows 上运行时，不再需要安装日语语音、mbrola 二进制文件和解压 mbrola_ph 文件夹（如[此处](https://github.com/ASLP-lab/DiffRhythm/issues/15)和[此处](https://github.com/ASLP-lab/DiffRhythm/issues/22)所述）。参见 https://github.com/ASLP-lab/DiffRhythm/issues/17#issuecomment-2705058729、[此提交](https://github.com/ASLP-lab/DiffRhythm/commit/2ea9424274df10670ddc613b5d61cc16d13e2b88)和[此提交](https://github.com/ASLP-lab/DiffRhythm/commit/1ad7229e1a774c9a2a0c4888103dd4ea7176aebb)。*

在此之后，您也可以在 Windows 上运行推理脚本（请注意这里将使用英文歌词）：
```batch
rem : 使用参考 WAV 文件进行推理
call scripts\infer_wav_ref.bat
rem : 使用文本提示参考进行推理
call scripts\infer_prompt_ref.bat
```

lrc 和参考音频的示例文件可以在 `infer/example` 中找到。

您可以使用我们在 huggingface 上提供的[工具](https://huggingface.co/spaces/ASLP-lab/DiffRhythm)来生成 lrc。

**注意 DiffRhythm-base 至少需要 8G 的显存。为了满足 8G 显存要求，运行推理时使用 `--chunked` 参数。如果禁用分块解码，可能需要更高的显存。**

## 训练

即将推出...

## 许可证和免责声明

DiffRhythm（代码和 DiT 权重）在 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) 下发布。此开源许可证允许您自由使用、修改和分发模型，只要您包含适当的版权声明和免责声明。

我们不会从此模型中获利。我们的目标是提供高质量的音乐生成基础模型，促进 AI 音乐的创新，为人类创造力的进步做出贡献。我们希望 DiffRhythm 能够成为 AI 生成音乐领域进一步研究和开发的基础。

DiffRhythm 能够创作跨多种流派的原创音乐，支持艺术创作、教育和娱乐领域的应用。虽然设计用于积极用例，但潜在风险包括通过风格相似性无意中侵犯版权、不当混合文化音乐元素以及误用生成有害内容。为确保负责任的部署，用户必须实施验证机制以确认音乐原创性，在生成的作品中披露 AI 参与，并在改编受保护风格时获得许可。

## 引用
```
@article{ning2025diffrhythm,
  title={{DiffRhythm}: Blazingly Fast and Embarrassingly Simple End-to-End Full-Length Song Generation with Latent Diffusion},
  author={Ziqian, Ning and Huakang, Chen and Yuepeng, Jiang and Chunbo, Hao and Guobin, Ma and Shuai, Wang and Jixun, Yao and Lei, Xie},
  journal={arXiv preprint arXiv:2503.01183},
  year={2025}
}
```
## 联系我们

如果您有兴趣给我们的研究团队留言，请发送邮件至 `nzqiann@gmail.com`。
<p align="center">
    <a href="http://www.nwpu-aslp.org/">
        <img src="src/ASLP.jpg" width="400"/>
    </a>
</p>
