# Copyright (c) 2025 ASLP-LAB
#               2025 Ziqian Ning   (ningziqian@mail.nwpu.edu.cn)
#               2025 Huakang Chen  (huakang@mail.nwpu.edu.cn)
#               2025 Guobin Ma     (guobin.ma@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" This implementation is adapted from github repo:
    https://github.com/SWivid/F5-TTS.
"""

from __future__ import annotations
from typing import Callable
from random import random

import torch
from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torchdiffeq import odeint

from model.utils import (
    exists,
    list_str_to_idx,
    list_str_to_tensor,
    lens_to_mask,
    mask_from_frac_lengths,
)

def custom_mask_from_start_end_indices(start: int["b"], end: int["b"], device, max_seq_len):  # noqa: F722 F821
    max_seq_len = max_seq_len
    seq = torch.arange(max_seq_len, device=device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask

class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            method="euler"
        ),
        odeint_options: dict = dict(
            min_step=0.01  # 减小最小步长
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        style_drop_prob=0.1,
        lrc_drop_prob=0.1,
        num_channels=None,
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
        max_frames=2048
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.style_drop_prob = style_drop_prob
        self.lrc_drop_prob = lrc_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs
        
        self.odeint_options = odeint_options

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map
        
        self.max_frames = max_frames

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        style_prompt = None,
        style_prompt_lens = None,
        negative_style_prompt = None,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=4.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=6144,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
        start_time=None,
        latent_pred_start_frame=0,
        latent_pred_end_frame=2048,
        progress_callback=None,
    ):
        self.eval()

        if next(self.parameters()).dtype == torch.float16:
            cond = cond.half()

        # raw wave
        if cond.shape[1] > duration:
            cond = cond[:, :duration, :]

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # duration
        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        latent_pred_start_frame = torch.tensor([latent_pred_start_frame]).to(cond.device)
        latent_pred_end_frame = duration
        latent_pred_end_frame = torch.tensor([latent_pred_end_frame]).to(cond.device)
        fixed_span_mask = custom_mask_from_start_end_indices(latent_pred_start_frame, latent_pred_end_frame, device=cond.device, max_seq_len=duration)

        fixed_span_mask = fixed_span_mask.unsqueeze(-1)
        step_cond = torch.where(fixed_span_mask, torch.zeros_like(cond), cond)

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # test for no ref audio
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        def fn(t, x):
            # predict flow
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=False, drop_text=False, drop_prompt=False,
                style_prompt=style_prompt, style_prompt_lens=style_prompt_lens, start_time=start_time
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=True, drop_text=True, drop_prompt=False,
                style_prompt=negative_style_prompt, style_prompt_lens=style_prompt_lens, start_time=start_time
            )
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))
        
        t = torch.linspace(t_start, 1, steps, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        # 添加进度监控的ODE求解器
        if progress_callback is not None:
            # 使用原始的odeint方法，但添加进度监控和超时机制
            import threading
            import time
            
            progress_active = True
            odeint_completed = False
            
            def progress_monitor():
                """进度监控线程"""
                start_time = time.time()
                
                # 根据设备类型调整超时阈值
                if self.device == 'cuda':
                    expected_time_per_step = 1.0  # GPU 每步预计时间（秒）
                else:
                    expected_time_per_step = 30.0  # CPU 每步预计时间（秒），大幅增加
                
                expected_total_time = steps * expected_time_per_step
                
                while progress_active:
                    # 检查是否已完成
                    if odeint_completed:
                        break
                    
                    # 检查是否超时
                    elapsed = time.time() - start_time
                    
                    # 动态调整超时阈值
                    if elapsed > expected_total_time * 2:
                        print(f"⚠️  警告：扩散过程可能已卡住，已运行 {elapsed:.1f} 秒")
                        print(f"   • 当前设备: {self.device}")
                        print(f"   • 建议检查内存使用情况")
                        if self.device != 'cuda':
                            print(f"   • 提示：在CPU上运行会比GPU慢很多，请耐心等待")
                    
                    time.sleep(5.0)  # 每5秒检查一次，减少CPU占用
            
            # 启动进度监控线程
            monitor_thread = threading.Thread(target=progress_monitor)
            monitor_thread.daemon = True  # 设置为守护线程
            monitor_thread.start()
            
            # 在开始前调用进度回调
            progress_callback(0, steps, 0.0)
            
            try:
                # 检查GPU内存使用情况
                if self.device == 'cuda':
                    print(f"🔍 开始ODE求解，共 {steps} 步，使用设备: {self.device}")
                    print(f"   • GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    print(f"   • GPU内存峰值: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
                else:
                    print(f"🔍 开始ODE求解，共 {steps} 步，使用设备: {self.device}")
                    print(f"   • 提示：在CPU上运行会比GPU慢很多，请耐心等待")
                
                # 直接使用odeint求解整个时间序列
                trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
                odeint_completed = True
                print(f"✅ ODE求解完成")
                
            except Exception as e:
                print(f"❌ ODE求解出错: {e}")
                # 尝试使用CPU作为备选方案
                if self.device != 'cpu':
                    print("⚠️  尝试使用CPU进行ODE求解...")
                    try:
                        # 将数据移动到CPU
                        y0_cpu = y0.cpu()
                        t_cpu = t.cpu()
                        
                        # 创建CPU版本的函数
                        def fn_cpu(t_cpu, y_cpu):
                            t_gpu = t_cpu.to(self.device)
                            y_gpu = y_cpu.to(self.device)
                            result = fn(t_gpu, y_gpu)
                            return result.cpu()
                        
                        # 使用CPU求解
                        trajectory_cpu = odeint(fn_cpu, y0_cpu, t_cpu, **self.odeint_kwargs)
                        trajectory = trajectory_cpu.to(self.device)
                        odeint_completed = True
                        print("✅ CPU ODE求解完成")
                            
                    except Exception as e2:
                        print(f"❌ CPU ODE求解也失败: {e2}")
                        print("💡 建议尝试以下解决方案：")
                        print("   1. 减少扩散步数（如从32步减少到16步）")
                        print("   2. 检查GPU内存使用情况")
                        print("   3. 重启程序或系统")
                        raise
                else:
                    raise
            finally:
                # 停止进度监控
                progress_active = False
                odeint_completed = True
                
            # 最终进度更新，确保显示100%
            progress_callback(steps, steps, 1.0)
        else:
            # 使用原始的odeint方法
            trajectory = odeint(fn, y0, t, **self.odeint_kwargs)

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(fixed_span_mask, out, cond)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        style_prompt = None,
        style_prompt_lens = None,
        lens: int["b"] | None = None,  # noqa: F821
        noise_scheduler: str | None = None,
        grad_ckpt = False,
        start_time = None,
    ):

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths, self.max_frames)

        if exists(mask):
            rand_span_mask = mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.normal(mean=0, std=1, size=(batch,), device=self.device)
        time = torch.nn.functional.sigmoid(time)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        drop_text = random() < self.lrc_drop_prob
        drop_prompt = random() < self.style_drop_prob

        # if want rigourously mask out padding, record in collate_fn in dataset.py, and pass in here
        # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
        pred = self.transformer(
            x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, drop_prompt=drop_prompt,
            style_prompt=style_prompt, style_prompt_lens=style_prompt_lens, grad_ckpt=grad_ckpt, start_time=start_time
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred
