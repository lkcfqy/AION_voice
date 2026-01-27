import sys
import os
import time
import torch
import numpy as np
import librosa
import sounddevice as sd
import queue
import threading

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.lsm import AION_LSM_Network
from src.adapter import RandomProjectionAdapter
from src.gwt import GlobalWorkspace
from src.drive import SocialDrive
from src.hrr import HDCWorldModel
from src.mhn import ModernHopfieldNetwork
from src.dashboard import AIONDashboard
from src.config import LSM_N_NEURONS, OBS_SHAPE, AUDIO_SR, HOP_LENGTH, SEED, LSM_STEPS_PER_SAMPLE, WEIGHTS_PATH


class IntegratedAIONAgent:
    def __init__(self, device='cpu'):
        self.device = device
        print("正在初始化 AION 完整认知集成智能体...")
        
        # 1. 动力层 (LSM) & 感知适配器
        # 1. 动力层 (LSM) & 感知适配器
        self.lsm = AION_LSM_Network()
        if os.path.exists(WEIGHTS_PATH):
            print(f"正在加载 LSM 读出层权重 ({WEIGHTS_PATH})...")
            self.lsm.W_out = torch.load(WEIGHTS_PATH)
        else:
             print(f"⚠️ 未找到权重文件 {WEIGHTS_PATH}，将使用随机初始化。")

        self.adapter = RandomProjectionAdapter(device=device)

        # 2. 控制层与存储层 (GWT, HDC, MHN, Drive)
        self.gwt = GlobalWorkspace(device=device)
        self.drive = SocialDrive()
        self.wm = HDCWorldModel(n_actions=1, device=device)
        self.gwt = GlobalWorkspace(device=device)
        self.drive = SocialDrive()
        self.wm = HDCWorldModel(n_actions=1, device=device)
        self.memory = ModernHopfieldNetwork(device=device)
        
        # Dashboard 集成
        try:
             self.dashboard = AIONDashboard()
             self.use_dashboard = True
             print("✅ Visdom 仪表盘连接成功。")
        except Exception as e:
             print(f"⚠️ 无法连接到 Visdom 服务器 ({e})。仪表盘将被禁用。")
             print("   请运行 'python -m visdom.server' 以启用可视化。")
             self.use_dashboard = False
        
        # 3. 状态同步与多线程
        self.input_queue = queue.Queue()
        self.chunk_size = HOP_LENGTH
        self.feedback_factor = 0.05   # 大幅调低反馈，由 0.3 降至 0.05 以防止震荡
        self.last_prediction = np.zeros(OBS_SHAPE)
        self.volume_factor = 0.5      # 全局音量缩放
        self.alpha_smooth = 0.7       # 频谱平滑系数
        self.prev_audio_tail = np.zeros(self.chunk_size) # 用于平滑衔接
        
        # 4. 预计算音频合成所需的 Mel 矩阵
        # 用于将 Mel 转换回 STFT 幅度
        self.mel_basis = librosa.filters.mel(sr=AUDIO_SR, n_fft=self.chunk_size*2, n_mels=OBS_SHAPE)
        self.mel_basis_inv = np.linalg.pinv(self.mel_basis)
        
        # 共享状态（用于跨线程通讯）
        self.shared_state = {
            'cognitive_bias': np.zeros(LSM_N_NEURONS),
            'last_spikes': np.zeros(LSM_N_NEURONS),
            'dopamine': 0.0,
            'running': True
        }
        
    def audio_callback(self, indata, outdata, frames, time, status):
        """快速物理环 (32ms 延迟)"""
        # A. 感知输入
        y = indata.flatten()
        mel = librosa.feature.melspectrogram(y=y, sr=AUDIO_SR, n_mels=OBS_SHAPE, hop_length=self.chunk_size, n_fft=self.chunk_size*2)
        # 转为对数分贝，使用固定参考值 1.0 (必须与训练一致)
        mel_db = librosa.power_to_db(mel, ref=1.0)
        # 确保形状为 (OBS_SHAPE,)，取所有时间帧的平均值
        mel_vec = np.mean(mel_db, axis=1)
        # 归一化 (使用与训练完全一致的 [-80, 0] 映射)
        mel_vec = (mel_vec + 80) / 80.0
        mel_vec = (mel_vec + 80) / 80.0
        mel_vec = np.clip(mel_vec, 0, 1)

        # 实时更新耳蜗视图 (如果启用 Dashboard)
        if self.use_dashboard and hasattr(self, 'dashboard'):
             # 发送这一帧的 Mel 频谱
             # 为了显示好看，将其从 (OBS_SHAPE,) 扩展为 (OBS_SHAPE, 1, 1) 或类似的图像格式
             # Dashboard 期望 (H, W, 3) 
             # 简单的可视化：将向量扩展为条形图
             pass # 在 cognitive loop 更新可能更好，或者在这里更新 fast update
             # 由于 audio callback 频率很高，我们可能需要降采样
             # 暂时只在 dashboard 类中做频谱图累积？
             # Dashboard 的 update_env_view 期望图像。
             # 我们可以简单地把 mel vector 构造成一个热力图条
             
             # 构造一个 图像 (OBS_SHAPE, 10, 3) 用伪彩色
             # 简便起见，只在 cognitive loop 更新慢速信息。
             pass

        # 增加噪声门 (Noise Gate): 如果输入信号太弱，直接置零
        if np.mean(mel_vec) < 0.1: # 稍微调高一点门限
            mel_vec.fill(0)

        # B. 注入与生成 (接受认知偏置和多巴胺调节)
        bias = self.shared_state['cognitive_bias']
        dopamine = self.shared_state['dopamine']
        
        spikes, next_mel = self.lsm.step(mel_vec + (self.feedback_factor * self.last_prediction), 
                                         dopamine=dopamine, 
                                         cognitive_bias=bias)
                                         
        # 增加平滑：避免预测值跳变剧烈导致滋滋声
        self.last_prediction = self.alpha_smooth * self.last_prediction + (1 - self.alpha_smooth) * next_mel
        self.shared_state['last_spikes'] = spikes # 更新脉冲状态供逻辑环采样

        # C. 播放输出
        # 将 Mel 归一化 DB 转回近似幅度 (限制在 0-1 范围内防止爆炸)
        next_mel_safe = np.clip(next_mel, 0, 1)
        mel_db = next_mel_safe * 80.0 - 80.0
        mel_power = librosa.db_to_power(mel_db)
        
        # 手动转回 STFT 幅度 (Linear)
        stft_power = self.mel_basis_inv @ mel_power
        stft_mag = np.sqrt(np.maximum(stft_power, 0))
        
        # 使用 ISTFT 进行实时合成 (零位由于无相位信息)
        audio_out = librosa.istft(stft_mag.reshape(-1, 1), 
                                 hop_length=self.chunk_size, 
                                 win_length=self.chunk_size*2,
                                 length=self.chunk_size)
        
        # 写入输出流，使用 tanh 进行软剪切并应用音量因子
        audio_final = np.tanh(audio_out) * self.volume_factor
        outdata[:] = audio_final.reshape(-1, 1)

    def cognitive_loop(self):
        """慢速逻辑环 (约 100ms 周期)"""
        print("🧠 认知逻辑环已启动。")
        while self.shared_state['running']:
            # 1. 采样 LSM 脉冲并投影到 HDC 空间
            spikes = self.shared_state['last_spikes']
            if np.any(spikes):
                # 将脉冲转换为 HDC 概念
                concept = self.adapter.forward(spikes)
                self.gwt.update_sense(concept)
                
                # 2. 情节记忆检索与能量计算 (FEP 相关)
                energy = self.memory.compute_energy(concept)
                self.memory.add_memory(concept)
                
                # 3. 目标驱动与惊讶度计算
                surprise = self.gwt.compute_surprise()
                self.drive.step(heard_voice=(np.mean(spikes) > 0.1))
                
                # 4. 生成意图 (Top-down Intent)
                # 预测下一刻的高维概念
                intent_concept = self.wm.predict(concept, 0)
                self.gwt.update_pred(intent_concept)
                
                # 5. 反向投影：将“意图”转化为 LSM 的物理偏置
                bias = self.adapter.backward(intent_concept)
                self.shared_state['cognitive_bias'] = bias
                
                # 6. 多巴胺调节 (基于惊喜度和孤独感)
                # 产生的多巴胺会影响 LSM 的 3-因子学习
                # 6. 多巴胺调节 (基于惊喜度和孤独感)
                # 产生的多巴胺会影响 LSM 的 3-因子学习
                self.shared_state['dopamine'] = 0.1 if surprise < 0.2 else -0.05
            
            # 7. 更新 Dashboard
            if self.use_dashboard:
                 # 耳蜗视图 (当前输入) -> 需要从 callback 获取一份副本
                 # 为简单起见，我们暂时只更新逻辑状态
                 
                 # LSM 光栅图
                 active_neurons = np.where(spikes > 0)[0]
                 self.dashboard.update_lsm_raster(active_neurons)
                 
                 # HDC 相似度 (Surprise 的反面或 Goal Delta)
                 self.dashboard.update_hdc_similarity(1.0 - surprise) # 相似度越高，惊喜度越低
                 
                 # 能量 / 驱动
                 free_energy = self.drive.compute_free_energy(surprise)
                 self.dashboard.update_energy(free_energy)
                 self.dashboard.update_survival(free_energy, self.drive.loneliness)
            
            time.sleep(0.1)

    def run(self):
        print("\n=== AION 完整认知集成交互已启动 ===")
        print("架构：GWT + HDC + MHN + FEP + Generative LSM")
        print("按 Ctrl+C 停止。")
        
        # 启动认知线程
        cog_thread = threading.Thread(target=self.cognitive_loop)
        cog_thread.daemon = True
        cog_thread.start()
        
        try:
            with sd.Stream(samplerate=AUDIO_SR,
                           blocksize=self.chunk_size,
                           channels=1,
                           callback=self.audio_callback):
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n正在停止...")
            self.shared_state['running'] = False
            cog_thread.join(timeout=1.0)
            print("已停止。")

if __name__ == "__main__":
    agent = IntegratedAIONAgent()
    agent.run()
