import sys
import os
import time
import torch
import numpy as np
import librosa
import sounddevice as sd
import queue
import threading
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.lsm import AION_LSM_Network
from src.adapter import RandomProjectionAdapter
from src.gwt import AttentionGWT
from src.hrr import ResonatorNetwork
from src.mhn import ModernHopfieldNetwork
from src.drive import SocialDrive
from src.dashboard import AIONDashboard
from src.config import LSM_N_NEURONS, OBS_SHAPE, AUDIO_SR, HOP_LENGTH, DEVICE, WEIGHTS_PATH, ADAPTER_SCALING

class AION_Agent_GPU:
    def __init__(self):
        print(f"[!] 初始化 AION GPU 认知架构 (Device: {DEVICE})...")
        
        # 1. 核心模块 (Core Modules)
        self.device = DEVICE
        self.lsm = AION_LSM_Network(device=DEVICE)
        self.adapter = RandomProjectionAdapter(device=DEVICE)
        self.gwt = AttentionGWT(device=DEVICE)
        self.mhn = ModernHopfieldNetwork(device=DEVICE)
        self.resonator = ResonatorNetwork(device=DEVICE)
        self.drive = SocialDrive()
        
        # 加载权重
        if os.path.exists(WEIGHTS_PATH):
            print(f"[INFO] 加载权重: {WEIGHTS_PATH}")
            loaded_tensor = torch.load(WEIGHTS_PATH, map_location=DEVICE)
            self.lsm.W_out.data = loaded_tensor.to(DEVICE)
        else:
            print("[WARNING] 未找到预训练权重，使用随机初始化 (将无法正常说话)")

        # 2. 状态管理
        self.running = True
        self.is_sleeping = False
        self.silence_timer = 0
        self.last_activity_time = time.time()
        
        # 音频缓冲
        self.chunk_size = HOP_LENGTH
        self.n_fft = self.chunk_size * 2
        self.in_buffer = np.zeros(self.n_fft)
        
        # 音频处理矩阵 (CPU -> GPU 在循环中处理)
        self.mel_basis = librosa.filters.mel(sr=AUDIO_SR, n_fft=self.n_fft, n_mels=OBS_SHAPE)
        self.mel_basis_inv = np.linalg.pinv(self.mel_basis)
        
        # 睡眠设置
        self.SLEEP_THRESHOLD = 5.0 # 秒 (无声多长时间后入睡) - 已修改为5秒以便测试
        
        # 仪表盘
        try:
            self.dashboard = AIONDashboard()
            self.use_dashboard = True
        except:
            print("[WARNING] Dashboard 未连接")
            self.use_dashboard = False

        # 从数据集预加载记忆
        self.preload_memories()
            
    def preload_memories(self):
        """从数据集中加载先天记忆"""
        import glob
        import random
        
        print("📥 正在植入先天记忆 (从训练集)...")
        # 使用绝对路径确保能找到文件
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_pattern = os.path.join(project_root, "datasets", "**", "*.wav")
        print(f"   Searching in: {dataset_pattern}")
        
        wav_files = glob.glob(dataset_pattern, recursive=True)
        print(f"   Found {len(wav_files)} files.")
        
        if not wav_files:
            print("[WARNING] 未找到数据集文件，大脑将以空白状态启动。")
            return
            
        # 随机选择 5 个文件
        count = 5
        selected_files = random.sample(wav_files, min(len(wav_files), count))
        
        for wav_path in selected_files:
            try:
                # 快速处理流程 (不播放，只记忆)
                y, sr = librosa.load(wav_path, sr=AUDIO_SR)
                # 截取一小段 (1秒)
                if len(y) > AUDIO_SR:
                    y = y[:AUDIO_SR]
                
                # 填充以防过短
                if len(y) < self.n_fft:
                    y = np.pad(y, (0, self.n_fft - len(y)))
                    
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=OBS_SHAPE, hop_length=self.chunk_size, n_fft=self.n_fft)
                mel_db = librosa.power_to_db(mel, ref=1.0)
                if mel_db.shape[1] > 0:
                    mel_vec = (mel_db[:, -1] + 80.0) / 80.0 # 取最后一帧作为特征
                    mel_vec = np.clip(mel_vec, 0, 1)
                    
                    # 转为 Tensor
                    mel_tensor = torch.tensor(mel_vec, dtype=torch.float32, device=self.device)
                    
                    # 激活 LSM 获取脉冲模式
                    self.lsm.reset()
                    # 预热几步
                    for _ in range(5):
                        spikes = self.lsm.forward(mel_tensor)
                        
                    # 形成概念并存储
                    concept = self.adapter.forward(spikes.flatten())
                    added = self.mhn.add_memory(concept)
                    if added:
                        print(f"   Mapped: {os.path.basename(wav_path)}")
                
            except Exception as e:
                print(f"   Skipped {wav_path}: {e}")
                
        print(f"[BRAIN] 成功植入 {self.mhn.memory_count} 条先天记忆！")

    def audio_callback(self, indata, outdata, frames, time_info, status):
        """实时音频回调 (运行在独立线程)"""
        if self.is_sleeping:
            # 睡眠模式：不处理外界输入，只播放内部生成的“梦话”
            # 这里我们通过 check_dream_queue 或类似机制获取输出
            # 简单起见，睡眠时的输出由 cognitive_loop 直接写入 sounddevice 的 OutputStream?
            # 或者在这里填零，由主线程控制播放。
            outdata.fill(0)
            return

        # 1. 输入处理 (Input Processing)
        new_data = indata.flatten()
        self.in_buffer = np.roll(self.in_buffer, -self.chunk_size)
        self.in_buffer[-self.chunk_size:] = new_data
        
        # 麦克风增益 (Pre-amp Gain)
        gain = 100.0
        buffer_boosted = self.in_buffer * gain
        input_rms = np.sqrt(np.mean(buffer_boosted**2))
        
        # 活动检测 (降低阈值)
        if input_rms > 0.02:
            self.last_activity_time = time.time()
        
        # 计算 Mel 频谱 (CPU)
        mel = librosa.feature.melspectrogram(y=buffer_boosted, sr=AUDIO_SR, n_mels=OBS_SHAPE, hop_length=self.chunk_size, n_fft=self.n_fft)
        mel_db = librosa.power_to_db(mel, ref=1.0)
        mel_vec = (mel_db[:, -1] + 80.0) / 80.0
        mel_vec = np.clip(mel_vec, 0, 1)
        
        # 噪声门 (Noise Gate) (降低阈值)
        if np.max(mel_vec) < 0.05:
            mel_vec.fill(0)
            
        # 2. 传输到 GPU
        mel_tensor = torch.tensor(mel_vec, dtype=torch.float32, device=self.device)
        
        # 3. LSM 模拟步 (GPU)
        # 注入 自上而下 (Top-down) 偏置 (来自 GWT 广播)
        bias = self.gwt.workspace_content # (1, D)
        # Adapter 反向传播: HDC -> Neurons
        bias_current = None
        if bias is not None:
             bias_current = self.adapter.backward(bias).flatten() # numpy
             bias_current = torch.tensor(bias_current, device=self.device)
             
        # 修复逻辑错误: 输入信号 (Mel) 和 偏置电流 (Neuron Space) 维度不同，不能直接相加。
        # 我们使用 lsm.forward 的 external_current 参数注入偏置。
        scaled_bias = 0.01 * bias_current if bias_current is not None else None
        spikes = self.lsm.forward(mel_tensor, external_current=scaled_bias)
        spikes = spikes.flatten() # (1, N) -> (N,)
        
        # 读出预测 (Readout)
        prediction = spikes @ self.lsm.W_out # (N,) @ (N, Out) -> (Out,)
        
        # 4. 神经音频合成 (GPU -> CPU)
        pred_np = prediction.detach().cpu().numpy().flatten()
        pred_np = np.clip(pred_np, 0, 1)
        
        # 信号重建 (Mel -> Linear -> Waveform)
        # Mel -> Linear
        mel_db_out = pred_np * 80.0 - 80.0
        mel_p = librosa.db_to_power(mel_db_out)
        stft_p = self.mel_basis_inv @ mel_p
        stft_mag = np.sqrt(np.maximum(stft_p, 0))
        
        # 逆傅里叶变换 (IFFT)
        wav_chunk = np.fft.irfft(stft_mag, n=self.n_fft)
        windowed = wav_chunk * np.hanning(self.n_fft)
        # 简化版 OLA: 直接输出切片中心部分以降低延迟
        out_chunk = windowed[:self.chunk_size] 
        
        outdata[:] = np.tanh(out_chunk).reshape(-1, 1) * 1.0

        # 更新全局状态供认知循环使用
        self.current_spikes = spikes

    def cognitive_cycle(self):
        """慢速认知循环 (10Hz)"""
        while self.running:
            # 睡眠检查
            if time.time() - self.last_activity_time > self.SLEEP_THRESHOLD:
                if not self.is_sleeping:
                    print("\n[SLEEP] 环境安静，进入在线睡眠巩固模式 (做梦)...")
                    self.is_sleeping = True
                    self.enter_dream_state()
            else:
                 if self.is_sleeping:
                     print("\n[WAKE] 检测到活动，唤醒中...")
                     self.is_sleeping = False
            
            if not self.is_sleeping:
                # 正常认知处理
                if hasattr(self, 'current_spikes'):
                    spikes = self.current_spikes # (N,) Tensor
                    
                    # 1. 感知: LSM -> HDC
                    concept = self.adapter.forward(spikes) # (D,)
                    
                    # 2. 注意力广播
                    # 查询 = 驱动 (孤独/需求) - 暂未实现 Drive 向量化，先用 Concept
                    # 广播: 这里的输入源可以是 视觉, 音频, 记忆
                    # 目前只有 音频 (Concept)
                    broadcast = self.gwt.broadcast(query=concept, input_modules={'audio': concept})
                    
                    # 3. 记忆
                    self.mhn.add_memory(broadcast)
                    
                    # 4. 仪表盘
                    if self.use_dashboard:
                        self.dashboard.update_lsm_raster(torch.where(spikes > 0)[0].cpu().numpy())
                        
            time.sleep(0.1)

    def enter_dream_state(self):
        """做梦模式：随机回放记忆并生成声音"""
        while self.is_sleeping and self.running:
            # 检测是否被唤醒
            if time.time() - self.last_activity_time < 0.5:
                break
                
            if self.mhn.memory_count > 0:
                # 1. 回忆 (随机采样)
                idx = np.random.randint(0, self.mhn.memory_count)
                memory = self.mhn.memory_matrix[idx] # (D,)
                
                # 2. 想象 (Top-down)
                # HDC -> LSM Neurons
                bias = self.adapter.backward(memory) # (N,) numpy
                bias_tensor = torch.tensor(bias, device=self.device)
                
                # 3. 激活 LSM (无输入，只有 bias)
                # 模拟一段 "梦境" (例如 100ms)
                print(f"\r[DREAM] 正在回放记忆片段 #{idx}...", end="")
                
                generated_audio = []
                # 重置 LSM 内部状态以获得清晰的梦境
                self.lsm.reset()
                
                for _ in range(10): # 10 帧
                    spikes = self.lsm.forward(bias_tensor * 2.0) # 强刺激
                    pred = spikes @ self.lsm.W_out
                    
                    # 合成音频
                    p = pred.detach().cpu().numpy()
                    p = np.clip(p, 0, 1)
                    # ... (简单的合成，类似于回调函数)
                    mel_p = librosa.db_to_power(p * 80 - 80)
                    wav = np.fft.irfft(np.sqrt(np.maximum(self.mel_basis_inv @ mel_p, 0)), n=self.n_fft)
                    generated_audio.append(wav[:self.chunk_size])
                    
                # 播放梦境声音
                full_dream = np.concatenate(generated_audio)
                sd.play(np.tanh(full_dream) * 0.5, AUDIO_SR)
                sd.wait()
                
            else:
                print("\r[BRAIN] 记忆库为空！请先对着麦克风说话，让我积累一些素材...", end="")
                time.sleep(1.0)
                
            time.sleep(1.0) # 梦境间隔

    def run(self):
        cog_thread = threading.Thread(target=self.cognitive_cycle)
        cog_thread.daemon = True
        cog_thread.start()
        
        print("\n" + "="*50)
        print("[MIC] AION 语音交互系统已启动")
        print("[TIP] 使用指南:")
        print("1. 对着麦克风说话 -> 它会学习并尝试跟随你的声音。")
        print("2. 保持安静 5 秒 -> 它会进入梦境，回放刚才学到的声音片段。")
        print("[WARNING] 注意：启动时记忆是空的，你必须先说话！")
        print("="*50 + "\n")
        
        print("[MIC] 麦克风监听中...")
        with sd.Stream(samplerate=AUDIO_SR, blocksize=self.chunk_size, channels=1, callback=self.audio_callback):
            while self.running:
                try:
                    time.sleep(1.0)
                except KeyboardInterrupt:
                    self.running = False
                    print("\n[EXIT] 正在停止...")

if __name__ == "__main__":
    agent = AION_Agent_GPU()
    agent.run()
