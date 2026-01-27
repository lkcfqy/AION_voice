import sys
import os
import glob
import torch
import numpy as np
import librosa
import argparse


# 添加项目根目录
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.lsm import AION_LSM_Network
from src.config import LSM_N_NEURONS, OBS_SHAPE, AUDIO_SR, HOP_LENGTH, WEIGHTS_PATH, DEVICE

def train_gpu(dataset_path, limit=None):
    print(f"🚀 启动 GPU 训练 (Device: {DEVICE})")
    print(f"   数据集: {dataset_path}")
    print(f"   神经元: {LSM_N_NEURONS}")
    
    # 1. 初始化模型
    lsm = AION_LSM_Network(device=DEVICE)
    
    # Ridge Regression 累积矩阵 (GPU Tensor)
    # S^T * S (NxN)
    STS = torch.zeros(LSM_N_NEURONS, LSM_N_NEURONS, device=DEVICE)
    # S^T * Y (NxOut)
    STY = torch.zeros(LSM_N_NEURONS, OBS_SHAPE, device=DEVICE)
    
    frames_count = 0
    
    # 2. 加载文件
    wav_files = glob.glob(os.path.join(dataset_path, "**/*.wav"), recursive=True)
    if not wav_files:
        print("❌ 未找到 WAV 文件")
        return
        
    if limit:
        wav_files = wav_files[:limit]
        
    print(f"📂 待处理文件数: {len(wav_files)}")
    
    # 3. 处理循环
    try:
        total = len(wav_files)
        for i, wav_path in enumerate(wav_files):
            if i % 10 == 0: print(f"正在处理 {i}/{total}...")
            # 加载音频
            y, sr = librosa.load(wav_path, sr=AUDIO_SR)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=OBS_SHAPE, hop_length=HOP_LENGTH, n_fft=HOP_LENGTH*2)
            mel_db = librosa.power_to_db(mel, ref=1.0)
            
            # 序列处理：保留时序结构 (T, F)
            mel_seq = (mel_db.T + 80.0)/80.0
            mel_seq = np.clip(mel_seq, 0, 1) # (T, 64)
            
            mel_tensor = torch.tensor(mel_seq, dtype=torch.float32, device=DEVICE)
            
            lsm.reset()
            
            # 收集状态 (Inputs) 和 目标 (Targets)
            # 输入: Mel[t]
            # 目标: Mel[t+1] (预测下一帧)
            inputs = mel_tensor[:-1]
            targets = mel_tensor[1:]
            
            # 逐帧运行仿真
            spikes_list = []
            
            # 自定义前向循环以维持状态
            for t in range(len(inputs)):
                # 输入需要是 (1, In) 维度
                s = lsm.forward(inputs[t].unsqueeze(0)) # (1, N)
                spikes_list.append(s)
                
            S = torch.cat(spikes_list, dim=0) # (T-1, N)
            Y = targets # (T-1, Out)
            
            # 累积相关性矩阵 (Ridge Regression)
            # STS += S.T @ S
            STS += torch.mm(S.T, S)
            # STY += S.T @ Y
            STY += torch.mm(S.T, Y)
            
            frames_count += len(S)
            pass
            
    except KeyboardInterrupt:
        print("\n🛑 停止训练...")
        
    # 4. 求解权重
    if frames_count > 0:
        print("🏗️ 正在求解权重...")
        # 岭回归 (Ridge Regularization)
        lambda_reg = 10.0
        I = torch.eye(LSM_N_NEURONS, device=DEVICE)
        
        A = STS + lambda_reg * I
        B = STY
        
        # Torch 求解: A * W = B

        # torch.linalg.solve assumes A is square batch.
        try:
            W_out = torch.linalg.solve(A, B)
            
            # Save
            torch.save(W_out, WEIGHTS_PATH)
            print(f"✅ 权重已保存至: {WEIGHTS_PATH}")
            print(f"   Max Weight: {torch.max(torch.abs(W_out)):.4f}")
            
        except RuntimeError as e:
            print(f"❌ 求解失败: {e}")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    train_gpu(args.data, args.limit)
