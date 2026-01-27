import sys
import os
import numpy as np
import librosa
import torch
import glob
import pickle

# 将项目根目录添加到路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.lsm import AION_LSM_Network
from src.lsm import AION_LSM_Network
from src.config import LSM_N_NEURONS, OBS_SHAPE, AUDIO_SR, HOP_LENGTH, SEED, WEIGHTS_PATH, TRAIN_CHECKPOINT_PATH


def preprocess_audio(audio_path):
    """加载音频并转换为 Mel 频谱序列"""
    y, sr = librosa.load(audio_path, sr=AUDIO_SR)
    # 提取 Mel 频谱 (n_mels = OBS_SHAPE)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=OBS_SHAPE, hop_length=HOP_LENGTH, n_fft=HOP_LENGTH*2)
    # 转为对数分贝，使用固定参考值 1.0 以保留绝对音量信息
    mel_db = librosa.power_to_db(mel, ref=1.0)
    # 使用全局固定范围归一化 [-80, 0] dB -> [0, 1]
    mel_norm = (mel_db + 80.0) / 80.0
    mel_norm = np.clip(mel_norm, 0, 1)
    return mel_norm.T # (n_frames, n_mels)

def train(dataset_path, checkpoint_path=TRAIN_CHECKPOINT_PATH, limit=None):

    print(f"正在准备数据集: {dataset_path}")
    wav_files = glob.glob(os.path.join(dataset_path, "**/*.wav"), recursive=True)
    if not wav_files:
        print("❌ 未找到音频文件！请提供有效的数据集路径。")
        return

    # 初始化 LSM 和 累积矩阵
    lsm = AION_LSM_Network()
    n_neurons = LSM_N_NEURONS
    n_out = OBS_SHAPE
    
    sts_total = np.zeros((n_neurons, n_neurons))  # S^T * S
    sty_total = np.zeros((n_neurons, n_out))      # S^T * Y
    processed_files = set()
    total_frames = 0

    # 尝试加载断点 (Checkpoint)
    if os.path.exists(checkpoint_path):
        print(f"🔄 发现断点文件 {checkpoint_path}，正在恢复进度...")
        checkpoint = torch.load(checkpoint_path)
        sts_total = checkpoint['sts_total']
        sty_total = checkpoint['sty_total']
        processed_files = checkpoint['processed_files']
        total_frames = checkpoint.get('total_frames', 0)
        print(f"已恢复: 已处理 {len(processed_files)} 个文件，共 {total_frames} 帧。")

    files_to_process = [f for f in wav_files if f not in processed_files]
    if limit:
        files_to_process = files_to_process[:limit]

    if not files_to_process:
        print("✨ 所有文件已处理完毕，或没有新文件需要处理。")
    else:
        print(f"正在处理 {len(files_to_process)} 个新文件...")
        
        try:
            for i, wav_file in enumerate(files_to_process):
                try:
                    mel_seq = preprocess_audio(wav_file)
                    
                    # 回声状态采集 (Harvesting States)
                    lsm.reset()
                    
                    # 采集该文件的 spikes 和 targets
                    file_spikes = []
                    file_targets = []
                    for t in range(len(mel_seq) - 1):
                        spikes, _ = lsm.step(mel_seq[t])
                        file_spikes.append(spikes)
                        file_targets.append(mel_seq[t+1])
                    
                    S = np.array(file_spikes)
                    Y = np.array(file_targets)
                    
                    # 增量累积矩阵
                    sts_total += S.T @ S
                    sty_total += S.T @ Y
                    total_frames += len(S)
                    processed_files.add(wav_file)
                    
                    # 每 50 个文件保存一次临时断点
                    if (i + 1) % 50 == 0:
                        torch.save({
                            'sts_total': sts_total,
                            'sty_total': sty_total,
                            'processed_files': processed_files,
                            'total_frames': total_frames
                        }, checkpoint_path)
                        print(f"💾 已保存进度: 处理到第 {len(processed_files)}/{len(wav_files)} 个文件...")
                        
                except Exception as e:
                    print(f"处理文件 {wav_file} 时出错: {e}")
                    
        except KeyboardInterrupt:
            print("\n🛑 收到停止指令，正在保存当前进度...")
            torch.save({
                'sts_total': sts_total,
                'sty_total': sty_total,
                'processed_files': processed_files,
                'total_frames': total_frames
            }, checkpoint_path)
            print("进度已保存。您可以随时重新运行脚本继续。")
            return

    # 最终计算权重
    if total_frames > 0:
        print(f"🏗️ 正在求解最终权重 (总帧数: {total_frames})...")
        lambda_reg = 1.0
        I = np.eye(n_neurons)
        A = sts_total + lambda_reg * I
        B = sty_total
        
        # 求解 A * W_out = B
        W_out = np.linalg.solve(A, B)
        
        # 保存权重
        weights_path = WEIGHTS_PATH
        torch.save(W_out, weights_path)

        
        # 同时保存最终断点以便未来继续扩充数据集
        torch.save({
            'sts_total': sts_total,
            'sty_total': sty_total,
            'processed_files': processed_files,
            'total_frames': total_frames
        }, checkpoint_path)
        
        print(f"✅ 训练完成！")
        print(f"   - 最终权重: {weights_path}")
        print(f"   - 断点状态: {checkpoint_path}")
    else:
        print("⚠️ 没有可用的训练数据。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AION 生成式 LSM 增量训练脚本 (支持断点续练)")
    parser.add_argument("--data", type=str, help="WAV 数据集目录路径")
    parser.add_argument("--limit", type=int, default=None, help="本次运行处理的最大文件数")
    parser.add_argument("--limit", type=int, default=None, help="本次运行处理的最大文件数")
    parser.add_argument("--checkpoint", type=str, default=TRAIN_CHECKPOINT_PATH, help="断点文件保存路径")
    args = parser.parse_args()

    
    if args.data:
        train(args.data, checkpoint_path=args.checkpoint, limit=args.limit)
    else:
        print("💡 使用说明:")
        print("  python scripts/train_generative.py --data LJSpeech-1.1/wavs")
        print("  按 Ctrl+C 可随时停止并保存进度。")
