import numpy as np
import librosa
import torch
from src.config import OBS_SHAPE

class Cochlea:
    """
    电子耳蜗（音频输入系统）。
    将原始音频波形转换为二维频谱图（LSM 输入）。
    
    设计：
    - 输入：原始音频（一维数组）
    - 处理：Mel 频谱图或 STFT
    - 输出：归一化的二维矩阵（频率 x 时间） -> 调整大小以适应 OBS_SHAPE（例如 64x64）
    """
    def __init__(self, sample_rate=16000, n_mels=64, hop_length=256):
        self.sr = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.target_shape = (OBS_SHAPE[0], OBS_SHAPE[1]) # e.g. (64, 64)
        
    def process(self, audio_segment):
        """
        将音频分段处理为频谱图。
        参数:
            audio_segment: numpy 数组 (1D)
        返回:
            spectrogram: 归一化到 [0,1] 的 numpy 数组 (64, 64, 1 或 3)
        """
        # 1. 预加重（可选，生物拟真）
        y = librosa.effects.preemphasis(audio_segment)
        
        # 2. Mel 频谱图
        # n_fft 决定频率分辨率。
        # 对于 64 个 mel 滤波器组，n_fft 应足够大（如 512 或 1024）
        mels = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels, 
            n_fft=1024, hop_length=self.hop_length
        )
        
        # 3. 对数刻度 (dB) - 模仿人类听觉
        # 关键修正：使用固定参考值 (1.0) 而不是每帧的最大值 (np.max)。
        # 每帧归一化 (np.max) 会将静音/噪声放大到 0dB (Max=1.0)，导致持续触发和噪声。
        # ref=1.0 假设输入音频在 [-1, 1] 之间，因此最大功率为 1.0 (0dB)。
        mels_db = librosa.power_to_db(mels, ref=1.0)
        
        # 4. 归一化到 [0, 1]
        # dB 范围通常是 [-80, 0] 或类似区间。
        # 我们在 -80dB 处截断底部
        mels_db = np.clip(mels_db, -80, 0)
        # 归一化
        img = (mels_db + 80) / 80.0
        
        # 5. 调整大小/剪裁到目标形状 (64x64)
        # 当前形状：(n_mels, time_steps)
        # 我们需要 (64, 64)
        
        # 如果时间步数 < 64，则填充
        # 如果时间步数 > 64，则剪裁（或调整大小）
        # 策略：调整大小（拉伸/压缩）还是滚动窗口？
        # 对于智能体来说，“滚动窗口”或“快照”更好。
        # 在这里，我们假设输入 audio_segment 大致对应于我们想要的窗口大小。
        # 为了提高鲁棒性，让我们通过图像插值调整大小。
        
        # img 格式为 (频率, 时间)。我们需要 (H, W)。
        # 我们可以将其视为一幅图像。
        
        # 使用简单的插值算法
        from scipy.ndimage import zoom
        
        curr_h, curr_w = img.shape
        target_h, target_w = self.target_shape
        
        # 计算缩放因子
        zoom_h = target_h / curr_h
        zoom_w = target_w / curr_w
        
        resized = zoom(img, (zoom_h, zoom_w))
        
        # 6. 如果需要，添加通道维度（当前文件中 OBS_SHAPE 有 3 个通道）
        # OBS_SHAPE 在当前配置中为 (64, 64, 3)。
        # 我们需要与其匹配。
        
        # 复制为 3 通道 (RGB)，因为 LSM 期望这种格式（或者以后更新 LSM）
        # 或者只使用 1 个通道，其他通道留白。
        # 复制通道对于视觉预训练的兼容性更安全。
        
        output = np.stack([resized, resized, resized], axis=-1)
        
        return output

