"""
AION 项目全局配置
"""

# 感官配置
OBS_SHAPE = 64  # 64 维 Mel 频谱向量

# 路径设置
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 资产路径（如有音频模板等）
# 资产路径（如有音频模板等）
ASSET_PATH = PROJECT_ROOT 

# 模型与权重路径
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
WEIGHTS_FILENAME = "lsm_readout_weights.pt"
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, WEIGHTS_FILENAME) # 为了兼容性暂时放在根目录，或者也可以移入 checkpoints
TRAIN_CHECKPOINT_FILENAME = "training_checkpoint.pt"
TRAIN_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, TRAIN_CHECKPOINT_FILENAME)


# Visdom 设置
VISDOM_SERVER = "http://localhost"
VISDOM_PORT = 8097
VISDOM_ENV = "AION_Dashboard"

# 仿真设置
SEED = 42 # 全局随机种子

# LSM (液体状态机) 设置
LSM_N_NEURONS = 2000
LSM_SPARSITY = 0.1     # 10% 递归连接
LSM_IN_SPARSITY = 0.1  # 10% 输入连接（用于处理 12k 输入）
LSM_STEPS_PER_SAMPLE = 50 # 模拟一个音频切片的迭代次数 (ms)
TARGET_FIRING_RATE = 20 # Hz (稳态目标频率)
PLASTICITY_RATE = 0.005 # 学习率（更稳定）
RATE_TAU = 0.05        # 频率估计时间常数 (50ms)
TAU_RC = 0.02          # 膜时间常数 (20ms)
TAU_REF = 0.002        # 迟滞期 (2ms)
DT = 0.001             # 仿真步长 (1ms)

# HDC 设置
HDC_DIM = 10000        # 超维向量维度
MHN_BETA = 20.0        # 现代 Hopfield 网络逆温度
MEMORY_THRESHOLD = 0.9 # 存储新记忆的相似度阈值（动态门控）

# 驱动设置 (Social Drive)
SOCIAL_WEIGHT = 1.0    # 惊喜度与社交渴望之间的平衡
DECAY_RATE = 0.001     # 每步社交满足感的衰减量

# 音频处理设置
AUDIO_SR = 16000     # 采样率
HOP_LENGTH = 512     # 帧移

# 学习与适应配置
LSM_LEARNING_RATE = 1e-4
ADAPTER_SCALING = 0.1 # 投影缩放因子

# 社交驱动配置
SOCIAL_RESTORE_VOICE = 0.05 # 听到声音的恢复量
SOCIAL_RESTORE_SPOKE = 0.01 # 说话的恢复量

# HRR 配置
HRR_DEFAULT_COUNTS = 100 # 现在的默认计数，直到支持序列化

