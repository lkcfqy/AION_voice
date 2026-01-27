import torch
import numpy as np
from src.config import HDC_DIM, LSM_N_NEURONS, SEED, ADAPTER_SCALING

class RandomProjectionAdapter:
    """
    使用随机投影将 LSM 模拟活动投影到 HDC 二进制空间。
    作为局部敏感哈希 (LSH) 使用。
    """
    def __init__(self, device='cpu'):
        self.input_dim = LSM_N_NEURONS
        self.output_dim = HDC_DIM
        self.device = device
        
        # 初始化随机投影矩阵
        # 形状：(输出, 输入) -> (10000, 400)
        # 我们使用固定种子以确保“概念空间”的可重复性
        torch.manual_seed(SEED)
        
        # 投影权重的标准正态分布
        # 为什么使用高斯分布？它保证了余弦相似性的 LSH 属性 (SimHash)
        self.projection_matrix = torch.randn(self.output_dim, self.input_dim, device=self.device)
        
        # 冻结权重（这里不进行学习，仅进行投影）
        self.projection_matrix.requires_grad_(False)
        
    def forward(self, lsm_activity):
        """
        参数:
            lsm_activity: (N_neurons,) numpy 数组或张量
        返回:
            hdc_vector: (HDC_DIM,) 范围为 {-1, 1} 的张量
        """
        # 如果需要，将输入转换为张量
        if isinstance(lsm_activity, np.ndarray):
            x = torch.from_numpy(lsm_activity.copy()).float().to(self.device)
        else:
            x = lsm_activity.float().to(self.device)
            
        # 线性投影
        # y = Wx
        y = torch.mv(self.projection_matrix, x)
        
        # 二值化 (Sign)
        # Sign(y) -> -1 或 1。(0 会变成 0，但在实际中通常是非零浮点数)
        # 我们强制 0 -> 1 以保持严格二值化
        h = torch.sign(y)
        h[h == 0] = 1.0 
        
        return h

    def backward(self, hdc_vector):
        """
        将 HDC 向量从高维空间投影回 LSM 神经元电平空间。
        用于 Top-down 影响。
        参数:
            hdc_vector: (HDC_DIM,) 张量或 numpy 数组
        返回:
            lsm_bias: (N_neurons,) numpy 数组
        """
        if isinstance(hdc_vector, np.ndarray):
            h = torch.from_numpy(hdc_vector).float().to(self.device)
        else:
            h = hdc_vector.float().to(self.device)
            
        # 反向投影: x_hat = W^T * h
        # 形状: (N_neurons, HDC_DIM) * (HDC_DIM, 1) -> (N_neurons, 1)
        x_hat = torch.mv(self.projection_matrix.T, h)
        
        # 归一化/缩放以适应 LSM 的输入电流范围
        x_hat = x_hat / self.output_dim * ADAPTER_SCALING # 经验缩放
        
        return x_hat.cpu().numpy()

    def batch_forward(self, lsm_batch):
        """
        参数:
            lsm_batch: (Batch, N_neurons)
        返回:
            hdc_batch: (Batch, HDC_DIM)
        """
        if isinstance(lsm_batch, np.ndarray):
            x = torch.from_numpy(lsm_batch).float().to(self.device)
        else:
            x = lsm_batch.float().to(self.device)
            
        y = torch.mm(x, self.projection_matrix.T)
        h = torch.sign(y)
        h[h == 0] = 1.0
        return h
