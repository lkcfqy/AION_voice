import torch
import numpy as np
from src.config import HDC_DIM, LSM_N_NEURONS, SEED, ADAPTER_SCALING, DEVICE

class RandomProjectionAdapter:
    """
    使用随机投影将 LSM 模拟活动投影到 HDC 二进制空间。
    (PyTorch GPU 版本)
    """
    def __init__(self, device=DEVICE):
        self.input_dim = LSM_N_NEURONS
        self.output_dim = HDC_DIM
        self.device = device
        
        torch.manual_seed(SEED)
        
        # 投影权重的标准正态分布
        self.projection_matrix = torch.randn(self.output_dim, self.input_dim, device=self.device)
        self.projection_matrix.requires_grad_(False)
        
    def forward(self, lsm_activity):
        """前向传播: 将 LSM 神经活动转换为 HDC 超维向量 (N -> D)"""
        if isinstance(lsm_activity, np.ndarray):
            x = torch.from_numpy(lsm_activity).float().to(self.device)
        else:
            x = lsm_activity.float().to(self.device)
            
        y = torch.mv(self.projection_matrix, x)
        h = torch.sign(y)
        h[h == 0] = 1.0 
        return h

    def backward(self, hdc_vector):
        """反向传播: 从 HDC 超维向量重建 LSM 神经状态 (D -> N)"""
        if isinstance(hdc_vector, np.ndarray):
            h = torch.from_numpy(hdc_vector).float().to(self.device)
        else:
            h = hdc_vector.float().to(self.device)
            
        # 反向投影: x_hat = W^T * h
        x_hat = torch.mv(self.projection_matrix.T, h)
        x_hat = x_hat / self.output_dim * ADAPTER_SCALING
        
        return x_hat.cpu().numpy()
