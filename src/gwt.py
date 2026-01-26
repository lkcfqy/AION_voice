import torch
import numpy as np
from src.config import HDC_DIM

class GlobalWorkspace:
    """
    全局工作空间 (GWT)。
    信息交换和冲突监测的核心枢纽。
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.dim = HDC_DIM
        
        # 初始化状态（None 或零）
        # 我们以 None 开始，表示信息缺失
        self.current_sense = None
        self.current_pred = None
        self.current_goal = None
        
    def _to_tensor(self, vec):
        if vec is None:
            return None
        if isinstance(vec, np.ndarray):
            return torch.from_numpy(vec).float().to(self.device)
        return vec.float().to(self.device)

    def update_sense(self, vector):
        self.current_sense = self._to_tensor(vector)
        
    def update_pred(self, vector):
        self.current_pred = self._to_tensor(vector)
        
    def set_goal(self, vector):
        self.current_goal = self._to_tensor(vector)
        
    def _hamming_dist(self, v1, v2):
        """
        归一化汉明距离。
        范围 [0, 1]。
        0 = 相同
        1 = 相反
        0.5 = 正交/随机
        公式: 对于 {-1, 1} 向量，(1 - mean(v1*v2)) / 2
        """
        if v1 is None or v2 is None:
            return 1.0 # 如果信息缺失，则惊喜度最大
            
        # 对于二值向量，平均乘积等同于 Cosine/Dot 除以模长
        mean_product = torch.mean(v1 * v2).item()
        
        # 将平均乘积 [-1, 1] 映射到距离 [1, 0]
        # dist = (1 - sim) / 2
        return (1.0 - mean_product) / 2.0

    def compute_surprise(self):
        """
        计算预测误差（自由能代理指标）。
        Dist(Sense, Pred)
        """
        return self._hamming_dist(self.current_sense, self.current_pred)
        
    def compute_goal_delta(self):
        """
        计算到目标的距离（社交缺失代理指标）。
        Dist(Sense, Goal)
        """
        return self._hamming_dist(self.current_sense, self.current_goal)
        
    def get_status(self):
        """返回当前指标的字典。"""
        return {
            "surprise": self.compute_surprise(),
            "goal_delta": self.compute_goal_delta(),
            "has_sense": self.current_sense is not None,
            "has_pred": self.current_pred is not None,
            "has_goal": self.current_goal is not None
        }
