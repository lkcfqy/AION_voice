import torch
import torch.nn.functional as F
import numpy as np
from src.config import HDC_DIM, DEVICE

class AttentionGWT:
    """
    基于注意力机制的全局工作空间 (GWT)。
    不再是简单的广播，而是根据 'Drive' (情感/需求) 主动聚焦于特定的输入模块。
    """
    def __init__(self, device=DEVICE):
        self.device = device
        self.dim = HDC_DIM
        
        # 状态记忆
        self.workspace_content = None # 当前广播的内容
        self.attention_weights = {}   # 当前的注意力分布
        
    def _to_tensor(self, vec):
        if vec is None: return None
        if isinstance(vec, np.ndarray):
            return torch.from_numpy(vec).float().to(self.device)
        return vec.float().to(self.device)

    def broadcast(self, query, input_modules):
        """
        核心注意力机制: Global Broadcast via Attention.
        
        参数:
            query (Q): 当前的驱动状态/目标向量 (1, D) - "我想找什么？"
            input_modules (K, V): 各模块的输入 {name: vector} - "大家都在说什么？"
            
        返回:
            broadcast_vector: 加权后的全局信息
        """
        q = self._to_tensor(query)
        
        if q is None:
            # 如果没有驱动，默认均匀注意 (或者随机)
            # 构造一个全 1 的 Query? 不，没有 Query 就无法聚焦。
            # 这种情况下，也许平均所有输入？
            # 为了简单，我们随机初始化一个 Query
            q = torch.randn(1, self.dim, device=self.device)
        
        if q.dim() == 1: q = q.unsqueeze(0)
            
        # 收集 Keys (K) 和 Values (V)
        # 假设 Key == Value (自注意力变体)
        names = []
        vectors = []
        for name, vec in input_modules.items():
            if vec is not None:
                v = self._to_tensor(vec)
                if v.dim() == 1: v = v.unsqueeze(0)
                names.append(name)
                vectors.append(v)
                
        if not vectors:
            return None
            
        # 堆叠 V: (Batch, N_Modules, D) -> (1, N, D)
        K = torch.cat(vectors, dim=0) # (N, D)
        V = K 
        
        # 注意力分数: Q * K^T
        # (1, D) * (D, N) -> (1, N)
        scores = torch.mm(q, K.T)
        
        # 缩放 (Scale)
        scores = scores / np.sqrt(self.dim)
        
        # Softmax 归一化
        attn_weights = F.softmax(scores, dim=-1) # (1, N)
        
        # 加权求和: Attn * V
        # (1, N) * (N, D) -> (1, D)
        output = torch.mm(attn_weights, V)
        
        # 二值化输出 (GWT 通常广播清晰的信号)
        # output = torch.sign(output) # 可选：是否硬广播？
        # Attention 允许混合态，这可能更好 (软广播 / Soft Broadcast)
        
        # 保存状态供仪表盘 (Dashboard) 使用
        self.workspace_content = output
        self.attention_weights = {name: w.item() for name, w in zip(names, attn_weights[0])}
        
        return output

    def compute_surprise(self, sense, pred):
        """兼容接口：计算自由能"""
        if sense is None or pred is None: return 1.0
        sense = self._to_tensor(sense)
        pred = self._to_tensor(pred)
        # Cosine Distance
        sim = F.cosine_similarity(sense, pred, dim=-1).item()
        return (1.0 - sim) / 2.0
