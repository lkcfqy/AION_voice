import torch
import torch.nn.functional as F
from src.config import MHN_BETA, HDC_DIM

class ModernHopfieldNetwork:
    """
    现代 Hopfield 网络 (MHN)（稠密关联记忆）。
    使用 Softmax 注意力机制进行检索。
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.beta = MHN_BETA
        # 记忆存储：张量列表或单个堆叠张量
        # 初始为空。
        self.memory_matrix = torch.empty(0, HDC_DIM, device=self.device)
        
    def add_memory(self, pattern):
        """
        向记忆中添加新模式。
        使用动态门控：仅当模式足够新颖（相似度 < 阈值）时才添加。
        pattern: (HDC_DIM,) 张量或 numpy 数组
        """
        from src.config import MEMORY_THRESHOLD
        
        if not isinstance(pattern, torch.Tensor):
            pattern = torch.from_numpy(pattern).float().to(self.device)
        else:
            pattern = pattern.float().to(self.device)
            
        # 确保形状为 (1, D)
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)
            
        # 动态门控检查
        if self.memory_matrix.shape[0] > 0:
            # 检查与所有现有记忆的相似度
            # Sim = X * M^T / (Norms) -- 假设是二进制/双极化的归一化向量
            # 对于双极化 {-1, 1}，点积 / D 即为相似度 [-1, 1]
            # 但通常我们使用原始点积来计算能量。
            # 让我们使用余弦相似度进行鲁棒的门控检查。
            
            # 高效实现：点积
            similarities = torch.mm(pattern, self.memory_matrix.T) / pattern.shape[1]
            max_sim = similarities.max().item()
            
            if max_sim > MEMORY_THRESHOLD:
                # 记忆已存在或非常相似。
                # 可选：我们可以在这里更新/强化现有的记忆权重（固化）
                # 但目前我们直接跳过以节省空间。
                return False
            
        # 追加到记忆矩阵
        self.memory_matrix = torch.cat([self.memory_matrix, pattern], dim=0)
        return True
        
    def retrieve(self, query):
        """
        检索最接近的记忆模式。
        公式: X_new = Softmax(beta * X * M^T) * M
        参数:
            query: (HDC_DIM,) 或 (Batch, HDC_DIM)
        返回:
            recalled: 二值化的还原模式
        """
        if self.memory_matrix.shape[0] == 0:
            return query # 无记忆，按原样返回查询内容
            
        if not isinstance(query, torch.Tensor):
            query = torch.from_numpy(query).float().to(self.device)
        else:
            query = query.float().to(self.device)
            
        # 处理批次
        is_batch = query.dim() > 1
        if not is_batch:
            query = query.unsqueeze(0) # (1, D)
            
        # 1. 相似度 (能量)
        # 查询 (Query): (B, D), 记忆 (Memory): (N, D)
        # E = Q * M^T -> (B, N)
        similarity = torch.mm(query, self.memory_matrix.T)
        
        # 2. 注意力 (Softmax)
        # 在记忆上进行加权
        weights = F.softmax(self.beta * similarity, dim=-1) # (B, N)
        
        # 3. 重构
        # Recon = W * M -> (B, N) * (N, D) -> (B, D)
        reconstruction = torch.mm(weights, self.memory_matrix)
        
        # 4. 二值化 (Sign)
        # 自动关联任务通常需要干净的输出
        output = torch.sign(reconstruction)
        output[output == 0] = 1.0
        
        if not is_batch:
            return output.squeeze(0)
            
        return output
    
    @property
    def memory_count(self):
        return self.memory_matrix.shape[0]

    def compute_energy(self, query):
        """
        计算查询状态相对于记忆的标量能量。
        E = -lse(beta * X * M^T)
        """
        if self.memory_count == 0: return 0.0
        
        if not isinstance(query, torch.Tensor):
            query = torch.from_numpy(query).float().to(self.device)
        else:
            query = query.float().to(self.device)
            
        if query.dim() == 1: query = query.unsqueeze(0)
        
        similarity = torch.mm(query, self.memory_matrix.T) # (1, N)
        energy = -torch.logsumexp(self.beta * similarity, dim=-1)
        
        return energy.item()

    def load_memory(self, memory_matrix):
        """
        加载记忆矩阵。
        
        参数:
            memory_matrix: 张量 (N, D)
        """
        if not isinstance(memory_matrix, torch.Tensor):
            # 如果是 numpy，尝试转换
            import numpy as np
            if isinstance(memory_matrix, np.ndarray):
                memory_matrix = torch.from_numpy(memory_matrix)
            else:
                raise TypeError("错误：memory_matrix 必须是张量或 numpy 数组")
                return

        self.memory_matrix = memory_matrix.float().to(self.device)
        print(f"已将 {self.memory_count} 条记忆加载到 MHN")
