import torch
from src.config import HDC_DIM, SEED, HRR_DEFAULT_COUNTS

def bind(t1, t2):
    """
    HDC 绑定操作 (XOR)。
    t1, t2: 范围为 {-1, 1} 的张量
    在二进制空间 {0, 1} 中: xor(a, b)
    在双极空间 {-1, 1} 中: mul(a, b)
    """
    return torch.mul(t1, t2)

def bundle(tensor_list):
    """
    HDC 叠加操作（多数规则）。
    tensor_list: 张量列表或堆叠张量
    返回: Sign(Sum(tensors))
    """
    if isinstance(tensor_list, list):
        stacked = torch.stack(tensor_list, dim=0)
    else:
        stacked = tensor_list
        
    s = torch.sum(stacked, dim=0)
    # 符号函数: >0 -> 1, <0 -> -1, 0 -> 1 (偏置向 1)
    res = torch.sign(s)
    res[res == 0] = 1.0
    return res

class HDCWorldModel:
    """
    使用带有置换 (Permutation) 的 HRR 学习转移状态动力学 T(s, a) -> s'。
    
    改进版本：每个动作独立记忆，避免跨动作噪声干扰。
    - 原版: 单一M_sum存储所有(s,a)->s'，噪声累积
    - 新版: M_per_action[action_id]分离存储，预测更清晰
    """
    def __init__(self, device='cpu', n_actions=1):
        self.device = device
        self.dim = HDC_DIM
        self.n_actions = n_actions
        # 每个动作独立记忆
        self.M_per_action = [torch.zeros(self.dim, device=device) for _ in range(n_actions)]
        # 兼容性设置
        self.counts = [0] * n_actions
        
    def _permute(self, t, shifts=1):
        """用于置换的循环移位。"""
        return torch.roll(t, shifts=shifts, dims=-1)
        
    def _inverse_permute(self, t, shifts=1):
        return torch.roll(t, shifts=-shifts, dims=-1)

    def learn(self, state, action, next_state):
        """
        转移状态的一步学习。
        
        参数:
            state: HDC 向量或张量
            action: int (动作 ID) - 必须是整数 
            next_state: HDC 向量或张量
        """
        # 严格要求整数类型
        action_id = action
        
        s = state.to(self.device).float()
        ns = next_state.to(self.device).float()
        
        # 置换下一状态以保持方向性
        p_ns = self._permute(ns)
        
        # trace = s * P(ns)，无需绑定action因为按动作分离
        trace = bind(s, p_ns)
        
        # 累加到对应动作的记忆
        self.M_per_action[action_id] += trace
        self.counts[action_id] += 1
        
        
    def predict(self, state, action):
        """
        预测下一状态。
        
        参数:
            state: HDC 向量
            action: int (动作 ID) - 必须是整数
        返回:
            pred: 预测的下一状态 HDC 向量
        """
        # 严格要求整数类型
        action_id = action
            
        s = state.to(self.device).float()
        M = self.M_per_action[action_id]
        
        if self.counts[action_id] == 0:
            # 该动作从未学习，返回当前状态
            return s
        
        # 解绑: result = M * s
        res = bind(M, s)
        
        # 二值化清理噪声
        res_binary = torch.sign(res)
        res_binary[res_binary == 0] = 1.0
        
        # 逆置换得到预测的下一状态
        pred = self._inverse_permute(res_binary)
        
        return pred


    def load_state_dict(self, state_dict):
        """
        加载状态字典 (HDC 记忆)。
        
        参数:
            state_dict: 张量列表 [M_action_0, M_action_1, ...]
        """
        if isinstance(state_dict, list):
            self.M_per_action = [t.to(self.device).float() for t in state_dict]
            # 假设预测不需要严格的计数，或者我们也应该保存计数。
            # 目前重置计数以避免可能的除以零？
            # 实际上预测会检查 counts[id] == 0。
            # 我们应该假设加载的模型包含数据。
            self.counts = [HRR_DEFAULT_COUNTS] * self.n_actions # 技巧: 标记为已学习
        else:
             print("警告：HDCWorldModel 的 state_dict 格式无效")
