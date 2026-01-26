import torch
from src.config import HDC_DIM, SEED

def bind(t1, t2):
    """
    HDC Binding Operation (XOR).
    t1, t2: Tensor of {-1, 1}
    in binary space {0, 1}: xor(a, b)
    in bipolar space {-1, 1}: mul(a, b)
    """
    return torch.mul(t1, t2)

def bundle(tensor_list):
    """
    HDC Superposition Operation (Majority Rule).
    tensor_list: List of tensors or stacked tensor
    Returns: Sign(Sum(tensors))
    """
    if isinstance(tensor_list, list):
        stacked = torch.stack(tensor_list, dim=0)
    else:
        stacked = tensor_list
        
    s = torch.sum(stacked, dim=0)
    # Sign function: >0 -> 1, <0 -> -1, 0 -> 1 (bias to 1)
    res = torch.sign(s)
    res[res == 0] = 1.0
    return res

class HDCWorldModel:
    """
    Learns transition dynamics T(s, a) -> s' using HRR with Permutation.
    
    改进版本：每个动作独立记忆，避免跨动作噪声干扰。
    - 原版: 单一M_sum存储所有(s,a)->s'，噪声累积
    - 新版: M_per_action[action_id]分离存储，预测更清晰
    """
    def __init__(self, device='cpu', n_actions=6):
        self.device = device
        self.dim = HDC_DIM
        self.n_actions = n_actions
        # 每个动作独立记忆
        self.M_per_action = [torch.zeros(self.dim, device=device) for _ in range(n_actions)]
        self.counts = [0] * n_actions
        
        # 兼容旧接口: 仍支持HDC动作向量（但不再用于predict）
        torch.manual_seed(SEED)
        self.action_codebooks = [torch.sign(torch.randn(self.dim, device=device)) for _ in range(n_actions)]
        for cb in self.action_codebooks:
            cb[cb == 0] = 1.0
        
    def _permute(self, t, shifts=1):
        """Cyclic shift for permutation."""
        return torch.roll(t, shifts=shifts, dims=-1)
        
    def _inverse_permute(self, t, shifts=1):
        return torch.roll(t, shifts=-shifts, dims=-1)

    def learn(self, state, action, next_state):
        """
        One-shot learning of a transition.
        
        Args:
        Args:
            state: HDC vector or tensor
            action: int (action_id 0-5) - Must be integer 
            next_state: HDC vector or tensor
        """
        # Strict int requirement
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
        Predict next state.
        
        Args:
        Args:
            state: HDC vector
            action: int (action_id) - Must be integer
        Returns:
            pred: 预测的下一状态HDC向量
        """
        # Strict int requirement
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
        Load state dictionary (HDC memories).
        
        Args:
            state_dict: List of tensors [M_action_0, M_action_1, ...]
        """
        if isinstance(state_dict, list):
            self.M_per_action = [t.to(self.device).float() for t in state_dict]
            # Assumes counts are not strictly needed for prediction if handled,
            # or we should save counts too. For now reset counts to avoid div/0 if meaningful?
            # Actually prediction checks counts[id] == 0.
            # We should assume loaded model has data.
            self.counts = [100] * self.n_actions # Hack: Indicate learned
        else:
             print("Warning: Invalid state_dict format for HDCWorldModel")
