import torch
import torch.nn as nn
import numpy as np
from src.config import LSM_N_NEURONS, OBS_SHAPE, TAU_RC, TAU_REF, DT, DEVICE, SEED

class AION_LSM_Network(nn.Module):
    """
    基于 PyTorch 的液体状态机 (Liquid State Machine)
    使用 Leaky Integrate-and-Fire (LIF) 神经元模型。
    支持 GPU 加速。
    """
    def __init__(self, n_neurons=LSM_N_NEURONS, n_in=OBS_SHAPE, sparsity=0.1, device=DEVICE):
        super().__init__()
        self.n = n_neurons
        self.n_in = n_in
        self.device = device
        
        # 神经元参数 (Tensor)
        self.tau_rc = TAU_RC
        self.tau_ref = TAU_REF
        self.dt = DT
        
        # 衰减因子
        self.decay = np.exp(-self.dt / self.tau_rc)
        
        torch.manual_seed(SEED)
        
        # 1. 初始化权重 (保持 Chaos Edge, 谱半径 ~ 1.0)
        # 输入权重 (Sparse)
        self.W_in = self._init_sparse_weights(n_in, n_neurons, sparsity * 2, gain=5.0) # 增强输入增益
        
        # 递归权重 (Sparse, 谱半径调整)
        self.W_rec = self._init_sparse_weights(n_neurons, n_neurons, sparsity, gain=1.2) # 略微混沌
        
        # 读出权重 (可训练) - 初始化为零
        self.W_out = nn.Parameter(torch.zeros(n_neurons, n_in, device=device))
        
        # 2. 神经元状态
        self.reset()
        
        self.to(device)
        print(f"[LSM] LSM 初始化完成: {n_neurons} 神经元 (Device: {device})")

    def _init_sparse_weights(self, n_in, n_out, sparsity, gain=1.0):
        """生成稀疏连接矩阵"""
        # 使用 Kaiming Uniform 但稀疏化
        weights = torch.randn(n_in, n_out, device=self.device) * gain / np.sqrt(n_in)
        mask = torch.rand(n_in, n_out, device=self.device) < sparsity
        return weights * mask.float()

    def reset(self):
        """重置神经元状态 (电压, 不应期)"""
        self.v = torch.zeros(self.n, device=self.device) # 膜电位
        self.ref = torch.zeros(self.n, device=self.device) # 不应期计时器
        self.spikes = torch.zeros(self.n, device=self.device) # 脉冲输出

    def forward(self, input_signal, external_current=None):
        """
        一步仿真
        input_signal: (Batch, Input_Dim) 或 (Input_Dim)
        external_current: (Optional) 外部注入电流 (Batch, N_Neurons)
        """
        if input_signal.dim() == 1:
            input_signal = input_signal.unsqueeze(0) # (1, In)
            
        batch_size = input_signal.shape[0]
        
        # 确保状态维度匹配 Batch
        if self.v.shape[0] != batch_size and self.v.dim() == 1:
             # 如果是首次运行或 batch 变化，广播状态 (通常用于 Batch 训练)
             # 对于实时交互，batch=1
             pass 
 
        # 1. 电流整合
        # I = W_in * Input + W_rec * Spikes
        input_current = input_signal @ self.W_in
        rec_current = self.spikes @ self.W_rec
        total_current = input_current + rec_current
        
        if external_current is not None:
            total_current += external_current
        
        # 2. LIF 动力学更新
        # dv/dt = (I - v) / tau
        # v[t+1] = v[t] * decay + I * (1-decay)
        
        # 处于不应期的神经元电压保持为 0
        non_refractory = (self.ref <= 0).float()
        
        new_v = self.v * self.decay + total_current * (1 - self.decay)
        
        # 应用不应期屏蔽
        self.v = new_v * non_refractory
        
        # 3. 脉冲生成 (阈值 = 1.0)
        self.spikes = (self.v > 1.0).float()
        
        # 4. 复位与不应期设定
        # 发放脉冲后，电压重置为 0 (Hard Reset) 或减去阈值 (Soft Reset). 这里用 Hard Reset。
        self.v = self.v * (1 - self.spikes)
        
        # 设定不应期计时 (秒)
        self.ref = torch.where(self.spikes > 0, torch.ones_like(self.ref) * self.tau_ref, self.ref)
        self.ref -= self.dt
        
        return self.spikes

    def step(self, input_vec, dopamine=0.0, cognitive_bias=None):
        """
        完整的认知单步 (包含读出预测)
        """
        # 转为 Tensor
        if not isinstance(input_vec, torch.Tensor):
            x = torch.tensor(input_vec, dtype=torch.float32, device=self.device)
        else:
            x = input_vec
            
        # 注入 自上而下 (Top-down) 的偏置 (作为额外的电流注入)
        bias_tensor = None
        if cognitive_bias is not None:
            if not isinstance(cognitive_bias, torch.Tensor):
                 bias_tensor = torch.tensor(cognitive_bias, dtype=torch.float32, device=self.device)
            else:
                 bias_tensor = cognitive_bias
            # 假设 cognitive_bias 已经是 (N_Neurons,) 的维度
            
        # 运行动力学
        spikes = self.forward(x, external_current=bias_tensor)
        
        # 计算读出层输出 (预测值)
        # y = Spikes @ W_out
        prediction = spikes @ self.W_out
        
        # 学习规则 (如果开启在线学习/多巴胺)
        # 简单 Hebbian: dW = eta * dopamine * (Pre * Post)
        # 这里仅在 Sleep 阶段训练，所以实时步跳过 W_out 更新
        
        return spikes.cpu().numpy().flatten(), prediction.cpu().detach().numpy().flatten()
