import nengo
import numpy as np
import torch
import os
from src.config import (
    HDC_DIM, LSM_N_NEURONS, LSM_SPARSITY, LSM_IN_SPARSITY,
    TAU_RC, TAU_REF, DT, SEED
)

class MotorCortex:
    """
    运动皮层（基于 LSM 的语音控制器）。
    将 HDC 意图向量转换为连续的发音参数。
    """
    def __init__(self, n_output_params=10):
        self.n_neurons = LSM_N_NEURONS
        self.input_dim = HDC_DIM
        self.output_dim = n_output_params
        self.dt = DT
        
        # 1. 状态：HDC 意图投影矩阵（目前为静态）
        rng = np.random.RandomState(SEED)
        self.intent_projection = rng.randn(self.n_neurons, self.input_dim) * 0.1
        
        # 2. 状态：输出读取权重 (Readout Weights)
        self.readout_weights = rng.randn(self.output_dim, self.n_neurons) * 0.01
        
        # 动态输入状态
        self.current_bias = np.zeros(self.n_neurons)
        
        # 3. Nengo 模型
        self.model = nengo.Network(label="运动皮层", seed=SEED)
        with self.model:
            # Lambda 节点从 self.current_bias 读取数据
            self.input_node = nengo.Node(lambda t: self.current_bias)
            
            # 递归储备池 (Recurrent Reservoir)
            self.reservoir = nengo.Ensemble(
                n_neurons=self.n_neurons,
                dimensions=1, 
                neuron_type=nengo.LIF(tau_rc=TAU_RC, tau_ref=TAU_REF),
                seed=SEED
            )
            
            # 将输入连接到神经元
            nengo.Connection(self.input_node, self.reservoir.neurons, synapse=None)
            
            # 递归连接
            recurrent_weights = rng.randn(self.n_neurons, self.n_neurons) * 0.05
            mask = rng.rand(self.n_neurons, self.n_neurons) < LSM_SPARSITY
            recurrent_weights = recurrent_weights * mask
            
            nengo.Connection(
                self.reservoir.neurons, 
                self.reservoir.neurons, 
                transform=recurrent_weights,
                synapse=0.01
            )
            
            self.spike_probe = nengo.Probe(self.reservoir.neurons)
            
        self.sim = nengo.Simulator(self.model, dt=self.dt, progress_bar=False)

    def load_weights(self, path):
        if os.path.exists(path):
            data = torch.load(path)
            self.readout_weights = data['readout_weights']
            self.intent_projection = data['intent_projection']
            print(f"MotorCortex: Weights loaded from {path}")
        else:
            print(f"MotorCortex: Weights not found at {path}")

    def process_intent(self, hdc_vector):
        if isinstance(hdc_vector, torch.Tensor):
            hdc_vector = hdc_vector.detach().cpu().numpy()
        return self.intent_projection @ hdc_vector

    def step(self, hdc_vector, n_steps=50):
        """
        模拟运动储备池 n_steps 步，以获得稳定的神经状态。
        返回: 发音参数（在窗口内取平均值）
        """
        # 1. 更新节点读取的状态
        self.current_bias[:] = self.process_intent(hdc_vector)
        
        # 2. 运行多个步骤，让“液体”沉淀到意图轨迹中
        # 我们在此窗口内累积脉冲 (Spikes)
        accumulated_spikes = np.zeros(self.n_neurons)
        
        for _ in range(n_steps):
            self.sim.step()
            accumulated_spikes += self.sim.data[self.spike_probe][-1]
            
        # 3. 从累积/平均后的活动中解码参数
        avg_activity = accumulated_spikes / (n_steps * self.dt)
        params = self.readout_weights @ avg_activity
        
        # 4. 后处理：确保参数符合物理有效性（无负频率）
        params = np.maximum(5.0, params) # 下限
        # 特别是对于 F0，让我们将其保持在人类范围内 [50, 500]
        params[8] = np.clip(params[8], 50, 500)
        
        return params

    def train_readout(self, X_activities, Y_params):
        """
        线性读取层的离线训练。
        X_activities: (T, N_Neurons) - 平均脉冲发放率
        Y_params: (T, N_Params)
        """
        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=1.0)
        reg.fit(X_activities, Y_params)
        self.readout_weights = reg.coef_
        print("运动皮层: 已通过岭回归更新读取层权重。")

if __name__ == "__main__":
    # Smoke test
    cortex = MotorCortex()
    intent = np.random.randn(HDC_DIM)
    params = cortex.step(intent)
    print(f"Produced parameters: {params.shape} -> {params[:3]}...")
