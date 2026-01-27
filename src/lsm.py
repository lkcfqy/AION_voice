import nengo
import numpy as np
from src.config import (
    LSM_N_NEURONS, LSM_SPARSITY, LSM_IN_SPARSITY, 
    TARGET_FIRING_RATE, PLASTICITY_RATE, RATE_TAU,
    TAU_RC, TAU_REF, DT, OBS_SHAPE, SEED
)

# 生产级 LSM，支持动态输入和在线学习
class AION_LSM_Network:
    """
    生产级液体状态机 (LSM)，具有：
    - 动态输入回调（基于 lambda 的 Node）
    - 三因子赫布学习 (Hebbian learning)（多巴胺调节）
    - 通过 scipy.sparse 生成稀疏权重
    - 支持运行时权重修改
    """
    def __init__(self):
        self.current_input = np.zeros(np.prod(OBS_SHAPE))
        self.cognitive_input_val = np.zeros(LSM_N_NEURONS) # 用于接收来自 HDC 的 Top-down 信号
        
        # （概念上）重写父类中的 input_node 定义
        # 我们只是在 __init__ 内部有效地复制了逻辑
        
        self.dt = DT
        self.n_neurons = LSM_N_NEURONS
        self.input_size = np.prod(OBS_SHAPE)
        
        self.bias_correction = np.zeros(self.n_neurons)
        self.filtered_rates = np.zeros(self.n_neurons)
        self.rate_tau = RATE_TAU

        self.model = nengo.Network(label="AION_LSM", seed=SEED)
        
        with self.model:
            # 动态输入节点
            self.input_node = nengo.Node(
                lambda t: self.current_input, 
                size_out=self.input_size, 
                label="音频输入"
            )

            # 认知输入节点 (Top-down Bias)
            self.cognitive_node = nengo.Node(
                lambda t: self.cognitive_input_val,
                size_out=self.n_neurons,
                label="认知偏置"
            )

            # 储备池 (Reservoir)
            self.reservoir = nengo.Ensemble(
                n_neurons=self.n_neurons,
                dimensions=1, 
                neuron_type=nengo.LIF(tau_rc=TAU_RC, tau_ref=TAU_REF),
                seed=SEED
            )

            # 输入权重 (稀疏)
            # 注意：如果需要可重复性，我们必须确保 RNG (随机数生成器) 的一致性
            rng = np.random.RandomState(SEED)
            # 手动创建稀疏矩阵，以避免 Nengo 分布有时出现的形状问题
            # 形状 (n_neurons, input_size)
            # 生成完整的 12k*400 矩阵是非常沉重的 (~5M 个浮点数)。 
            # 12M 个浮点数 = ~48MB，对于内存来说完全没问题。
            
            # 使用 Nengo 的稀疏生成器
            import scipy.sparse

            # ... (在 __init__ 内部)
            # 输入权重 (稀疏)
            print("正在生成权重...")

            # 使用 scipy 生成稀疏权重
            # 密度 = 1.0 - LSM_IN_SPARSITY
            # 我们需要的连通性 = LSM_IN_SPARSITY (例如 0.1)
            # scipy.sparse.random 使用 density (0.0-1.0)
            
            # 生成 Nengo 兼容稀疏矩阵的助手函数
            def generate_sparse_weights(n_rows, n_cols, density, rng):
                # 使用 scipy.sparse.random
                # 使用 scipy.sparse.random

                
                S = scipy.sparse.random(n_rows, n_cols, density=density, format='csr', random_state=rng)
                # 将 [0, 1] 映射到高斯分布
                # 或者直接分配新数据
                if S.nnz > 0:
                   S.data = rng.standard_normal(S.nnz) * 0.005 # 缩放 0.005 (优化过的起始值)
                return S

            self.input_weights = generate_sparse_weights(self.n_neurons, self.input_size, LSM_IN_SPARSITY, rng)
            
            nengo.Connection(
                self.input_node, 
                self.reservoir.neurons, 
                transform=self.input_weights.toarray(), 
                synapse=None
            )

            # 将认知输入直接连接到神经元（全连接 1:1）
            nengo.Connection(
                self.cognitive_node,
                self.reservoir.neurons,
                synapse=0.01
            )

            # 递归权重 (稀疏)
            recurrent_weights = generate_sparse_weights(self.n_neurons, self.n_neurons, LSM_SPARSITY, rng)
            
            self.recurrent_conn = nengo.Connection(
                self.reservoir.neurons, 
                self.reservoir.neurons,
                transform=recurrent_weights.toarray(),
                synapse=0.01
            )

            # 稳态 (Homeostasis)
            def homeostasis_func(t, x):
                spikes = x
                alpha = self.dt / self.rate_tau
                inst_rate = spikes / self.dt
                self.filtered_rates += alpha * (inst_rate - self.filtered_rates)
                
                error = self.filtered_rates - TARGET_FIRING_RATE
                self.bias_correction -= PLASTICITY_RATE * error * self.dt
                
                return self.bias_correction

            self.homeostasis_node = nengo.Node(
                homeostasis_func,
                size_in=self.n_neurons,
                size_out=self.n_neurons,
                label="稳态调节"
            )
            
            nengo.Connection(self.reservoir.neurons, self.homeostasis_node, synapse=None)
            nengo.Connection(self.homeostasis_node, self.reservoir.neurons, synapse=0.01) # 叠加电流

            # 脉冲探针
            # 我们启用它以可视化光栅图
            self.spike_probe = nengo.Probe(self.reservoir.neurons)
            
        print("正在构建模拟器...")
        self.sim = nengo.Simulator(self.model, dt=self.dt, progress_bar=False)
        
        # 在模拟器中定位权重信号
        # 这样我们就可以在运行时修改权重
        self.weight_sig = self.sim.model.sig[self.recurrent_conn]['weights']
        
        # 赫布追踪 (Hebbian Traces)
        self.last_spikes = np.zeros(self.n_neurons)

        # 读出层 (Readout Layer) - 用于生成音频频谱
        # 形状: (n_neurons, input_size) 即 (2000, 64)
        self.W_out = np.zeros((self.n_neurons, self.input_size))

    def step(self, spectrogram_input, dopamine=0.0, cognitive_bias=None):
        """
        运行模拟的一个步骤。
        spectrogram_input: (Obs Shape) 形状的 numpy 数组
        dopamine: 标量强化信号 (-1.0 到 1.0)
        cognitive_bias: (N_neurons,) 形状的 numpy 数组，用于 top-down 干扰
        """
        # 1. 更新输入
        if cognitive_bias is not None:
            self.cognitive_input_val[:] = cognitive_bias
        else:
            self.cognitive_input_val.fill(0)

        flat = spectrogram_input.flatten()
        if flat.max() > 1.1: 
            flat = flat / 255.0
            
        self.current_input[:] = flat 
        
        # 2. 模拟器前进
        self.sim.step()
        
        # 3. 获取脉冲
        spikes = self.sim.data[self.spike_probe][-1]
        
        # 4. 应用三因子学习规则（权重更新）
        # dW = eta * D * (Post * Pre)
        # Nengo 默认权重是 (Post, Pre)
        # 我们使用一个简单的类 STDP 规则（基于频率/脉冲事件）？
        # 由于这些是脉冲（0 或 1/dt），只有当两者都触发时，乘积才为 1。
        # 这是赫布重合。
        
        if dopamine != 0.0:
            learning_rate = 1e-4
            
            # 识别协同激活的神经元
            # 我们使用当前的脉冲作为 Post，上一时刻的脉冲作为 Pre（因果性）
            # 或者只是瞬时重合。
            # 为了在这个基于时间步的模型中简化，我们先使用瞬时的，
            # 或者更好的方式是：Pre=self.last_spikes, Post=spikes
            
            pre = self.last_spikes
            post = spikes
            
            # 外积 -> (N_post, N_pre)
            # 仅当有活动时才更新
            if np.any(pre) and np.any(post):
                dW = learning_rate * dopamine * np.outer(post, pre)
                
                # 将更新应用到模拟器信号
                # 注意：这是对 Nengo 的内部访问
                # 如果需要，强制设置为可写
                weights = self.sim.signals[self.weight_sig]
                if not weights.flags.writeable:
                    weights.setflags(write=1)
                
                weights += dW
                
                # 限制权重？也许可以防止权重爆炸
                # self.sim.signals[self.weight_sig] = np.clip(self.sim.signals[self.weight_sig], -1.0, 1.0)
        
        # Store history
        self.last_spikes = spikes.copy()
        
        # 计算预测输出 (Prediction)
        # prediction = np.dot(spikes, self.W_out)
        prediction = spikes @ self.W_out
        
        return spikes, prediction

    def update_readout(self, all_spikes, all_targets, lambda_reg=1.0):
        """
        使用岭回归 (Ridge Regression) 一次性更新读出层权重 W_out。
        数学原理: W_out = (S^T S + lambda * I)^-1 S^T Y
        all_spikes: 形状为 (N_samples, N_neurons) 的矩阵
        all_targets: 形状为 (N_samples, OBS_SHAPE) 的矩阵
        """
        print(f"正在更新读出层权重 (样本数: {len(all_spikes)})...")
        
        S = all_spikes
        Y = all_targets
        
        # 岭回归公式求解
        # S.T @ S 形状 (N_neurons, N_neurons)
        I = np.eye(self.n_neurons)
        A = S.T @ S + lambda_reg * I
        B = S.T @ Y
        
        # 求解线性方程组 A * W_out = B
        self.W_out = np.linalg.solve(A, B)
        print("✅ 读出层权重更新完成。")

    def reset(self):
        """无需重新构建即可重置模拟器状态。"""
        self.sim.reset()
        self.last_spikes = np.zeros(self.n_neurons)
        # 如果需要，重新定位权重信号（通常保持不变）
        # 注意：权重属于信号状态的一部分，但递归权重通常在 sim.signals 中修改。
        # 如果我们想在重置时保留学习到的权重，sim.reset() 是好的，因为它会重置神经元状态
        # 但如果全局信号值被修改了，不一定会重置它们。
        # 实际上，sim.reset() 会将所有信号重置为它们的初始值。
        # 如果我们想保留权重，可能需要自定义重置函数。
        # 然而，对于 Broca 训练，我们目前并不更新 LSM 中的权重。
        # 所以 sim.reset() 是完美的。

