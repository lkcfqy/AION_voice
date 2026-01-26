import sys
import os
import numpy as np
import torch

# 将项目根目录添加到路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.voice_body.parametric_synthesizer import ParametricSynthesizer
from src.voice_body.motor_cortex import MotorCortex
from src.voice_body.cochlea import Cochlea
from src.lsm import AION_LSM_Network
import os

def run_formal_babbling(n_samples=150):
    """
    正式校准：教授运动 LSM 如何将神经模式映射到发音参数。
    使用多步模拟以获得稳定的神经活动。
    """
    print(f"正在启动正式运动校准（{n_samples} 个样本，每个样本 50ms）...")
    
    synth = ParametricSynthesizer()
    motor = MotorCortex(n_output_params=10) 
    
    vowel_types = ['a', 'i', 'u', 'e', 'o']
    all_activities = []
    all_targets = []
    
    for i in range(n_samples):
        v = np.random.choice(vowel_types)
        seed_val = ord(v)
        rng = np.random.RandomState(seed_val)
        intent = rng.randn(10000)
        intent += np.random.randn(10000) * 0.1
        
        # 1. 模拟 50 个步骤 (50ms) 以获得稳定的平均脉冲发放率
        # 我们需要一种新方法来获取用于训练的神经活动。
        # 我们可以修改 MotorCortex.step 以便在需要时返回中间状态，
        # 或者直接使用其当前的实现。
        
        # 目前的 motor.step(intent, n_steps=50) 返回的是参数，
        # 但我们在这里希望获取原始的神经活动，以便自己训练读取层。
        
        # 技巧：运行模拟并获取累积的脉冲 (Spikes)
        motor.current_bias[:] = motor.process_intent(intent)
        accumulated_spikes = np.zeros(motor.n_neurons)
        for _ in range(50):
            motor.sim.step()
            accumulated_spikes += motor.sim.data[motor.spike_probe][-1]
        
        avg_activity = accumulated_spikes / (50 * motor.dt)
        
        # 2. 获取目标参数
        target_map = {
            'a': [800, 1200, 2500, 3500, 100, 100, 150, 200, 220, 1.0],
            'i': [300, 2300, 3000, 4000, 50, 150, 200, 200, 220, 1.0],
            'u': [300, 800, 2200, 3200, 50, 100, 150, 150, 220, 1.0],
            'e': [500, 1800, 2600, 3600, 100, 150, 200, 200, 220, 1.0],
            'o': [500, 1000, 2400, 3400, 100, 100, 150, 150, 220, 1.0],
        }
        target = np.array(target_map[v])
        
        all_activities.append(avg_activity)
        all_targets.append(target)
        
        if (i+1) % 100 == 0:
            print(f"   已收集 {i+1} 个样本...")
            
    # 3. 训练 (Training)
    X = np.array(all_activities)
    Y = np.array(all_targets)
    
    print(f"正在根据 {X.shape} 形状的神经活动训练读取层...")
    motor.train_readout(X, Y)
    
    # 3. 持久化 (Persistence)
    model_path = "motor_cortex_weights.pt"
    # 保存整个读取层权重矩阵
    torch.save({
        'readout_weights': motor.readout_weights,
        'intent_projection': motor.intent_projection
    }, model_path)
    print(f"✅ 校准成功。权重已保存至 {model_path}")
    
    # 4. 最终验证 (Final Validation)
    print("\n正在验证元音 'a' 的生成...")
    test_intent = np.random.RandomState(ord('a')).randn(10000)
    pred_params = motor.step(test_intent)
    print(f"预测生成的 'a' 参数（前 3 位）: {pred_params[:3]}")
    print("目标 'a' 参数（前 3 位）: [800, 1200, 2500]")

if __name__ == "__main__":
    run_formal_babbling()
