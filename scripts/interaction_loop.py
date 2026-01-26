import sys
import os
import time
import torch
import numpy as np

# 将项目根目录添加到路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.voice_body.environment import AudioEnvironment
from src.lsm import AION_LSM_Network
from src.adapter import RandomProjectionAdapter
from src.voice_body.motor_cortex import MotorCortex
from src.drive import SocialDrive
from src.gwt import GlobalWorkspace
from src.hrr import HDCWorldModel
from src.dashboard import AIONDashboard
from src.mhn import ModernHopfieldNetwork
from src.config import OBS_SHAPE, LSM_STEPS_PER_SAMPLE

class InteractionAgent:
    def __init__(self, device='cpu'):
        self.device = device
        print("正在初始化 AION 语音智能体（运动动力学版）...")
        
        # 1. 身体与环境 (Body & Environment)
        try:
            import sounddevice as sd
            sd.query_devices(kind='input')
            self.env = AudioEnvironment(use_microphone=True)
            print("✅ 麦克风和音箱初始化成功。")
        except Exception:
            print("⚠️ 硬件初始化失败。正在回退到模拟模式 (SIMULATION)。")
            self.env = AudioEnvironment(use_microphone=False)
            
        # 2. 感知 (耳蜗 - Ear)
        self.lsm = AION_LSM_Network()
        self.adapter = RandomProjectionAdapter()
        
        # 3. 大脑 (记忆与全局工作空间 - Brain, Memory & GWT)
        self.gwt = GlobalWorkspace(device=device)
        self.drive = SocialDrive()
        self.wm = HDCWorldModel(n_actions=1, device=device)
        
        if os.path.exists("association_memory.pt"):
            print("正在加载关联记忆...")
            self.wm.M_per_action = torch.load("association_memory.pt")
            
        # 4. 动作 (运动皮层 - Motor Cortex)
        self.motor = MotorCortex()
        if os.path.exists("motor_cortex_weights.pt"):
            self.motor.load_weights("motor_cortex_weights.pt")
        
        # 5. 情节记忆 (MHN)
        print("正在初始化情节记忆 (MHN)...")
        self.memory = ModernHopfieldNetwork(device=device)

        # 6. 控制面板 (Dashboard)
        try:
            self.dashboard = AIONDashboard()
            print("✅ 控制面板已连接。")
        except Exception:
            self.dashboard = None

        self.state = "LISTEN" 
        self.silence_counter = 0

    def run(self):
        print("\n=== AION 运动交互循环已启动 ===")
        print("按 Ctrl+C 停止。")
        try:
            while True:
                if self.state == "LISTEN":
                    # 倾听时意图为 None
                    obs, _, _, _, _ = self.env.step(intent_vector=None)
                    activity = np.mean(obs)
                    
                    if activity > 0.05: 
                        print(f"👂 听到声音 (强度: {activity:.2f})")
                        spikes = self.process_hearing(obs)
                        self.state = "PONDER"
                        self.silence_counter = 0
                        self.drive.step(heard_voice=True)

                        if self.dashboard:
                            self.dashboard.update_env_view(obs)
                            self.dashboard.update_lsm_raster(spikes)

                        current_concept = self.gwt.current_sense
                        self.memory.add_memory(current_concept)
                    else:
                        self.silence_counter += 1
                        time.sleep(0.1)
                        self.drive.step(heard_voice=False)
                        
                        if self.drive.loneliness > 0.5 and self.silence_counter > 50:
                            print("😞 感到寂寞... 正在主动开启对话。")
                            self.state = "SPEAK_INITIATIVE"
                            
                elif self.state == "PONDER":
                    current_concept = self.gwt.current_sense
                    print("🤔 思考中...")
                    
                    energy = self.memory.compute_energy(current_concept)
                    recalled = self.memory.retrieve(current_concept)
                    
                    if self.dashboard:
                        self.dashboard.update_energy(energy)
                        sim = torch.nn.functional.cosine_similarity(current_concept, recalled, dim=-1).item()
                        self.dashboard.update_hdc_similarity(sim)

                    reply_concept = self.wm.predict(current_concept, 0)
                    self.gwt.update_pred(reply_concept)
                    self.state = "SPEAK"
                    
                elif self.state == "SPEAK":
                    print("🗣️ 正在回复...")
                    concept = self.gwt.current_pred
                    self.speak(concept)
                    self.state = "LISTEN"
                    
                elif self.state == "SPEAK_INITIATIVE":
                    print("🗣️ 正在主动发起对话...")
                    fake_concept = torch.randn(10000).to(self.device).sign()
                    fake_concept[fake_concept==0] = 1.0
                    self.speak(fake_concept)
                    self.state = "LISTEN"
                    self.silence_counter = 0

                if self.dashboard:
                    surprise = self.gwt.compute_surprise()
                    free_energy = self.drive.compute_free_energy(surprise)
                    self.dashboard.update_survival(free_energy=free_energy, loneliness=self.drive.loneliness)
                    
        except KeyboardInterrupt:
            print("\n已停止。")

    def process_hearing(self, obs):
        self.lsm.reset()
        spikes_accumulated = np.zeros(self.lsm.n_neurons)
        for _ in range(LSM_STEPS_PER_SAMPLE):
            spikes_accumulated += self.lsm.step(obs)
            
        concept = self.adapter.forward(torch.from_numpy(spikes_accumulated.copy()).float())
        concept = concept.to(self.device)
        self.gwt.update_sense(concept)
        return spikes_accumulated
        
    def speak(self, concept):
        """通过运动皮层生成并播放声音。"""
        # 环境类处理音箱逻辑
        self.env.step(intent_vector=concept)
        self.drive.step(spoke=True)

if __name__ == "__main__":
    agent = InteractionAgent()
    agent.run()
