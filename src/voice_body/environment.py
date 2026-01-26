import numpy as np
import os
import sounddevice as sd
from src.voice_body.cochlea import Cochlea
from src.voice_body.parametric_synthesizer import ParametricSynthesizer
from src.voice_body.motor_cortex import MotorCortex
from src.config import OBS_SHAPE

class AudioEnvironment:
    """
    AION 语音音频环境（运动动力学版本）。
    
    传感器：麦克风（通过耳蜗 Cochlea）
    执行器：扬声器（通过参数化合成器 ParametricSynthesizer）
    """
    def __init__(self, use_microphone=False):
        self.cochlea = Cochlea()
        self.synth = ParametricSynthesizer()
        self.motor = MotorCortex()
        if os.path.exists("motor_cortex_weights.pt"):
            self.motor.load_weights("motor_cortex_weights.pt")
        self.use_mic = use_microphone
        
        # “音频”观察结果的缓冲区
        self.current_obs = np.zeros(OBS_SHAPE)
        
        # 音频参数
        self.block_size = 16384 # 16k 采样率下约 1s
        self.sr = 16000
        
    def reset(self):
        """返回空白观察结果。"""
        self.current_obs = np.zeros(OBS_SHAPE)
        return self.current_obs, {}
        
    def step(self, intent_vector=None):
        """
        环境推进一步。
        参数:
            intent_vector: HDC 意图向量 (10000) 或 None。
        返回:
            obs: 来自麦克风的 (64, 64, 3) 频谱图
        """
        # 1. 执行动作（通过运动控制发声）
        if intent_vector is not None:
            # A. 运行运动皮层获取发音参数
            m_params = self.motor.step(intent_vector)
            
            # B. 根据这些源自大脑的参数合成音频
            audio_out = self.synth.synthesize_from_params(m_params, duration_s=0.2)
            
            if self.use_mic:
                sd.play(audio_out, self.sr)
                sd.wait()
        
        # 2. 感知（倾听）
        if self.use_mic:
            recording = sd.rec(int(self.block_size), samplerate=self.sr, channels=1, blocking=True)
            recording = recording.flatten() * 5.0
            self.current_obs = self.cochlea.process(recording)
        else:
            # 模拟模式：逻辑保持相似
            if intent_vector is not None:
                # 大脑产生内容的感知反馈
                m_params = self.motor.step(intent_vector)
                audio_feedback = self.synth.synthesize_from_params(m_params, duration_s=0.2)
                self.current_obs = self.cochlea.process(audio_feedback)
            else:
                self.current_obs = np.zeros(OBS_SHAPE)
            
        return self.current_obs, 0.0, False, False, {}

