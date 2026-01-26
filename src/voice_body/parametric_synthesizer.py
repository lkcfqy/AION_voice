import numpy as np
import pyworld as pw
import soundfile as sf
import os

class ParametricSynthesizer:
    """
    使用 WORLD 声码器的参数化语音合成器。
    作为受运动 LSM 控制的“肌肉”。
    """
    def __init__(self, sample_rate=16000, frame_period=5.0):
        self.fs = sample_rate
        self.frame_period = frame_period # ms
        
    def synthesize(self, f0, sp, ap):
        """
        根据 WORLD 参数合成波形。
        f0: (T,) 基频
        sp: (T, N_fft//2 + 1) 频谱包络 (Spectral Envelope)
        ap: (T, N_fft//2 + 1) 非周期性参数 (Aperiodicity)
        """
        # 确保类型正确
        f0 = f0.astype(np.float64)
        sp = sp.astype(np.float64)
        ap = ap.astype(np.float64)
        
        y = pw.synthesize(f0, sp, ap, self.fs, frame_period=self.frame_period)
        return y

    def synthesize_from_params(self, params, duration_s=0.2):
        """
        根据 10 维控制向量合成声音。
        params: [F1, F2, F3, F4, B1, B2, B3, B4, F0, 发音性质 Voicing]
        """
        f1, f2, f3, f4, b1, b2, b3, b4, f0_val, voicing = params
        
        num_frames = int(duration_s * 1000 / self.frame_period)
        f0 = np.ones(num_frames) * max(50, f0_val) # 将 F0 限制在人耳可听范围内
        
        fft_size = pw.get_cheaptrick_fft_size(self.fs)
        freqs = np.linspace(0, self.fs / 2, fft_size // 2 + 1)
        
        # 基础包络
        envelope = np.exp(-freqs / 2000.0)
        
        def add_formant(env, f_center, width):
            # 确保带宽为正数
            width = max(10, width)
            return env + 2.0 * np.exp(-((freqs - f_center)**2) / (2 * width**2))
            
        envelope = add_formant(envelope, f1, b1)
        envelope = add_formant(envelope, f2, b2)
        envelope = add_formant(envelope, f3, b3)
        envelope = add_formant(envelope, f4, b4)
        
        sp_frame = envelope**2 * 1e-3
        sp = np.tile(sp_frame, (num_frames, 1))
        
        # 基于发音性质的 AP (1.0 = 纯有声, 0.0 = 纯噪声)
        ap_val = 0.001 if voicing > 0.5 else 0.5
        ap = np.ones((num_frames, fft_size // 2 + 1)) * ap_val
        
        return self.synthesize(f0, sp, ap)


    def save_wav(self, audio, path):
        sf.write(path, audio, self.fs)
        print(f"音频已保存至 {path}")

if __name__ == "__main__":
    # Test generation
    synth = ParametricSynthesizer()
    # Smoke test synthesis
    params = np.array([500, 1000, 2400, 3400, 100, 100, 150, 150, 220, 1.0])
    wave = synth.synthesize_from_params(params)
    synth.save_wav(wave, "test_synth.wav")
