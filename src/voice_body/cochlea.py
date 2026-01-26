import numpy as np
import librosa
import torch
from src.config import OBS_SHAPE

class Cochlea:
    """
    Electronic Cochlea (Audio Input System).
    Converts raw audio waveform -> 2D Spectrogram (LSM Input).
    
    Design:
    - Input: Raw Audio (1D array)
    - Process: Mel-Spectrogram or STFT
    - Output: Normalized 2D Matrix (Freq x Time) -> Resized to OBS_SHAPE (e.g. 64x64)
    """
    def __init__(self, sample_rate=16000, n_mels=64, hop_length=256):
        self.sr = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.target_shape = (OBS_SHAPE[0], OBS_SHAPE[1]) # e.g. (64, 64)
        
    def process(self, audio_segment):
        """
        Process audio segment into spectrogram.
        Args:
            audio_segment: numpy array (1D)
        Returns:
            spectrogram: numpy array (64, 64, 1 or 3) normalized [0,1]
        """
        # 1. Pre-emphasis (optional, biological realism)
        y = librosa.effects.preemphasis(audio_segment)
        
        # 2. Mel Spectrogram
        # n_fft determines frequency resolution. 
        # For 64 mels, n_fft should be sufficient (e.g. 512 or 1024)
        mels = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels, 
            n_fft=1024, hop_length=self.hop_length
        )
        
        # 3. Log scale (dB) - mimics human hearing
        # CRITICAL FIX: Use fixed reference (1.0) instead of np.max (per-frame).
        # Per-frame norm (np.max) amplifies silence/noise to 0dB (Max=1.0), causing constant trigger and noise.
        # ref=1.0 assumes input audio is [-1, 1], so max power is 1.0 (0dB).
        mels_db = librosa.power_to_db(mels, ref=1.0)
        
        # 4. Normalize to [0, 1]
        # dB range is usually [-80, 0] or similar.
        # We clip bottom at -80dB
        mels_db = np.clip(mels_db, -80, 0)
        # Normalize
        img = (mels_db + 80) / 80.0
        
        # 5. Resize/Crop to target shape (64x64)
        # Current shape: (n_mels, time_steps)
        # We want (64, 64)
        
        # If time steps < 64, pad
        # If time steps > 64, crop (or resize)
        # Strategy: Resize (Stretch/Squeeze) or Rolling Window?
        # For an agent, "Rolling Window" or "Snapshot" is better.
        # Here we assume the input audio_segment roughly corresponds to the window size we want.
        # But to be robust, let's resize (image interpolation).
        
        # img format is (freq, time). We want (H, W).
        # We can treat it as an image.
        
        # Use simple interpolation
        from scipy.ndimage import zoom
        
        curr_h, curr_w = img.shape
        target_h, target_w = self.target_shape
        
        # Calculate zoom factors
        zoom_h = target_h / curr_h
        zoom_w = target_w / curr_w
        
        resized = zoom(img, (zoom_h, zoom_w))
        
        # 6. Add Channel dim if needed (OBS_SHAPE has 3 channels?)
        # OBS_SHAPE is (56, 56, 3) or (64, 64, 3)
        # Let's check config. OBS_SHAPE is (56, 56, 3) in current file.
        # We need to match it.
        
        # Replicate to 3 channels (RGB) since LSM expects it (or update LSM)
        # Or just use 1 channel and zeros for others.
        # Replicating is safer for visual pretraining compatibility.
        
        output = np.stack([resized, resized, resized], axis=-1)
        
        return output

    def get_audio_len_for_image(self):
        """Return estimated samples needed to fill one image roughly without stretching"""
        # If hop_length=256, and we want 64 time steps.
        # samples = 256 * 64 = 16384 samples (~1 second at 16k)
        return self.hop_length * self.target_shape[1]
