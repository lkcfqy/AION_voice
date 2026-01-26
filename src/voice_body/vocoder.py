import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import zoom

class Vocoder:
    """
    Vocoder (Audio Output System).
    Converts 2D Spectrogram (LSM Output/Imagination) -> Raw Audio Waveform.
    Using Griffin-Lim for non-trainable reconstruction.
    """
    def __init__(self, sample_rate=16000, n_mels=64, hop_length=256):
        self.sr = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = 1024 # Must match cochlea
        
    def inverse(self, spectrogram_img, duration_sec=None):
        """
        Convert spectrogram image back to audio.
        Args:
            spectrogram_img: (64, 64, 3) or (64, 64) normalized [0, 1]
            duration_sec: Target duration in seconds (optional)
        """
        # 1. Image -> Mel Spectrogram (Denormalize)
        # Take first channel if 3D
        if len(spectrogram_img.shape) == 3:
            img = spectrogram_img[:, :, 0]
        else:
            img = spectrogram_img
            
        # 2. Resize back to spectral domain?
        # Ideally, we want detailed time steps.
        # But we only have 64 time steps from the image.
        # We can stretch it in time domain to match target duration?
        # Or just synthesize 64 frames.
        
        # Denormalize: [0, 1] -> [-80, 0] dB
        mels_db = (img * 80.0) - 80.0
        
        # dB -> Power
        mels = librosa.db_to_power(mels_db)
        
        # 3. Griffin-Lim Reconstruction
        # We need Linear S_spectrogram first? 
        # Librosa feature.inverse.mel_to_stft is needed.
        # Ensure librosa >= 0.7.0
        
        try:
           stft_reconstructed = librosa.feature.inverse.mel_to_stft(
               mels, sr=self.sr, n_fft=self.n_fft, power=2.0
           )
           
           # Griffin-Lim
           audio_reconstructed = librosa.griffinlim(
               stft_reconstructed, n_iter=32, hop_length=self.hop_length, win_length=self.n_fft
           )
           
           # 5. Inverse Pre-emphasis (De-emphasis)
           # y[t] = x[t] + 0.97 * y[t-1]  (Inverse of y[t] = x[t] - 0.97 * x[t-1])
           # Using scipy filter
           from scipy.signal import lfilter
           # Transfer function: H(z) = 1 / (1 - 0.97 z^-1)
           audio_reconstructed = lfilter([1], [1, -0.97], audio_reconstructed)
           
        except AttributeError:
             print("Warning: librosa version too old? Fallback.")
             return np.zeros(100)
             
        return audio_reconstructed

    def save_audio(self, audio, filename="output.wav"):
        sf.write(filename, audio, self.sr)
