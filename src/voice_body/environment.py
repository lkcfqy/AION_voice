import numpy as np
import time
import sounddevice as sd
from src.voice_body.cochlea import Cochlea
from src.voice_body.vocoder import Vocoder
from src.config import OBS_SHAPE

class AudioEnvironment:
    """
    Audio Environment for AION Voice.
    Replaces PyBulletEnv.
    
    Sensors: Microphone (via Cochlea)
    Actuators: Speaker (via Vocoder)
    """
    def __init__(self, use_microphone=False):
        self.cochlea = Cochlea()
        self.vocoder = Vocoder()
        self.use_mic = use_microphone
        
        # Buffer for 'visual' obs
        self.current_obs = np.zeros(OBS_SHAPE)
        
        # Audio params
        self.block_size = 16384 # ~1s at 16k
        self.sr = 16000
        
    def reset(self):
        """Return blank observation."""
        self.current_obs = np.zeros(OBS_SHAPE)
        return self.current_obs, {}
        
    def step(self, action_spectrogram=None):
        """
        Step the environment.
        Args:
            action_spectrogram: Generated 'Visual' Sound from Broca (64,64,3) or None.
        Returns:
            obs: (64, 64, 3) Spectrogram from Mic
            reward: 0 (Intrinsic only)
            done: False
            truncated: False
            info: {}
        """
        # 1. Execute Action (Speak)
        if action_spectrogram is not None:
             audio_out = self.vocoder.inverse(action_spectrogram)
             if self.use_mic:
                 # Blocking playback? Or Async?
                 # ideally async, but for turn-taking, blocking is fine.
                 sd.play(audio_out, self.sr)
                 sd.wait()
        
        # 2. Perception (Listen)
        if self.use_mic:
            # Record 1 second
            recording = sd.rec(int(self.block_size), samplerate=self.sr, channels=1, blocking=True)
            # Digital Gain (Pre-amp): Boost mic levels to match Training Data (LibriSpeech)
            # Raw mic is often -30dB vs normalized 0dB training data.
            recording = recording.flatten() * 5.0
            
            # Process to Spectrogram
            self.current_obs = self.cochlea.process(recording)
        else:
            # Simulation Mode: Silence or Random Noise
            # For debugging, maybe return the action as an echo?
            if action_spectrogram is not None:
                # Perfect Echo
                self.current_obs = action_spectrogram.copy()
            else:
                self.current_obs = np.zeros(OBS_SHAPE)
            
        reward = 0.0
        done = False
        
        return self.current_obs, reward, done, False, {}

    def listen_file(self, filename):
        """Process a wav file as observation."""
        import librosa
        y, sr = librosa.load(filename, sr=self.sr)
        # Take first 1s
        if len(y) > self.block_size:
            y = y[:self.block_size]
        else:
            y = np.pad(y, (0, self.block_size - len(y)))
            
        self.current_obs = self.cochlea.process(y)
        return self.current_obs
