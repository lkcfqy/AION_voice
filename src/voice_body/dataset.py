import os
import glob
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
from src.voice_body.cochlea import Cochlea
from src.config import OBS_SHAPE

class AudioDataset(Dataset):
    """
    Dataset for loading raw audio files (wav/flac) and converting to Spectrograms.
    """
    def __init__(self, root_dir, sr=16000, duration=1.0, file_ext="*.flac"):
        """
        Args:
            root_dir: Directory containing audio files (recursive search).
            sr: Target sample rate.
            duration: Target duration in seconds (files will be cropped/padded).
            file_ext: File extension to search for.
        """
        self.root_dir = root_dir
        self.sr = sr
        self.target_len = int(sr * duration)
        
        # Recursive search
        search_path = os.path.join(root_dir, "**", file_ext)
        self.files = glob.glob(search_path, recursive=True)
        
        if len(self.files) == 0:
            # Try wav if flac not found
            search_path = os.path.join(root_dir, "**", "*.wav")
            self.files = glob.glob(search_path, recursive=True)
            
        print(f"found {len(self.files)} audio files in {root_dir}")
        
        self.cochlea = Cochlea(sample_rate=sr)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        
        # Load audio
        # librosa loads as float32, normalized -1 to 1
        try:
            y, _ = librosa.load(path, sr=self.sr)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(OBS_SHAPE).permute(2, 0, 1) # Return dummy
            
        # Fix length (Pad or Crop)
        if len(y) > self.target_len:
            # Random crop for training variety? Or center?
            # Let's do random crop
            start = np.random.randint(0, len(y) - self.target_len)
            y = y[start : start + self.target_len]
        else:
            # Pad with zeros
            pad_len = self.target_len - len(y)
            y = np.pad(y, (0, pad_len))
            
        # Process via Cochlea -> Spectrogram (64, 64, 3)
        spectrogram = self.cochlea.process(y)
        
        # To Tensor
        # Obs is (64, 64, 3). PyTorch usually prefers (C, H, W) for Conv2D,
        # but BrocaNet (Phase 2) is MLP taking flat input, so (H, W, C) is fine if flattened.
        # Wait, BrocaNet Phase 2 is MLP.
        # Let's keep it as numpy (H, W, C) -> Tensor.
        
        spec_t = torch.from_numpy(spectrogram).float()
        
        # If we use Conv2d later, we might want permute. keeping as is matching env outputs.
        return spec_t
