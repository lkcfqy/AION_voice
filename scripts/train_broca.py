import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.voice_body.environment import AudioEnvironment
from src.lsm import AION_LSM_Network
from src.adapter import RandomProjectionAdapter
from src.voice_body.broca import BrocaNet
from src.voice_body.dataset import AudioDataset
from src.config import HDC_DIM

class BrocaTrainer:
    def __init__(self, device='cpu', data_dir=None):
        self.device = device
        self.data_dir = data_dir
        print("Initializing Components...")
        
        # 1. Environment (Mic & Speaker) - Not used in Dataset mode
        # self.env = AudioEnvironment(use_microphone=False) 
        
        # 2. Perception (Fixed)
        self.lsm = AION_LSM_Network()
        self.adapter = RandomProjectionAdapter()
        
        # 3. Generator (Trainable)
        self.broca = BrocaNet().to(self.device)
        self.optimizer = optim.Adam(self.broca.parameters(), lr=1e-4) # Lower LR for real data
        self.criterion = nn.MSELoss()
        
    def train_loop(self, n_samples=50000, batch_size=16):
        print(f"Starting Training Loop...")
        
        if self.data_dir and os.path.exists(self.data_dir):
            print(f"Protocol: Real Speech Data from {self.data_dir}")
            dataset = AudioDataset(self.data_dir)
            if len(dataset) == 0:
                print("❌ No files found in data dir. Fallback to simulation.")
                self._run_simulation_loop(n_samples)
                return
                
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            self._run_dataset_loop(loader, n_samples)
        else:
            print("Protocol: Random Noise Babbling (Simulation)")
            self._run_simulation_loop(n_samples)
            
    def _run_dataset_loop(self, loader, max_steps):
        step = 0
        total_loss = 0
        
        # Loop mainly based on epochs, but we limit total steps for demo
        while step < max_steps:
            for spectrograms in loader:
                # Specs: (Batch, 64, 64, 3) tensor
                batch_size = spectrograms.shape[0]
                target_img = spectrograms.to(self.device)
                
                # Forward Pass (One by one for LSM statefulness?)
                # LSM is stateful and usually takes 1 item.
                # Process concepts in loop or batch?
                # AION_LSM_Network is non-batch currently.
                # We need to process each item. A bit slow but correct for spiking sim.
                
                concepts = []
                for i in range(batch_size):
                    # Convert Tensor -> Numpy for LSM
                    # LSM step takes (64, 64, 3)
                    img_np = target_img[i].cpu().numpy()
                    
                    # Need to reset LSM for each sample to avoid cross-sample contamination?
                    # "Tuning the ear": maybe let state carry over (continuous hearing)?
                    # Ideally reset for i.i.d samples.
                    self.lsm.reset() 
                    
                    spikes = self.lsm.step(img_np)
                    # Fix for PyTorch/NumPy writable warning
                    spikes_t = torch.from_numpy(spikes.copy()).float()
                    c = self.adapter.forward(spikes_t)
                    concepts.append(c)
                
                concept_batch = torch.stack(concepts).to(self.device) # (B, 10000)
                
                # Helper: Normalize targets to [0,1] just in case
                # Dataset already output normalized.
                
                # Train Broca (Batch Mode)
                self.optimizer.zero_grad()
                reconstructed = self.broca(concept_batch) # (B, 64, 64, 3)
                
                loss = self.criterion(reconstructed, target_img)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                step += batch_size
                
                if (step // batch_size) % 10 == 0:
                     print(f"Step {step}/{max_steps} | Loss: {loss.item():.6f}")
                     
                if step >= max_steps: break
        
        self._save_model()

    def _run_simulation_loop(self, n_samples):
        loss_history = []
        for i in range(n_samples):
            target_spectrogram = self._generate_fake_spectrogram()
            
            # Reset LSM for fresh perception
            self.lsm.reset()
            spikes = self.lsm.step(target_spectrogram)
            
            spikes_t = torch.from_numpy(spikes).float()
            concept_hdc = self.adapter.forward(spikes_t)
            concept_hdc = concept_hdc.to(self.device)
            target_t = torch.from_numpy(target_spectrogram).float().to(self.device)
            
            self.optimizer.zero_grad()
            reconstructed_spectrogram = self.broca(concept_hdc)
            loss = self.criterion(reconstructed_spectrogram, target_t)
            loss.backward()
            self.optimizer.step()
            
            if i % 10 == 0:
                print(f"Iter {i}/{n_samples} | Loss: {loss.item():.6f}")
        
        self._save_model()

    def _save_model(self):
        torch.save(self.broca.state_dict(), "broca_model.pth")
        print("Model saved to broca_model.pth")
        
    def _generate_fake_spectrogram(self):
        """Generate random patterns (stripes/blobs) to mimic spectrogram features."""
        img = np.zeros(OBS_SHAPE, dtype=np.float32)
        freq_idx = np.random.randint(0, 64)
        img[freq_idx:freq_idx+5, :, :] = 1.0
        time_idx = np.random.randint(0, 64)
        img[:, time_idx:time_idx+5, :] += 0.5
        img = np.clip(img, 0, 1)
        return img

def main():
    # Check for Data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'LibriSpeech')
    
    # Optional: Argparse to override
    if not os.path.exists(data_path):
        data_path = None # Fallback to sim
    
    
    trainer = BrocaTrainer(data_dir=data_path)
    # Increase default training steps for better results with real data
    # 500 is learning nothing; 10000+ starts to sound okay.
    trainer.train_loop(n_samples=20000)

if __name__ == "__main__":
    main()
