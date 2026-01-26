import torch
import torch.nn as nn
from src.config import HDC_DIM, OBS_SHAPE

class BrocaNet(nn.Module):
    """
    Broca's Area (Voice Production Network).
    Inverse of LSM (Concept -> Perception).
    
    Input: HDC Vector (10000)
    Output: Spectrogram (64, 64, 3)
    
    Architecture:
    Simple MLP Decoder for Phase 2.
    """
    def __init__(self, input_dim=HDC_DIM, output_shape=OBS_SHAPE):
        super().__init__()
        self.input_dim = input_dim
        self.output_shape = output_shape # (64, 64, 3)
        self.output_dim = output_shape[0] * output_shape[1] * output_shape[2]
        
        # Simple MLP Generator
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.output_dim),
            nn.Sigmoid() # Output [0, 1] for normalized spectrogram
        )
        
    def forward(self, hdc_vector):
        """
        Generate spectrogram from concept.
        Args:
            hdc_vector: (Batch, 10000) or (10000)
        Returns:
            spectrogram: (Batch, 64, 64, 3)
        """
        x = hdc_vector
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        flat_img = self.net(x)
        
        img = flat_img.view(-1, *self.output_shape)
        
        if hdc_vector.dim() == 1:
            return img.squeeze(0)
        return img
        
    def generate_sound(self, hdc_vector, vocoder):
        """
        Helper: Generate audio directly.
        """
        params = list(self.parameters())
        # If running on GPU? 
        device = params[0].device
        
        if not isinstance(hdc_vector, torch.Tensor):
            hdc_vector = torch.tensor(hdc_vector).float().to(device)
            
        with torch.no_grad():
            spec = self.forward(hdc_vector)
            
        # Convert to numpy
        spec_np = spec.cpu().numpy()
        
        # Vocode
        return vocoder.inverse(spec_np)
