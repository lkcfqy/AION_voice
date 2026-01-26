import torch
import numpy as np
from src.config import HDC_DIM, LSM_N_NEURONS, SEED

class RandomProjectionAdapter:
    """
    Projects LSM analog activity into HDC binary space using Random Projection.
    serves as Locality Sensitive Hashing (LSH).
    Task 1.2: The Adapter
    """
    def __init__(self, device='cpu'):
        self.input_dim = LSM_N_NEURONS
        self.output_dim = HDC_DIM
        self.device = device
        
        # Initialize Random Projection Matrix
        # Shape: (Output, Input) -> (10000, 1000)
        # We use a fixed seed for reproducibility of the "concept space"
        torch.manual_seed(SEED)
        
        # Standard Normal Distribution for projection weights
        # Why Gaussian? It guarantees LSH property for cosine similarity (SimHash)
        self.projection_matrix = torch.randn(self.output_dim, self.input_dim, device=self.device)
        
        # Freeze weights (no learning here, just projection)
        self.projection_matrix.requires_grad_(False)
        
    def forward(self, lsm_activity):
        """
        Args:
            lsm_activity: (N_neurons,) numpy array or tensor
        Returns:
            hdc_vector: (HDC_DIM,) tensor of {-1, 1}
        """
        # Convert input to tensor if needed
        if isinstance(lsm_activity, np.ndarray):
            x = torch.from_numpy(lsm_activity.copy()).float().to(self.device)
        else:
            x = lsm_activity.float().to(self.device)
            
        # Linear Projection
        # y = Wx
        y = torch.mv(self.projection_matrix, x)
        
        # Binarization (Sign)
        # Sign(y) -> -1 or 1. (0 becomes 0, but usually practically non-zero float)
        # We enforce 0 -> 1 to keep strictly binary
        h = torch.sign(y)
        h[h == 0] = 1.0 
        
        return h

    def batch_forward(self, lsm_batch):
        """
        Args:
            lsm_batch: (Batch, N_neurons)
        Returns:
            hdc_batch: (Batch, HDC_DIM)
        """
        if isinstance(lsm_batch, np.ndarray):
            x = torch.from_numpy(lsm_batch).float().to(self.device)
        else:
            x = lsm_batch.float().to(self.device)
            
        y = torch.mm(x, self.projection_matrix.T)
        h = torch.sign(y)
        h[h == 0] = 1.0
        return h
