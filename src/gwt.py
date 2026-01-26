import torch
import numpy as np
from src.config import HDC_DIM

class GlobalWorkspace:
    """
    Global Workspace (GWT).
    Central hub for information exchange and conflict monitoring.
    Task 3.1: The Controller
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.dim = HDC_DIM
        
        # Initialize states (None or Zeros)
        # We start with None to indicate missing info
        self.current_sense = None
        self.current_pred = None
        self.current_goal = None
        
    def _to_tensor(self, vec):
        if vec is None:
            return None
        if isinstance(vec, np.ndarray):
            return torch.from_numpy(vec).float().to(self.device)
        return vec.float().to(self.device)

    def update_sense(self, vector):
        self.current_sense = self._to_tensor(vector)
        
    def update_pred(self, vector):
        self.current_pred = self._to_tensor(vector)
        
    def set_goal(self, vector):
        self.current_goal = self._to_tensor(vector)
        
    def _hamming_dist(self, v1, v2):
        """
        Normalized Hamming Distance.
        Range [0, 1].
        0 = Identical
        1 = Opposite
        0.5 = Orthogonal/Random
        Formula: (1 - mean(v1*v2)) / 2 for {-1, 1} vectors
        """
        if v1 is None or v2 is None:
            return 1.0 # Max surprise if missing info
            
        # Cosine/Dot over Norm is equivalent to mean product for binary vectors
        mean_product = torch.mean(v1 * v2).item()
        
        # Map mean_product [-1, 1] to distance [1, 0]
        # dist = (1 - sim) / 2
        return (1.0 - mean_product) / 2.0

    def compute_surprise(self):
        """
        Calculate Prediction Error (Free Energy proxy).
        Dist(Sense, Pred)
        """
        return self._hamming_dist(self.current_sense, self.current_pred)
        
    def compute_goal_delta(self):
        """
        Calculate distance to Goal (Hunger proxy).
        Dist(Sense, Goal)
        """
        return self._hamming_dist(self.current_sense, self.current_goal)
        
    def get_status(self):
        """Return dict of current metrics."""
        return {
            "surprise": self.compute_surprise(),
            "goal_delta": self.compute_goal_delta(),
            "has_sense": self.current_sense is not None,
            "has_pred": self.current_pred is not None,
            "has_goal": self.current_goal is not None
        }
