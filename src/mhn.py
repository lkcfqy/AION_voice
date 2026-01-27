import torch
import torch.nn.functional as F
from src.config import MHN_BETA, HDC_DIM, MEMORY_THRESHOLD, DEVICE

class ModernHopfieldNetwork:
    """
    现代 Hopfield 网络 (MHN) (PyTorch GPU 版本)
    """
    def __init__(self, device=DEVICE):
        self.device = device
        self.beta = MHN_BETA
        self.memory_matrix = torch.empty(0, HDC_DIM, device=self.device)
        
    def add_memory(self, pattern):
        if not isinstance(pattern, torch.Tensor):
            pattern = torch.from_numpy(pattern).float().to(self.device)
        else:
            pattern = pattern.float().to(self.device)
            
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)
            
        if self.memory_matrix.shape[0] > 0:
            similarities = torch.mm(pattern, self.memory_matrix.T) / pattern.shape[1]
            max_sim = similarities.max().item()
            if max_sim > MEMORY_THRESHOLD:
                return False
            
        self.memory_matrix = torch.cat([self.memory_matrix, pattern], dim=0)
        return True
        
    def retrieve(self, query):
        if self.memory_matrix.shape[0] == 0: return query
        
        if not isinstance(query, torch.Tensor):
            query = torch.from_numpy(query).float().to(self.device)
        else:
            query = query.float().to(self.device)
            
        is_batch = query.dim() > 1
        if not is_batch: query = query.unsqueeze(0)
            
        similarity = torch.mm(query, self.memory_matrix.T)
        weights = F.softmax(self.beta * similarity, dim=-1)
        reconstruction = torch.mm(weights, self.memory_matrix)
        
        output = torch.sign(reconstruction)
        output[output == 0] = 1.0
        
        if not is_batch: return output.squeeze(0)
        return output
    
    @property
    def memory_count(self):
        return self.memory_matrix.shape[0]
