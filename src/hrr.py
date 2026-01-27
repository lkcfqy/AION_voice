import torch
import torch.nn.functional as F
from src.config import HDC_DIM, DEVICE

class ResonatorNetwork:
    """
    HDC 谐振器网络 (Resonator Network)
    用于解决叠加态分解问题: S = A * x + B * y + ...
    或者乘法分解: S = A * B * C
    
    这里实现经典的迭代因子分解:
    给定复合向量 S，和码本 {Codebook 1, Codebook 2}，找出 S = c1 * c2。
    """
    def __init__(self, device=DEVICE, dim=HDC_DIM):
        self.device = device
        self.dim = dim
        
    def bind(self, t1, t2):
        """乘法绑定 (XOR in binary, Mul in bipolar)"""
        return torch.mul(t1, t2)
        
    def bundle(self, tensor_list):
        """叠加"""
        stacked = torch.stack(tensor_list, dim=0)
        return torch.sign(torch.sum(stacked, dim=0))

    def factorize(self, compound_vector, codebooks, max_iter=100, threshold=0.99):
        """
        谐振器核心算法: Parsing/Factorization.
        假设 compound = f1 * f2 * ... * fn (来自不同的码本)
        
        参数:
            compound_vector: 合成向量 S (D,)
            codebooks: 码本列表 [C1 (N1, D), C2 (N2, D), ...]
                       我们想找到每个码本中的哪个因子参与了合成。
        返回:
            indices: 每个码本中最匹配的索引列表 [idx1, idx2, ...]
            converged: 是否收敛
        """
        # 初始化猜测: 随机叠加或全叠加
        # 更好的初始化: 随机选择一个作为初始猜测
        # 这里使用全叠加初始化 (所有原子的叠加)
        estimates = []
        for cb in codebooks:
            # 归一化的全叠加: sum / sqrt(N)
            # 或者简单地随机选一个
            est = torch.sum(cb, dim=0)
            est = torch.sign(est) # 双极化 (Bipolar)
            estimates.append(est)
            
        S = compound_vector
        n_factors = len(codebooks)
        
        for k in range(max_iter):
            old_estimates = [e.clone() for e in estimates]
            chunks_changed = 0
            
            # 对每个因子进行更新
            # estimate[i] = S * (product of all other estimates)^-1
            # 在双极系统中，逆运算就是自身 (x^-1 = x)
            
            for i in range(n_factors):
                # 1. 计算 "其他因子" 的乘积
                other_product = torch.ones(self.dim, device=self.device)
                for j in range(n_factors):
                    if i != j:
                        other_product = self.bind(other_product, estimates[j])
                
                # 2. 从 S 中移除 "其他因子" -> 得到对当前因子的猜测
                # guess = S * other_product^-1 = S * other_product
                guess = self.bind(S, other_product)
                
                # 3. 在码本中清理 (Clean-up)
                # 找到当前码本中最接近 guess 的项
                # Sim = guess * CB^T
                sims = torch.mv(codebooks[i], guess)
                best_idx = torch.argmax(sims)
                
                # 更新估计值 (硬判决)
                estimates[i] = codebooks[i][best_idx]
                
                # 检查是否变化
                # 检查余弦相似度 (Cosine Sim)
                change_sim = F.cosine_similarity(estimates[i].unsqueeze(0), old_estimates[i].unsqueeze(0))
                if change_sim < threshold:
                    chunks_changed += 1
            
            if chunks_changed == 0:
                print(f"谐振器在第 {k} 步收敛")
                break
                
        # 提取最终索引
        final_indices = []
        for i, est in enumerate(estimates):
            sims = torch.mv(codebooks[i], est)
            final_indices.append(torch.argmax(sims).item())
            
        return final_indices
