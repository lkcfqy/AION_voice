import numpy as np
from src.config import DECAY_RATE, SOCIAL_WEIGHT, SOCIAL_RESTORE_VOICE, SOCIAL_RESTORE_SPOKE

class SocialDrive:
    """
    驱动系统 V2：社交驱动。
    用于管理智能体的互动欲望。
    
    核心变量：
    - social_fulfillment (0.0 - 1.0):
        1.0 = 完全满足（刚刚进行了对话）
        0.0 = 极其孤独（长期沉默）
        
     动态机制：
    - 衰减 (Decay)：随时间减少（孤独感累积）。
    - 恢复 (Restore)：听到声音或获得回应时增加。
    """
    def __init__(self):
        self.social_fulfillment = 1.0 # 初始状态为快乐
        self.decay_rate = DECAY_RATE
        self.loneliness_weight = SOCIAL_WEIGHT # Lambda 参数
        
    def step(self, heard_voice=False, spoke=False):
        """
        更新驱动状态。
        参数:
            heard_voice: bool，如果麦克风检测到语音则为真
            spoke: bool，如果智能体说话则为真
        """
        # 衰减（孤独感袭来）
        self.social_fulfillment -= self.decay_rate
        
        # 恢复逻辑
        if heard_voice:
            # 听到别人的声音是非常令人满足的
            self.social_fulfillment += SOCIAL_RESTORE_VOICE
            
        if spoke:
            # 说话本身会带来一些缓解（表达），
            # 但程度低于听到声音（互动）。
            self.social_fulfillment += SOCIAL_RESTORE_SPOKE
            
        # 限制范围
        self.social_fulfillment = np.clip(self.social_fulfillment, 0.0, 1.0)
        
    @property
    def loneliness(self):
        return 1.0 - self.social_fulfillment
        
    def compute_free_energy(self, surprise):
        """
        计算总自由能。
        F = 惊喜度 (预测误差) + Lambda * 孤独感
        """
        return surprise + self.loneliness_weight * self.loneliness
