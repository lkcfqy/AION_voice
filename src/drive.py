import numpy as np

class SocialDrive:
    """
    Drive System V2: Social Drive.
    Replaces BiologicalDrive (Drone Battery/Hunger).
    
    Core Variable:
    - social_fulfillment (0.0 - 1.0):
        1.0 = Fully satisfied (Just had a conversation)
        0.0 = Extremely lonely (Long silence)
        
     dynamics:
    - Decay: Decreases over time (Loneliness accumulates).
    - Restore: Increases when hearing voice or getting response.
    """
    def __init__(self, decay_rate=0.001):
        self.social_fulfillment = 1.0 # Start happy
        self.decay_rate = decay_rate
        self.loneliness_weight = 1.0 # Lambda
        
    def step(self, heard_voice=False, spoke=False):
        """
        Update drive state.
        Args:
            heard_voice: bool, true if microphone detected speech
            spoke: bool, true if agent spoke
        """
        # Decay (Loneliness creeps in)
        self.social_fulfillment -= self.decay_rate
        
        # Restore logic
        if heard_voice:
            # Hearing someone is very fulfilling
            self.social_fulfillment += 0.05
            
        if spoke:
            # Speaking itself induces some relief (expression), 
            # but less than hearing (interaction).
            self.social_fulfillment += 0.01
            
        # Clip
        self.social_fulfillment = np.clip(self.social_fulfillment, 0.0, 1.0)
        
    @property
    def loneliness(self):
        return 1.0 - self.social_fulfillment
        
    def compute_free_energy(self, surprise):
        """
        Calculate Total Free Energy.
        F = Surprise (Prediction Error) + Lambda * Loneliness
        """
        return surprise + self.loneliness_weight * self.loneliness
