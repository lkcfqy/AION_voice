import sys
import os
import time
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.voice_body.environment import AudioEnvironment
from src.lsm import AION_LSM_Network
from src.adapter import RandomProjectionAdapter
from src.voice_body.broca import BrocaNet
from src.drive import SocialDrive
from src.gwt import GlobalWorkspace
from src.hrr import HDCWorldModel
from src.dashboard import AIONDashboard
from src.mhn import ModernHopfieldNetwork
from src.config import OBS_SHAPE

class InteractionAgent:
    def __init__(self, device='cpu'):
        self.device = device
        print("Initializing AION Voice Agent...")
        
        # 1. Body
        # Try to use mic, fallback to sim if failed
        try:
            import sounddevice as sd
            # Check devices
            try:
                sd.query_devices(kind='input')
                self.env = AudioEnvironment(use_microphone=True)
                print("✅ Microphone initialized.")
            except Exception as e:
                print(f"⚠️ Microphone init failed ({e}). Falling back to SIMULATION mode.")
                self.env = AudioEnvironment(use_microphone=False)
        except ImportError:
            print("⚠️ sounddevice not found. Falling back to SIMULATION mode.")
            self.env = AudioEnvironment(use_microphone=False)
            
        # Note: If mic fails, env usually runs in sim mode (silent)
        
        # 2. Perception
        self.lsm = AION_LSM_Network()
        self.adapter = RandomProjectionAdapter()
        
        # 3. Brain (Memory & GWT)
        self.gwt = GlobalWorkspace(device=device)
        self.drive = SocialDrive()
        self.wm = HDCWorldModel(n_actions=1, device=device)
        
        # Load Memory/Associations
        if os.path.exists("association_memory.pt"):
            print("Loading Association Memory...")
            self.wm.M_per_action = torch.load("association_memory.pt")
        else:
            print("⚠️ No association memory found. Agent will be a blank slate.")
            
        # 4. Action (Broca)
        self.broca = BrocaNet().to(device)
        self.broca_trained = False
        if os.path.exists("broca_model.pth"):
            print("Loading BrocaNet Weights...")
            self.broca.load_state_dict(torch.load("broca_model.pth"))
            self.broca.eval()
            self.broca_trained = True
        else:
            print("⚠️ No Broca weights found. Speaking will be random/noisy.")

        # 5. Episodic Memory (MHN)
        print("Initializing Episodic Memory (MHN)...")
        self.memory = ModernHopfieldNetwork(device=device)

        # 6. Dashboard (Visdom)
        try:
            self.dashboard = AIONDashboard()
            print("✅ Dashboard connected.")
        except Exception as e:
            print(f"⚠️ Dashboard init failed ({e}). Visdom server might be down.")
            print("   Run 'python -m visdom.server' to enable visualization.")
            self.dashboard = None

        # State Machine
        self.state = "LISTEN" # LISTEN, PONDER, SPEAK
        self.silence_counter = 0

    def run(self):
        print("\n=== AION Interaction Loop Started ===")
        print("Press Ctrl+C to stop.")
        
        try:
            while True:
                # 1. Listen Phase
                if self.state == "LISTEN":
                    obs, _, _, _, _ = self.env.step(action_spectrogram=None)
                    
                    # Detect if valid sound (simple energy threshold on spectrogram)
                    # Obs is (64, 64, 3) [0, 1]
                    # Silence is usually grey/black? Mel spec [0,1]
                    # If silence, cochlea output is low? Cochlea clips at -80db
                    # Let's check mean activity
                    activity = np.mean(obs)
                    
                    # Fix: Now that noise is gone (activity ~0.02), we can lower threshold back down.
                    if activity > 0.05: # Threshold for "Hearing Something"
                        print(f"👂 Heard sound (Activity: {activity:.2f})")
                        spikes = self.process_hearing(obs)
                        self.state = "PONDER"
                        self.silence_counter = 0
                        self.drive.step(heard_voice=True)

                        # Dashboard Update: Environment & Spikes
                        if self.dashboard:
                            self.dashboard.update_env_view(obs)
                            self.dashboard.update_lsm_raster(spikes)

                        # Memory: Store the concept
                        current_concept = self.gwt.current_sense
                        self.memory.add_memory(current_concept)
                        if self.dashboard:
                            # Update survival with memory count (just as a signal)
                            pass
                    else:
                        # Silence
                        self.silence_counter += 1
                        time.sleep(0.1)
                        
                        # Apply drive decay
                        self.drive.step(heard_voice=False)
                        
                        # If too lonely, speak something
                        if self.drive.loneliness > 0.5 and self.silence_counter > 50:
                            print("😞 Feeling lonely... Initiating speech.")
                            self.state = "SPEAK_INITIATIVE"
                            
                # 2. Ponder Phase (Reactive)
                elif self.state == "PONDER":
                    # We have current_sense in GWT (set by process_hearing)
                    current_concept = self.gwt.current_sense
                    
                    # Query Association
                    print("🤔 Thinking...")
                    
                    # Episodic Retrieval (MHN)
                    # Try to recall similar past experiences
                    energy = self.memory.compute_energy(current_concept)
                    recalled = self.memory.retrieve(current_concept)
                    
                    if self.dashboard:
                        self.dashboard.update_energy(energy)
                        # We can also plot similarity between current and recalled
                        sim = torch.nn.functional.cosine_similarity(current_concept, recalled, dim=-1).item()
                        self.dashboard.update_hdc_similarity(sim)

                    # Predict response (Action 0 = Dialogue)
                    # DEBUG: Switch to "Parrot Mode" (Echo) to verify audio quality
                    # Since Association Model fails to generalize to new voices immediately,
                    # prediction returns noise. Let's repeat the input to prove Broca works.
                    print("🦜 MIMIC MODE: Repeating what I heard...")
                    reply_concept = current_concept
                    
                    # Original logic (commented out for debug)
                    # reply_concept = self.wm.predict(current_concept, 0)
                    
                    # Check confidence/similarity?
                    # For now just trust it
                    self.gwt.update_pred(reply_concept)
                    self.state = "SPEAK"
                    
                # 3. Speak Phase (Response)
                elif self.state == "SPEAK":
                    print("🗣️ Replying...")
                    if not self.broca_trained:
                        print("   (Note: Output is noisy because Broca model is undertrained. Please run scripts/train_broca.py)")
                    
                    concept = self.gwt.current_pred
                    self.speak(concept)
                    self.state = "LISTEN"
                    
                # 4. Speak Phase (Initiative)
                elif self.state == "SPEAK_INITIATIVE":
                    # Generate random topic or greeting?
                    # For now, just generate from a random concept
                    print("🗣️ Initiating Conversation...")
                    fake_concept = torch.randn(10000).to(self.device).sign()
                    fake_concept[fake_concept==0] = 1.0
                    self.speak(fake_concept)
                    self.state = "LISTEN"
                    self.silence_counter = 0

                # Global Dashboard Updates (Every Loop)
                if self.dashboard:
                    # Survival: Free Energy (conceptually) vs Loneliness
                    # We don't have exact FE calc here easily, using random or 0 for now
                    # Or we can use the MHN energy as a proxy for "Surprise" (High Energy = Familiar, Low = Surprise in some formulations, or vice versa)
                    # MHN Energy E = -lse. Higher E = More Match. So Surprise = -E.
                    # But dashboard expects Free Energy. Let's just plot Loneliness for now.
                    self.dashboard.update_survival(free_energy=0.0, hunger=self.drive.loneliness)
                    
        except KeyboardInterrupt:
            print("\nStopped.")

    def process_hearing(self, obs):
        """Standard perception pipeline."""
        spikes = self.lsm.step(obs)
        # Fix for PyTorch/NumPy writable warning: use copy()
        concept = self.adapter.forward(torch.from_numpy(spikes.copy()).float())
        concept = concept.to(self.device)
        self.gwt.update_sense(concept)
        return spikes
        
    def speak(self, concept):
        """Generate and play sound."""
        with torch.no_grad():
            spectrogram = self.broca(concept)
            
        # Convert to numpy for env
        spec_np = spectrogram.cpu().numpy()
        
        # AudioEnv handles Vocoder + Speaker
        self.env.step(action_spectrogram=spec_np)
        self.drive.step(spoke=True)

if __name__ == "__main__":
    agent = InteractionAgent()
    agent.run()
