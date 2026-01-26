import sys
import os
import numpy as np
import time

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.voice_body.environment import AudioEnvironment

def main():
    print("=== AION Voice Body Test Loop ===")
    print("Testing the full audio chain: Mic -> Cochlea -> Spectrogram -> Vocoder -> Speaker")
    
    # 1. Initialize Environment (Use Mic=True for real test, False for simulation)
    # Note: In headless or remote envs, Mic might fail. 
    # We will try with Mic=False first to test the chain logically.
    try:
        env = AudioEnvironment(use_microphone=False)
        print("✅ Environment initialized (Simulation Mode)")
    except Exception as e:
        print(f"❌ Env Init Failed: {e}")
        return

    # 2. Test Step
    print("\n--- Step 1: Echo Test (Simulation) ---")
    # Create fake spectrogram (Random noise)
    fake_action_spec = np.random.rand(64, 64, 3)
    
    start_time = time.time()
    obs, _, _, _, _ = env.step(action_spectrogram=fake_action_spec)
    dt = time.time() - start_time
    
    print(f"Step Time: {dt:.4f}s")
    print(f"Obs Shape: {obs.shape}")
    print(f"Obs Range: [{obs.min():.2f}, {obs.max():.2f}]")
    
    # Verify Echo (Simulation Mode sends action back as obs)
    if np.allclose(obs, fake_action_spec):
        print("✅ Echo Verified (Output -> Input fed back correctly)")
    else:
        print("⚠️ Echo Mismatch")
        
    # 3. Test Vocoder Reconstruction (requires librosa)
    print("\n--- Step 2: Vocoder Test ---")
    try:
        audio = env.vocoder.inverse(obs)
        print(f"Generated Audio Shape: {audio.shape}")
        if np.max(np.abs(audio)) > 0:
            print("✅ Audio Signal Generated (Non-silent)")
        else:
             print("⚠️ Audio is silent (Expected if input was uniform? No, input was random)")
    except Exception as e:
        print(f"❌ Vocoder Failed: {e}")
        
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()
