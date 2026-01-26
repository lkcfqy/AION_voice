import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.lsm import AION_LSM_Network
from src.adapter import RandomProjectionAdapter
from src.hrr import HDCWorldModel
from src.voice_body.dataset import AudioDataset
from torch.utils.data import DataLoader
from src.config import HDC_DIM, OBS_SHAPE

def get_real_data_sample(dataset):
    """Get a random sample from dataset."""
    idx = np.random.randint(0, len(dataset))
    return dataset[idx]

def main():
    print("=== AION Associative Learning Training ===")
    print("Task: Learn 'Hello' (Sound A) -> 'Hi' (Sound B)")
    
    # 1. Init Components
    lsm = AION_LSM_Network()
    adapter = RandomProjectionAdapter()
    
    # We use HDCWorldModel as the Associative Memory
    # State: Heard Concept
    # Action: 0 (Dialogue)
    # Next State: Reply Concept
    wm = HDCWorldModel(n_actions=1) # Only 1 action type: Dialogue
    
    # 2. Create Concepts
    # 2. Load Real Data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'LibriSpeech')
    if not os.path.exists(data_path):
        print("❌ Real data not found. Please run download_sample_data.py first.")
        # Fallback to dummy?
        return

    print(f"Loading data from {data_path}...")
    dataset = AudioDataset(data_path)
    
    if len(dataset) < 2:
        print("❌ Not enough data files.")
        return

    print("Selecting two random speech samples for association...")
    # Sample A
    spec_a = get_real_data_sample(dataset).cpu().numpy()
    # Sample B
    spec_b = get_real_data_sample(dataset).cpu().numpy()
    
    print("Sample A and B loaded.")
    
    # Process to HDC
    print("Perceiving...")
    spikes_a = lsm.step(spec_a)
    concept_a = adapter.forward(torch.from_numpy(spikes_a).float())
    
    spikes_b = lsm.step(spec_b) # Note: LSM state changes, context matters? 
    # For robust association, we might want context-independent concepts initially.
    # But LSM is stateful. Let's reset LSM to simulate "fresh hearing".
    lsm.reset()
    spikes_b = lsm.step(spec_b)
    concept_b = adapter.forward(torch.from_numpy(spikes_b).float())
    
    # 3. Train Association
    print("Training Association: A -> B ...")
    # Action 0 = "Reply"
    wm.learn(concept_a, 0, concept_b)
    
    # 4. Save Model
    torch.save(wm.M_per_action, "association_memory.pt")
    print("✅ Association Saved to association_memory.pt")
    
    # 5. Verify In-Memory
    print("Verifying...")
    predicted_b = wm.predict(concept_a, 0)
    
    # Check similarity
    # Cosine sim
    sim = torch.nn.functional.cosine_similarity(predicted_b.unsqueeze(0), concept_b.unsqueeze(0))
    print(f"Recall Similarity (A->B): {sim.item():.4f}")
    
    if sim.item() > 0.8:
        print("SUCCESS: Strong Association Formed.")
    else:
        print("WARNING: Association weak.")

if __name__ == "__main__":
    main()
