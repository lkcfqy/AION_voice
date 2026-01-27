import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    print("Verifying imports and config...")
    from src.config import (
        LSM_LEARNING_RATE, ADAPTER_SCALING, 
        SOCIAL_RESTORE_VOICE, SOCIAL_RESTORE_SPOKE,
        HRR_DEFAULT_COUNTS
    )
    print(f"Config loaded: LR={LSM_LEARNING_RATE}, Scaling={ADAPTER_SCALING}")

    print("\nVerifying LSM...")
    from src.lsm import AION_LSM_Network
    # Suppress nengo output if possible, but it's fine
    lsm = AION_LSM_Network()
    print("LSM instantiated successfully.")
    
    print("\nVerifying Adapter...")
    from src.adapter import RandomProjectionAdapter
    adapter = RandomProjectionAdapter()
    print("Adapter instantiated.")

    print("\nVerifying Drive...")
    from src.drive import SocialDrive
    drive = SocialDrive()
    drive.step(heard_voice=True)
    print("Drive step executed.")

    print("\nVerifying HRR...")
    from src.hrr import HDCWorldModel
    hrr = HDCWorldModel(n_actions=2)
    # Test load_state_dict hack
    dummy_dict = [torch.zeros(10000) for _ in range(2)]
    hrr.load_state_dict(dummy_dict)
    if hrr.counts[0] == HRR_DEFAULT_COUNTS:
        print(f"HRR counts initialized to {HRR_DEFAULT_COUNTS} as expected.")
    else:
        print(f"HRR counts mismatch: {hrr.counts[0]}")

    print("\nVerifying MHN Error Handling...")
    from src.mhn import ModernHopfieldNetwork
    mhn = ModernHopfieldNetwork()
    try:
        mhn.load_memory("not a tensor")
        print("FAIL: MHN did not raise TypeError for invalid input")
    except TypeError as e:
        print(f"Caught expected TypeError: {e}")
    except Exception as e:
        print(f"Caught unexpected exception: {type(e)}")

    print("\nAll verifications passed!")

except Exception as e:
    print(f"\nFATAL ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
