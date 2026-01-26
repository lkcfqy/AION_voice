"""
AION Project Global Configuration
"""

# Environment Settings
ENV_ID = "MiniGrid-Empty-8x8-v0"
RENDER_MODE = "rgb_array"
OBS_SHAPE = (64, 64, 3)  # Audio Spectrogram Shape

# PATHS
import os
import pybullet_data
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Assuming assets are in the root directory for now, based on previous loadURDF calls using relative paths.
# If they are just in the current running dir (previous behavior), we should point to PROJECT_ROOT.
ASSET_PATH = PROJECT_ROOT 

# Drone Action Space (6-DOF)
ACTION_NAMES = {
    0: "Hover",
    1: "Forward",
    2: "Rotate_Left",
    3: "Rotate_Right",
    4: "Up",
    5: "Down"
}

# Visdom Settings
VISDOM_SERVER = "http://localhost"
VISDOM_PORT = 8097
VISDOM_ENV = "AION_Dashboard"

# Simulation Settings
SEEDS = 42
SEED = 42 # Legacy support

# City Generation
CITY_GRID_SIZE = 10     # 10x10 city blocks
CITY_BLOCK_SIZE = 2.0   # Each block is 2x2 meters (Drone is ~0.5m)
CITY_DENSITY = 0.3      # 30% of blocks have buildings
CITY_MAX_HEIGHT = 3.0   # Max building height

# LSM (Liquid State Machine) Settings
LSM_N_NEURONS = 400
LSM_SPARSITY = 0.1     # 10% recurrent connectivity
LSM_IN_SPARSITY = 0.1  # 10% input connectivity (to handle 12k inputs)
TARGET_FIRING_RATE = 20 # Hz (Target rate for homeostasis)
PLASTICITY_RATE = 0.005 # Learning rate (More stable)
RATE_TAU = 0.05        # Rate estimation time constant (50ms)
TAU_RC = 0.02          # Membrane time constant (20ms)
TAU_REF = 0.002        # Refractory period (2ms)
DT = 0.001             # Simulation step (1ms)

# HDC Settings
HDC_DIM = 10000        # Hyperdimensional vector size
MHN_BETA = 20.0        # Modern Hopfield Network inverse temperature
MEMORY_THRESHOLD = 0.9 # Similarity threshold for storing new memories (Dynamic Gating)

# Drive Settings
LAMBDA_HUNGER = 1.0    # Balance between surprise and hunger
HUNGER_INC = 0.001     # Hunger increment per step
