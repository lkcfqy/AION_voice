import visdom
import numpy as np
import time
from src.config import VISDOM_SERVER, VISDOM_PORT, VISDOM_ENV

class AIONDashboard:
    """
    Real-time visualization dashboard using Visdom.
    Task 0.2: The Dashboard
    """
    def __init__(self):
        print(f"Connecting to Visdom at {VISDOM_SERVER}:{VISDOM_PORT}...")
        self.vis = visdom.Visdom(server=VISDOM_SERVER, port=VISDOM_PORT, env=VISDOM_ENV)
        
        if not self.vis.check_connection():
            raise ConnectionError("Could not connect to Visdom server! Please run 'python -m visdom.server'")

        self._init_plots()

    def _init_plots(self):
        """Initialize the 4 mandatory monitoring panels."""
        # Clear all existing plots in this environment to prevent duplicates
        self.vis.close(env=VISDOM_ENV)

        
        # 1. Environment View (RGB Stream)
        # 1. Environment View (RGB Stream)
        self.win_env = self.vis.image(
            np.zeros((3, 56, 56)),
            win="win_env",
            opts=dict(title="Retina (MiniGrid View)", caption="Raw sensory input")
        )

        # 2. LSM Raster Plot
        self.win_lsm = self.vis.line(
            X=np.array([0]), Y=np.array([0]),
            win="win_lsm",
            opts=dict(
                title="1. LSM Raster Plot (Spikes)",
                xlabel="Time",
                ylabel="Neuron Index",
                markers=True,
                markersize=2
            )
        )

        # 3. HDC Similarity
        self.win_hdc = self.vis.line(
            X=np.array([0]), Y=np.array([0]),
            win="win_hdc",
            opts=dict(
                title="2. HDC Similarity",
                xlabel="Time",
                ylabel="Cosine Similarity",
                ylim=[0, 1]
            )
        )

        # 4. Energy Landscape (Conceptual / Scalar tracking for now)
        self.win_energy = self.vis.line(
            X=np.array([0]), Y=np.array([0]),
            win="win_energy",
            opts=dict(
                title="3. Energy Landscape (MHN)",
                xlabel="Step",
                ylabel="Energy (-LSE)"
            )
        )

        # 5. Survival Curve (Loss + Hunger)
        self.win_survival = self.vis.line(
            X=np.array([[0, 0]]), Y=np.array([[0, 0]]),
            win="win_survival",
            opts=dict(
                title="4. Survival Curve",
                xlabel="Step",
                ylabel="Value",
                legend=["Free Energy (Surprise)", "Hunger"]
            )
        )
        
        self.start_time = time.time()
        self.step_count = 0

    def update_env_view(self, image):
        """
        Update the retina view.
        Args:
            image: numpy array of shape (H, W, 3) or (3, H, W).
                   Visdom expects (3, H, W).
        """
        # Ensure HWC
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
            
        # Upscale for visibility (Nearest Neighbor)
        scale = 6
        image = image.repeat(scale, axis=0).repeat(scale, axis=1)
        
        # HWC -> CHW for Visdom
        image = image.transpose(2, 0, 1)
        
        self.vis.image(image, win=self.win_env, opts=dict(title="Retina (MiniGrid View)"))

    def update_lsm_raster(self, active_neuron_indices, time_step=None):
        """
        Update spike raster plot.
        Args:
            active_neuron_indices: list/array of indices of neurons that fired this step.
        """
        # Simplification for realtime plotting: just scatter points for current step
        # Note: Visdom line update with 'append' is efficient enough for low frequency
        t = self.step_count if time_step is None else time_step
        if len(active_neuron_indices) > 0:
            X = np.full(len(active_neuron_indices), t)
            Y = np.array(active_neuron_indices)
            self.vis.line(X=X, Y=Y, win=self.win_lsm, update='append')

    def update_hdc_similarity(self, similarity):
        self.vis.line(X=np.array([self.step_count]), Y=np.array([similarity]), 
                      win=self.win_hdc, update='append')

    def update_energy(self, energy):
        self.vis.line(X=np.array([self.step_count]), Y=np.array([energy]), 
                      win=self.win_energy, update='append')

    def update_survival(self, free_energy, hunger):
        self.vis.line(
            X=np.array([[self.step_count, self.step_count]]), 
            Y=np.array([[free_energy, hunger]]), 
            win=self.win_survival, 
            update='append'
        )
        self.step_count += 1
