import nengo
import numpy as np
from src.config import (
    LSM_N_NEURONS, LSM_SPARSITY, LSM_IN_SPARSITY, 
    TARGET_FIRING_RATE, PLASTICITY_RATE, RATE_TAU,
    TAU_RC, TAU_REF, DT, OBS_SHAPE, SEED
)

class AdaptiveLSM:
    """
    Liquid State Machine (LSM) with Homeostatic Plasticity - Base Implementation.
    
    Note: This is the base/prototype class. For production use with dynamic input
    and 3-factor learning, use AION_LSM_Network which extends this class.
    """
    def __init__(self):
        self.dt = DT
        self.n_neurons = LSM_N_NEURONS
        self.input_size = np.prod(OBS_SHAPE)  # 64*64*3 = 12288
        
        # State variables for homeostasis
        # We process 'bias correction' externally in a Node for full control
        # Initialize bias correction to 0
        self.bias_correction = np.zeros(self.n_neurons)
        # Low-pass filtered firing rate estimate
        self.filtered_rates = np.zeros(self.n_neurons)
        self.rate_tau = 1.0  # Slow time constant for rate estimation (1s)

        self.model = nengo.Network(label="AION_LSM", seed=SEED)
        
        with self.model:
            # 1. Input Node
            # Receives the flattened image vector
            self.input_node = nengo.Node(size_in=self.input_size, label="Visual Input")

            # 2. Reservoir (The Liquid)
            # LIF Neurons with standard parameters
            self.reservoir = nengo.Ensemble(
                n_neurons=self.n_neurons,
                dimensions=1, # Dimensions don't matter much for a reservoir, we care about neuron activities
                # We interpret dimensions=1 loosely, we essentially want a pool of neurons.
                # In Nengo, setting dimensions=1 and high radius is a trick, 
                # but better is to treat it as a population interacting via weights.
                # Actually, connecting input to neurons directly is better for LSM.
                neuron_type=nengo.LIF(tau_rc=TAU_RC, tau_ref=TAU_REF),
                # Random initialization of gains and biases is default
                seed=SEED
            )

            # 3. Input Connection (Sparse)
            # Connect input directly to reservoir neurons (bypass encoder/decoder system for raw LSM)
            # Transform shape: (n_neurons, input_size)
            # We create a sparse random matrix
            print(f"Initializing sparse input weights ({self.n_neurons}x{self.input_size}, {LSM_IN_SPARSITY*100}%)...")
            self.input_weights = nengo.dists.Sparse(
                nengo.dists.Gaussian(0, 0.1), 
                sparsity=1.0 - LSM_IN_SPARSITY
            ).sample(self.n_neurons, self.input_size)
            
            nengo.Connection(
                self.input_node, 
                self.reservoir.neurons, 
                transform=self.input_weights,
                synapse=None # Direct current injection
            )

            # 4. Recurrent Connection (Sparse, Fixed for now, or Hebbian later)
            # Fixed random recurrence to create "Liquid" dynamics
            recurrent_weights = nengo.dists.Sparse(
                nengo.dists.Gaussian(0, 0.1),
                sparsity=1.0 - LSM_SPARSITY
            )
            nengo.Connection(
                self.reservoir.neurons, 
                self.reservoir.neurons,
                transform=recurrent_weights,
                synapse=0.01 # 10ms synaptic filter
            )

            # 5. Homeostasis Mechanism (The Adapter)
            # "If firing rate > target, increase threshold (lower bias)"
            # We implement this via a feedback working node
            
            def homeostasis_func(t, x):
                """
                x: Spikes from reservoir (n_neurons,)
                Returns: Bias correction current (n_neurons,)
                """
                spikes = x
                
                # 1. Update rate estimate (Low-pass filter)
                # alpha = dt / tau
                alpha = self.dt / self.rate_tau
                # Instantaneous rate = spikes / dt
                inst_rate = spikes / self.dt
                self.filtered_rates += alpha * (inst_rate - self.filtered_rates)
                
                # 2. Calculate error
                # Error > 0 means firing too fast -> Need to reduce bias (negative correction)
                error = self.filtered_rates - TARGET_FIRING_RATE
                
                # 3. Update bias correction (Integral control)
                # delta_bias = - learning_rate * error * dt
                # We assume excitatory/inhibitory balance is handled here
                self.bias_correction -= PLASTICITY_RATE * error * self.dt
                
                return self.bias_correction

            self.homeostasis_node = nengo.Node(
                homeostasis_func,
                size_in=self.n_neurons,
                size_out=self.n_neurons,
                label="Homeostasis"
            )

            # Loop: Reservoir Spikes -> Node -> Reservoir Input (Bias)
            nengo.Connection(self.reservoir.neurons, self.homeostasis_node, synapse=None)
            nengo.Connection(self.homeostasis_node, self.reservoir.neurons, synapse=0.01)

            # 6. Probes
            self.spike_probe = nengo.Probe(self.reservoir.neurons)
            self.rate_probe = nengo.Probe(self.homeostasis_node, synapse=None) # Records the bias correction, or we can probe rates if we output them
            
        # Build simulator
        self.sim = nengo.Simulator(self.model, dt=self.dt, progress_bar=False)

    def step(self, image_input):
        """
        Run one step of the simulation.
        obs: (64, 64, 3) image
        """
        # Flatten input
        flat_input = image_input.flatten()
        
        # We need to run the simulator for one step
        # Nengo Simulators are usually run for a duration.
        # But we can use sim.step() if we manually feed data?
        # Standard approach: sim.run(time) but that assumes fixed input for that time.
        # We want step-by-step control.
        # We can use a Node to supply input, but updating that Node's value from outside.
        
        # Hack to update input node value dynamically
        # In Nengo 3.0+, Nodes with output=func are called every step.
        # Nodes with output=None can be set? No.
        # Best way: A Node that reads from a class member.
        
        # Re-build input node to read from self.current_input
        pass 
        # (This is tricky in __init__, let's fix it by defining a lambda that reads self.current_input)

        self.current_input = flat_input
        self.sim.step()
        
        # Get latest spikes
        # sim.data[probe] returns all history. This is slow for long runs.
        # For a truly online agent, we probably shouldn't use `nengo.Probe` for history 
        # unless we clear it, but Nengo doesn't easily support clearing history.
        
        # Direct access to signals is better for "production" but harder.
        # For this prototype, we'll access the last step of the probe.
        
        spikes = self.sim.data[self.spike_probe][-1]
        return spikes

    def _setup_input_callback(self):
        """Helper to replace the static input node with a dynamic one"""
        # We need to do this before building the simulator.
        # Ideally, we redesign __init__ to use a function for the input node.
        pass

# Production-ready LSM with dynamic input and online learning support
class AION_LSM_Network:
    """
    Production Liquid State Machine with:
    - Dynamic input callback (lambda-based Node)
    - 3-factor Hebbian learning (dopamine modulation)
    - Sparse weight generation via scipy.sparse
    - Runtime weight modification support
    """
    def __init__(self):
        self.current_input = np.zeros(np.prod(OBS_SHAPE))
        
        # Override the input_node definition in super (conceptually)
        # We'll just copy the logic effectively inside __init__
        
        self.dt = DT
        self.n_neurons = LSM_N_NEURONS
        self.input_size = np.prod(OBS_SHAPE)
        
        self.bias_correction = np.zeros(self.n_neurons)
        self.filtered_rates = np.zeros(self.n_neurons)
        self.rate_tau = RATE_TAU

        self.model = nengo.Network(label="AION_LSM", seed=SEED)
        
        with self.model:
            # Dynamic Input Node
            self.input_node = nengo.Node(
                lambda t: self.current_input, 
                size_out=self.input_size, 
                label="Visual Input"
            )

            # Reservoir
            self.reservoir = nengo.Ensemble(
                n_neurons=self.n_neurons,
                dimensions=1, 
                neuron_type=nengo.LIF(tau_rc=TAU_RC, tau_ref=TAU_REF),
                seed=SEED
            )

            # Input Weights (Sparse)
            # Note: We must ensure consistent RNG if we want reproducibility
            rng = np.random.RandomState(SEED)
            # Create sparse matrix manually to avoid shape issues with Nengo dists sometimes
            # shape (n_neurons, input_size)
            # Generating full 12k*1k matrix is heavy (12M floats). 
            # We construct a scipy sparse matrix or list of indices?
            # Nengo handles sparse matrices in Transforms well if backend supports it.
            # But the reference backend is numpy, usually dense math. 
            # 12M floats = ~48MB, totally fine for RAM.
            
            # Using Nengo's sparse generator
            # Using Nengo's sparse generator
            import scipy.sparse

            # ... (inside __init__)
            # Input Weights (Sparse)
            print("Generating weights...")

            # Generate sparse weights using scipy
            # density = 1.0 - LSM_IN_SPARSITY
            # We want connectivity = LSM_IN_SPARSITY (e.g. 0.1)
            # scipy.sparse.random takes density (0.0-1.0)
            
            # Helper to generate Nengo-compatible sparse matrix
            def generate_sparse_weights(n_rows, n_cols, density, rng):
                # Using scipy.sparse.random
                # We want standard normal distribution for non-zero elements?
                # scipy.sparse.random values are 0-1 uniform.
                # We need to reshape them to Gaussian.
                
                S = scipy.sparse.random(n_rows, n_cols, density=density, format='csr', random_state=rng)
                # Map [0, 1] to Gaussian
                # Or just assign new data
                if S.nnz > 0:
                   S.data = rng.standard_normal(S.nnz) * 0.005 # scale 0.005 (Optimized start)
                return S

            self.input_weights = generate_sparse_weights(self.n_neurons, self.input_size, LSM_IN_SPARSITY, rng)
            
            nengo.Connection(
                self.input_node, 
                self.reservoir.neurons, 
                transform=self.input_weights.toarray(), 
                synapse=None
            )

            # Recurrent Weights (Sparse)
            recurrent_weights = generate_sparse_weights(self.n_neurons, self.n_neurons, LSM_SPARSITY, rng)
            
            self.recurrent_conn = nengo.Connection(
                self.reservoir.neurons, 
                self.reservoir.neurons,
                transform=recurrent_weights.toarray(),
                synapse=0.01
            )

            # Homeostasis
            def homeostasis_func(t, x):
                spikes = x
                alpha = self.dt / self.rate_tau
                inst_rate = spikes / self.dt
                self.filtered_rates += alpha * (inst_rate - self.filtered_rates)
                
                error = self.filtered_rates - TARGET_FIRING_RATE
                self.bias_correction -= PLASTICITY_RATE * error * self.dt
                
                return self.bias_correction

            self.homeostasis_node = nengo.Node(
                homeostasis_func,
                size_in=self.n_neurons,
                size_out=self.n_neurons,
                label="Homeostasis"
            )
            
            nengo.Connection(self.reservoir.neurons, self.homeostasis_node, synapse=None)
            nengo.Connection(self.homeostasis_node, self.reservoir.neurons, synapse=0.01) # Additive current

            # Spikes Probe
            # We enable this to visualize raster
            self.spike_probe = nengo.Probe(self.reservoir.neurons)
            
        print("Building simulator...")
        self.sim = nengo.Simulator(self.model, dt=self.dt, progress_bar=False)
        
        # Locate the weight signal in the simulator
        # This allows us to modify weights at runtime
        self.weight_sig = self.sim.model.sig[self.recurrent_conn]['weights']
        
        # Hebbian Traces
        self.last_spikes = np.zeros(self.n_neurons)

    def step(self, image_input, dopamine=0.0):
        """
        Run one step of the simulation.
        image_input: (Obs Shape) numpy array
        dopamine: Scalar reinforcement signal (-1.0 to 1.0)
                  If D > 0, reinforce correlated activity (LTP)
                  If D < 0, suppress correlated activity (LTD)
        """
        # 1. Update Input
        flat = image_input.flatten()
        if flat.max() > 1.1: 
            flat = flat / 255.0
            
        self.current_input[:] = flat 
        
        # 2. Step Simulator
        self.sim.step()
        
        # 3. Retrieve Spikes
        spikes = self.sim.data[self.spike_probe][-1]
        
        # 4. Apply 3-Factor Learning Rule (Weight Update)
        # dW = eta * D * (Post * Pre)
        # Nengo default weights are (Post, Pre)
        # We use a simple STDP-like rule on rate/spike events?
        # Since these are spikes (0 or 1/dt), product is only 1 when both fire.
        # This is Hebbian coincidence.
        
        if dopamine != 0.0:
            learning_rate = 1e-4
            
            # Identify co-active neurons
            # We use current spikes for Post and Last Spikes for Pre (Causal)
            # or just instantaneous coincidence.
            # Let's use instantaneous for simplicity in this timestep-based model, 
            # or better: Pre=self.last_spikes, Post=spikes
            
            pre = self.last_spikes
            post = spikes
            
            # Outer product -> (N_post, N_pre)
            # Only update if there is activity
            if np.any(pre) and np.any(post):
                dW = learning_rate * dopamine * np.outer(post, pre)
                
                # Apply update to simulator signals
                # Note: This is an internal Nengo access
                # Force writeable if needed
                weights = self.sim.signals[self.weight_sig]
                if not weights.flags.writeable:
                    weights.setflags(write=1)
                
                weights += dW
                
                # Clip weights? Maybe prevent explosion
                # self.sim.signals[self.weight_sig] = np.clip(self.sim.signals[self.weight_sig], -1.0, 1.0)
        
        # Store history
        self.last_spikes = spikes.copy()
        
        return spikes

    def reset(self):
        """Reset the simulator state."""
        self.sim = nengo.Simulator(self.model, dt=self.dt, progress_bar=False)
        # Re-locate signals
        self.weight_sig = self.sim.model.sig[self.recurrent_conn]['weights']
        self.last_spikes = np.zeros(self.n_neurons)

