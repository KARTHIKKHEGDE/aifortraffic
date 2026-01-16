"""
Mock Traffic Environment
For development and testing without SUMO installation.

This provides a simplified traffic simulation that mimics the behavior
of the real SUMO-based environment for testing agents.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional


class MockTrafficEnv(gym.Env):
    """
    Mock environment for development without SUMO.
    
    Simulates basic traffic dynamics with:
    - Queue buildup and discharge
    - Phase switching effects
    - Simple reward based on queue length
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        n_junctions: int = 4,
        obs_dim: int = 23,
        n_actions: int = 2,
        max_steps: int = 3600
    ):
        super().__init__()
        
        self.n_junctions = n_junctions
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.max_steps = max_steps
        
        # Flatten for single-agent
        self._flat_obs_dim = obs_dim * n_junctions
        self._flat_action_dim = n_actions ** n_junctions
        
        # Flattened spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._flat_obs_dim,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(self._flat_action_dim)
        
        # State
        self.current_step = 0
        self.queues = None
        self.phases = None
        self.time_since_switch = None
        
        # Traffic parameters
        self.arrival_rate = 0.3  # Probability of new vehicle per step
        self.discharge_rate = 0.5  # Probability of vehicle leaving when green
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Initialize queues (4 directions per junction)
        self.queues = np.random.uniform(5, 20, (self.n_junctions, 4))
        
        # Initialize phases (0-3 for each junction)
        self.phases = np.zeros(self.n_junctions, dtype=int)
        
        # Time since last switch
        self.time_since_switch = np.zeros(self.n_junctions)
        
        obs = self._get_observation()
        info = {'step': 0}
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Decode action (binary for each junction)
        actions = []
        for i in range(self.n_junctions):
            actions.append((action >> i) & 1)
        
        # Update phases based on actions
        for i, a in enumerate(actions):
            if a == 1:  # Switch
                self.phases[i] = (self.phases[i] + 1) % 4
                self.time_since_switch[i] = 0
            else:
                self.time_since_switch[i] += 1
        
        # Simulate traffic dynamics
        for i in range(self.n_junctions):
            for d in range(4):
                # Arrivals
                if np.random.random() < self.arrival_rate:
                    self.queues[i, d] += 1
                
                # Departures (based on phase)
                # Phases 0,2 = NS green; Phases 1,3 = EW green
                green_directions = [0, 1] if self.phases[i] in [0, 2] else [2, 3]
                
                if d in green_directions:
                    if np.random.random() < self.discharge_rate:
                        self.queues[i, d] = max(0, self.queues[i, d] - 1)
        
        # Clip queues
        self.queues = np.clip(self.queues, 0, 150)
        
        self.current_step += 1
        
        # Calculate reward
        reward = -np.sum(self.queues) * 0.01  # Negative queue length
        
        # Add switching penalty
        reward -= sum(actions) * 5.0
        
        # Observation
        obs = self._get_observation()
        
        # Done conditions
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Info
        info = {
            'step': self.current_step,
            'metrics': {
                'total_queue_length': np.sum(self.queues),
                'avg_queue': np.mean(self.queues),
            }
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        obs = []
        
        for i in range(self.n_junctions):
            # Queue lengths (normalized)
            obs.extend(self.queues[i] / 150.0)
            
            # Densities (mock)
            obs.extend(self.queues[i] / 200.0)
            
            # Speeds (inverse of queue - higher queue = lower speed)
            speeds = 1.0 - (self.queues[i] / 150.0)
            obs.extend(speeds.clip(0, 1))
            
            # Phase one-hot
            phase_one_hot = [0, 0, 0, 0]
            phase_one_hot[self.phases[i]] = 1
            obs.extend(phase_one_hot)
            
            # Time since switch (normalized)
            obs.append(min(self.time_since_switch[i] / 120.0, 1.0))
            
            # Emergency (mock - always 0)
            obs.append(0.0)
            
            # Weather (mock - always 0)
            obs.append(0.0)
        
        return np.array(obs, dtype=np.float32)
    
    def render(self):
        """Simple text rendering."""
        print(f"\nStep {self.current_step}")
        print("Queues:")
        for i in range(self.n_junctions):
            junction_names = ['Silk Board', 'Tin Factory', 'Hebbal', 'Marathahalli']
            name = junction_names[i] if i < len(junction_names) else f"Junction {i}"
            print(f"  {name}: N={self.queues[i,0]:.0f} S={self.queues[i,1]:.0f} "
                  f"E={self.queues[i,2]:.0f} W={self.queues[i,3]:.0f} "
                  f"Phase={self.phases[i]}")
    
    def close(self):
        pass
