"""
Bangalore Traffic Environment
Custom OpenAI Gym environment for traffic signal control
"""

import os
import sys
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import gymnasium as gym
from gymnasium import spaces

from .sumo_connector import SUMOConnector
from .queue_config import QueueLengthConfig, TrafficScenario
from ..emergency.priority_handler import EmergencyPriorityHandler
from ..weather.weather_model import WeatherModel
from ..utils.config import get_project_root, load_config
from ..utils.logger import setup_logger, TrainingLogger

logger = setup_logger("traffic_env")


class BangaloreTrafficEnv(gym.Env):
    """
    Custom Gymnasium Environment for Bangalore Traffic Signal Control
    
    Features:
    - Multi-junction control (Silk Board, Tin Factory, Hebbal, Marathahalli)
    - Emergency vehicle (ambulance) priority with green corridor
    - Weather-aware signal timing (rain adaptation)
    - Configurable queue length system
    - Multi-objective reward function
    
    State Space (per junction):
        - Queue lengths per lane (4 values)
        - Traffic density per lane (4 values)
        - Average speed per lane (4 values)
        - Current signal phase (1 value)
        - Phase duration elapsed (1 value)
        - Emergency vehicle presence (1 value)
        - Emergency vehicle lane (1 value)
        - Weather condition (1 value)
        - Downstream congestion pressure (1 value)
        Total: 18 dimensions per junction
    
    Action Space:
        - 0: Keep current phase
        - 1: Switch to North-South green
        - 2: Switch to East-West green
        - 3: Emergency override (force green for ambulance)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        net_file: str = None,
        route_file: str = None,
        junction_ids: List[str] = None,
        queue_mode: str = "realistic_bangalore",
        enable_emergency: bool = True,
        enable_weather: bool = True,
        gui: bool = False,
        max_steps: int = 3600,
        step_length: float = 1.0,
        seed: int = 42,
        render_mode: str = None,
        **kwargs
    ):
        """
        Initialize Bangalore Traffic Environment
        
        Args:
            net_file: Path to SUMO network file
            route_file: Path to route file
            junction_ids: List of junction IDs to control
            queue_mode: Queue configuration mode
            enable_emergency: Enable emergency vehicle handling
            enable_weather: Enable weather effects
            gui: Use SUMO GUI
            max_steps: Maximum steps per episode
            step_length: Simulation step length in seconds
            seed: Random seed
            render_mode: Render mode ('human' or 'rgb_array')
        """
        super().__init__()
        
        # Find files if not provided
        project_root = get_project_root()
        
        if net_file is None:
            # Find first available network file
            maps_dir = project_root / "maps"
            net_files = list(maps_dir.glob("*.net.xml"))
            if net_files:
                net_file = str(net_files[0])
            else:
                raise FileNotFoundError(f"No network files found in {maps_dir}")
        
        if route_file is None:
            # Find first available route file
            routes_dir = project_root / "routes"
            route_files = list(routes_dir.rglob("*.xml"))
            if route_files:
                route_file = str(route_files[0])
            else:
                raise FileNotFoundError(f"No route files found in {routes_dir}")
        
        # Store configuration
        self.net_file = net_file
        self.route_file = route_file
        self.gui = gui
        self.max_steps = max_steps
        self.step_length = step_length
        self.seed_value = seed
        self.render_mode = render_mode
        
        # Queue configuration
        self.queue_config = QueueLengthConfig(mode=queue_mode)
        
        # Feature flags
        self.enable_emergency = enable_emergency
        self.enable_weather = enable_weather
        
        # Initialize components (lazy initialization)
        self.sumo: Optional[SUMOConnector] = None
        self.junction_ids: List[str] = junction_ids or []
        
        # Emergency handler
        self.emergency_handler = EmergencyPriorityHandler() if enable_emergency else None
        
        # Weather model
        self.weather_model = WeatherModel() if enable_weather else None
        
        # Episode tracking
        self.current_step = 0
        self.episode_count = 0
        self.total_reward = 0.0
        
        # Metrics collection
        self.metrics_history = []
        self.episode_metrics = {
            'waiting_times': [],
            'queue_lengths': [],
            'throughput': 0,
            'emergency_clearance_times': [],
        }
        
        # Phase timing constraints
        self.min_green_time = 30  # seconds
        self.min_green_time_rain = 45  # seconds
        self.yellow_duration = 3  # seconds
        self.last_phase_change_step: Dict[str, int] = {}
        
        # Load reward weights from config
        try:
            env_config = load_config('env_config')
            reward_config = env_config.get('reward', {})
            weights = reward_config.get('weights', {})
            
            self.reward_weights = {
                'waiting_time': weights.get('waiting_time', 0.35),
                'queue_length': weights.get('queue_length', 0.25),
                'throughput': weights.get('throughput', 0.15),
                'stops': weights.get('stops', 0.10),
                'fuel': weights.get('fuel', 0.05),
                'emergency': weights.get('emergency', 0.10),
            }
            
            emergency_config = reward_config.get('emergency', {})
            self.emergency_bonus = emergency_config.get('clearance_bonus', 100.0)
            self.emergency_penalty = emergency_config.get('delay_penalty', 50.0)
            
            weather_config = reward_config.get('weather', {})
            self.rain_penalty_multiplier = weather_config.get('rain_penalty_multiplier', 1.3)
            
        except Exception:
            # Default weights
            self.reward_weights = {
                'waiting_time': 0.35,
                'queue_length': 0.25,
                'throughput': 0.15,
                'stops': 0.10,
                'fuel': 0.05,
                'emergency': 0.10,
            }
            self.emergency_bonus = 100.0
            self.emergency_penalty = 50.0
            self.rain_penalty_multiplier = 1.3
        
        # Define action and observation spaces
        # Will be properly set after first reset
        self.num_junctions = 1  # Default, updated after SUMO starts
        self.lanes_per_junction = 4  # Typical for 4-way intersection
        
        # State dimensions per junction
        self.state_dim_per_junction = 18
        
        # Observation space (will be adjusted based on actual junctions)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim_per_junction,),
            dtype=np.float32
        )
        
        # Action space: 4 discrete actions
        # 0: Keep phase, 1: NS green, 2: EW green, 3: Emergency override
        self.action_space = spaces.Discrete(4)
        
        # Random number generator
        self.np_random = np.random.RandomState(seed)
        
        logger.info(f"BangaloreTrafficEnv initialized")
        logger.info(f"  Network: {net_file}")
        logger.info(f"  Routes: {route_file}")
        logger.info(f"  Queue Mode: {queue_mode}")
        logger.info(f"  Emergency: {enable_emergency}")
        logger.info(f"  Weather: {enable_weather}")
    
    def _init_sumo(self) -> None:
        """Initialize SUMO connection"""
        if self.sumo is not None and self.sumo.is_running:
            self.sumo.close()
        
        self.sumo = SUMOConnector(
            net_file=self.net_file,
            route_file=self.route_file,
            gui=self.gui,
            step_length=self.step_length,
            end=self.max_steps,
            seed=self.seed_value + self.episode_count,
        )
        
        self.sumo.start()
        
        # Get actual junction IDs from simulation
        if not self.junction_ids:
            self.junction_ids = self.sumo.get_tls_ids()[:4]  # Limit to 4 junctions
        
        self.num_junctions = len(self.junction_ids)
        
        # Update observation space for actual number of junctions
        total_state_dim = self.state_dim_per_junction * self.num_junctions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_state_dim,),
            dtype=np.float32
        )
        
        # Initialize phase tracking
        for tls_id in self.junction_ids:
            self.last_phase_change_step[tls_id] = 0
        
        logger.info(f"SUMO initialized with {self.num_junctions} junctions: {self.junction_ids}")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset environment for new episode
        
        Args:
            seed: Random seed
            options: Additional options
        
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.seed_value = seed
            self.np_random = np.random.RandomState(seed)
        
        # Initialize/restart SUMO
        self._init_sumo()
        
        # Reset episode state
        self.current_step = 0
        self.episode_count += 1
        self.total_reward = 0.0
        
        # Reset metrics
        self.episode_metrics = {
            'waiting_times': [],
            'queue_lengths': [],
            'throughput': 0,
            'emergency_clearance_times': [],
        }
        
        # Reset weather (random initial state)
        if self.weather_model:
            self.weather_model.reset()
        
        # Reset emergency handler
        if self.emergency_handler:
            self.emergency_handler.reset()
        
        # Get initial observation
        obs = self._get_observation()
        
        # Info dict
        info = {
            'episode': self.episode_count,
            'junction_ids': self.junction_ids,
        }
        
        return obs, info
    
    def step(
        self,
        action: Union[int, np.ndarray, Dict[str, int]]
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute action and advance simulation
        
        Args:
            action: Action(s) to take
                - int: Single action for first junction
                - np.ndarray: Actions for each junction
                - dict: Mapping of junction_id -> action
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert action to dictionary format
        actions = self._parse_actions(action)
        
        # Check for emergency override
        if self.enable_emergency and self.emergency_handler:
            emergency_info = self._check_emergency()
            if emergency_info:
                actions = self.emergency_handler.override_actions(
                    actions, emergency_info
                )
        
        # Apply actions to traffic lights
        self._apply_actions(actions)
        
        # Advance simulation
        self.sumo.step()
        self.current_step += 1
        
        # Update weather
        if self.weather_model:
            old_state = self.weather_model.current_state
            self.weather_model.update(self.current_step)
            new_state = self.weather_model.current_state
            
            if old_state != new_state:
                self._apply_weather_effects()
        
        # Compute reward
        reward = self._compute_reward()
        self.total_reward += reward
        
        # Get next observation
        obs = self._get_observation()
        
        # Check termination
        terminated = self.sumo.is_simulation_ended()
        truncated = self.current_step >= self.max_steps
        
        # Collect step metrics
        self._collect_step_metrics()
        
        # Build info dict
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _parse_actions(self, action: Union[int, np.ndarray, Dict[str, int]]) -> Dict[str, int]:
        """Parse action input into dictionary format"""
        if isinstance(action, dict):
            return action
        
        if isinstance(action, (int, np.integer)):
            # Single action for first junction
            return {self.junction_ids[0]: int(action)}
        
        if isinstance(action, np.ndarray):
            # Array of actions for each junction
            return {
                jid: int(action[i]) 
                for i, jid in enumerate(self.junction_ids[:len(action)])
            }
        
        raise ValueError(f"Invalid action type: {type(action)}")
    
    def _apply_actions(self, actions: Dict[str, int]) -> None:
        """Apply actions to traffic lights"""
        for junction_id, action in actions.items():
            if junction_id not in self.junction_ids:
                continue
            
            current_phase = self.sumo.get_tls_phase(junction_id)
            steps_since_change = self.current_step - self.last_phase_change_step.get(junction_id, 0)
            
            # Get minimum green time based on weather
            min_green = self._get_min_green_time()
            
            # Action 0: Keep current phase
            if action == 0:
                continue
            
            # Action 3: Emergency override (highest priority)
            elif action == 3:
                if self.emergency_handler and self.emergency_handler.is_active:
                    emergency_phase = self.emergency_handler.get_emergency_phase(junction_id)
                    self.sumo.set_tls_phase(junction_id, emergency_phase)
                    self.last_phase_change_step[junction_id] = self.current_step
                continue
            
            # Actions 1 & 2: Phase changes (with constraints)
            elif action in [1, 2]:
                # Check minimum green time constraint
                if steps_since_change * self.step_length < min_green:
                    continue  # Cannot switch yet
                
                # Determine target phase
                # Phase 0: NS green, Phase 2: EW green (typical)
                target_phase = 0 if action == 1 else 2
                
                if current_phase != target_phase:
                    # Need to transition through yellow
                    num_phases = self.sumo.get_num_phases(junction_id)
                    
                    # Set yellow phase (typically current + 1)
                    yellow_phase = (current_phase + 1) % num_phases
                    self.sumo.set_tls_phase(junction_id, yellow_phase)
                    self.sumo.set_tls_phase_duration(junction_id, self.yellow_duration)
                    
                    self.last_phase_change_step[junction_id] = self.current_step
    
    def _get_min_green_time(self) -> float:
        """Get minimum green time based on weather"""
        if self.weather_model and self.weather_model.is_raining():
            return self.min_green_time_rain
        return self.min_green_time
    
    def _check_emergency(self) -> Optional[Dict[str, Any]]:
        """Check for emergency vehicles at junctions"""
        for junction_id in self.junction_ids:
            emergency = self.sumo.get_emergency_at_junction(junction_id)
            if emergency:
                self.emergency_handler.detect_emergency(emergency)
                return emergency
        return None
    
    def _apply_weather_effects(self) -> None:
        """Apply or remove weather effects based on current state"""
        if self.weather_model.is_raining():
            logger.info("Weather: Rain started - applying effects")
            self.sumo.apply_rain_effects(speed_factor=0.75)
        else:
            logger.info("Weather: Rain stopped - removing effects")
            self.sumo.remove_rain_effects()
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation
        
        Returns:
            State vector as numpy array
        """
        state_vector = []
        
        for junction_id in self.junction_ids:
            junction_state = self._get_junction_state(junction_id)
            state_vector.extend(junction_state)
        
        return np.array(state_vector, dtype=np.float32)
    
    def _get_junction_state(self, junction_id: str) -> List[float]:
        """
        Get state for a single junction
        
        State components (18 dimensions):
        - Queue lengths (4 lanes)
        - Density (4 lanes)
        - Average speeds (4 lanes)
        - Current phase (1)
        - Phase duration elapsed (1)
        - Emergency presence (1)
        - Emergency lane (1)
        - Weather condition (1)
        - Downstream congestion (1)
        """
        lanes = self.sumo.get_tls_lanes(junction_id)[:self.lanes_per_junction]
        
        # Pad if fewer lanes
        while len(lanes) < self.lanes_per_junction:
            lanes.append(lanes[-1] if lanes else "")
        
        # 1. Queue lengths (normalized)
        max_queue = self.queue_config.max_queue
        queues = [
            min(self.sumo.get_lane_queue_length(lane), max_queue) / max_queue
            for lane in lanes
        ]
        
        # 2. Density (already normalized 0-1)
        densities = [
            min(self.sumo.get_lane_density(lane), 0.5) / 0.5
            for lane in lanes
        ]
        
        # 3. Average speeds (normalized by max speed ~20 m/s)
        speeds = [
            self.sumo.get_lane_mean_speed(lane) / 20.0
            for lane in lanes
        ]
        
        # 4. Current signal phase (normalized by num phases)
        num_phases = max(self.sumo.get_num_phases(junction_id), 1)
        current_phase = self.sumo.get_tls_phase(junction_id) / num_phases
        
        # 5. Phase duration elapsed (normalized by max green time ~120s)
        steps_since_change = self.current_step - self.last_phase_change_step.get(junction_id, 0)
        phase_duration = min(steps_since_change * self.step_length, 120) / 120
        
        # 6-7. Emergency vehicle detection
        emergency_present = 0.0
        emergency_lane = -1.0
        
        if self.enable_emergency:
            emergency = self.sumo.get_emergency_at_junction(junction_id)
            if emergency:
                emergency_present = 1.0
                emergency_lane = emergency.get('lane_index', 0) / self.lanes_per_junction
        
        # 8. Weather condition
        weather = 0.0
        if self.weather_model:
            weather = 1.0 if self.weather_model.is_raining() else 0.0
        
        # 9. Downstream congestion pressure
        # Simplified: average queue of downstream lanes
        downstream_pressure = np.mean(queues) if queues else 0.0
        
        # Combine state
        state = (
            queues +                    # 4 values
            densities +                 # 4 values
            speeds +                    # 4 values
            [current_phase] +           # 1 value
            [phase_duration] +          # 1 value
            [emergency_present] +       # 1 value
            [emergency_lane] +          # 1 value
            [weather] +                 # 1 value
            [downstream_pressure]       # 1 value
        )  # Total: 18 values
        
        return state
    
    def _compute_reward(self) -> float:
        """
        Compute multi-objective reward
        
        Reward = 
            - α × avg_waiting_time
            - β × total_queue_length
            + γ × throughput
            - δ × num_stops
            - ε × fuel_consumption
            + ζ × emergency_clearance_bonus
            - η × unsafe_switching_penalty
        """
        # Metric collection
        waiting_times = []
        queue_lengths = []
        
        for veh_id in self.sumo.get_vehicle_ids():
            waiting_times.append(self.sumo.get_vehicle_waiting_time(veh_id))
        
        for junction_id in self.junction_ids:
            metrics = self.sumo.get_junction_metrics(junction_id)
            queue_lengths.append(metrics['queue_length'])
        
        # 1. Average waiting time (negative reward)
        avg_waiting = np.mean(waiting_times) if waiting_times else 0
        waiting_reward = -self.reward_weights['waiting_time'] * (avg_waiting / 60.0)  # Normalize by 60s
        
        # 2. Queue length (negative reward)
        total_queue = sum(queue_lengths)
        queue_reward = -self.reward_weights['queue_length'] * (total_queue / 100.0)  # Normalize by 100
        
        # 3. Throughput (positive reward)
        throughput = self.sumo.get_arrived_count()
        throughput_reward = self.reward_weights['throughput'] * throughput
        
        # 4. Stops (negative reward)
        num_stopped = sum(1 for veh in self.sumo.get_vehicle_ids() 
                         if self.sumo.get_vehicle_speed(veh) < 0.1)
        stops_reward = -self.reward_weights['stops'] * (num_stopped / 50.0)
        
        # 5. Fuel/CO2 (negative reward)
        total_co2 = self.sumo.get_total_co2_emission()
        fuel_reward = -self.reward_weights['fuel'] * (total_co2 / 10000.0)
        
        # 6. Emergency clearance (positive for cleared, negative for waiting)
        emergency_reward = 0.0
        if self.enable_emergency and self.emergency_handler:
            if self.emergency_handler.ambulance_just_cleared:
                emergency_reward = self.emergency_bonus
                clearance_time = self.emergency_handler.get_clearance_time()
                if clearance_time:
                    self.episode_metrics['emergency_clearance_times'].append(clearance_time)
            elif self.emergency_handler.is_active and self.emergency_handler.ambulance_waiting:
                emergency_reward = -self.emergency_penalty * 0.1  # Per-step penalty
        
        # Weather adjustment
        weather_multiplier = 1.0
        if self.weather_model and self.weather_model.is_raining():
            weather_multiplier = self.rain_penalty_multiplier
        
        # Combine rewards
        reward = (
            waiting_reward * weather_multiplier +
            queue_reward * weather_multiplier +
            throughput_reward +
            stops_reward +
            fuel_reward +
            emergency_reward
        )
        
        return float(reward)
    
    def _collect_step_metrics(self) -> None:
        """Collect metrics for this step"""
        waiting_times = [
            self.sumo.get_vehicle_waiting_time(veh)
            for veh in self.sumo.get_vehicle_ids()
        ]
        
        total_queue = sum(
            self.sumo.get_junction_metrics(jid)['queue_length']
            for jid in self.junction_ids
        )
        
        self.episode_metrics['waiting_times'].append(
            np.mean(waiting_times) if waiting_times else 0
        )
        self.episode_metrics['queue_lengths'].append(total_queue)
        self.episode_metrics['throughput'] += self.sumo.get_arrived_count()
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary for current step"""
        info = {
            'step': self.current_step,
            'total_reward': self.total_reward,
            'avg_waiting_time': np.mean(self.episode_metrics['waiting_times'][-100:]),
            'avg_queue_length': np.mean(self.episode_metrics['queue_lengths'][-100:]),
            'throughput': self.episode_metrics['throughput'],
        }
        
        if self.weather_model:
            info['weather'] = 'rain' if self.weather_model.is_raining() else 'clear'
        
        if self.emergency_handler:
            info['emergency_active'] = self.emergency_handler.is_active
            info['emergency_clearances'] = len(self.episode_metrics['emergency_clearance_times'])
        
        return info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment"""
        if self.render_mode == "human":
            # SUMO GUI handles rendering
            pass
        elif self.render_mode == "rgb_array":
            # Would need to capture SUMO screenshot
            pass
        return None
    
    def close(self) -> None:
        """Clean up resources"""
        if self.sumo:
            self.sumo.close()
            self.sumo = None
        logger.info("Environment closed")
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of completed episode"""
        return {
            'episode': self.episode_count,
            'total_reward': self.total_reward,
            'steps': self.current_step,
            'avg_waiting_time': np.mean(self.episode_metrics['waiting_times']),
            'avg_queue_length': np.mean(self.episode_metrics['queue_lengths']),
            'total_throughput': self.episode_metrics['throughput'],
            'emergency_clearances': len(self.episode_metrics['emergency_clearance_times']),
            'avg_emergency_clearance_time': (
                np.mean(self.episode_metrics['emergency_clearance_times'])
                if self.episode_metrics['emergency_clearance_times'] else 0
            ),
        }


def make_env(
    queue_mode: str = "baseline",
    enable_emergency: bool = True,
    enable_weather: bool = True,
    gui: bool = False,
    **kwargs
) -> BangaloreTrafficEnv:
    """
    Factory function to create environment
    
    Args:
        queue_mode: Queue configuration mode
        enable_emergency: Enable emergency handling
        enable_weather: Enable weather effects
        gui: Use SUMO GUI
        **kwargs: Additional arguments
    
    Returns:
        Configured BangaloreTrafficEnv instance
    """
    return BangaloreTrafficEnv(
        queue_mode=queue_mode,
        enable_emergency=enable_emergency,
        enable_weather=enable_weather,
        gui=gui,
        **kwargs
    )


if __name__ == "__main__":
    # Test environment
    print("Testing BangaloreTrafficEnv...")
    
    try:
        env = make_env(
            queue_mode="baseline",
            enable_emergency=False,
            enable_weather=False,
            gui=False
        )
        
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Info: {info}")
        
        # Run a few steps
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: reward={reward:.3f}, terminated={terminated}")
            
            if terminated or truncated:
                break
        
        env.close()
        print("Environment test passed!")
        
    except Exception as e:
        print(f"Environment test failed: {e}")
        import traceback
        traceback.print_exc()
