"""
Multi-Junction Traffic Environment
Complete Gymnasium environment for multi-agent traffic signal control.

Features:
- Real-time SUMO integration via TraCI
- Multi-junction observation and control
- Emergency vehicle priority handling
- Weather effects modeling
- Configurable queue length modes
- Comprehensive reward shaping
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Local imports
from .sumo_connector import SUMOConnector
from .queue_config import QueueLengthConfig
from ..emergency.priority_handler import EmergencyPriorityHandler
from ..weather.weather_model import WeatherModel, WeatherEffects
from ..utils.logger import setup_logger

logger = setup_logger("traffic_env")


@dataclass
class JunctionState:
    """State information for a single junction"""
    junction_id: str
    
    # Queue lengths per approach (normalized 0-1)
    queue_lengths: np.ndarray  # Shape: (4,) for N,S,E,W
    
    # Traffic density per approach
    densities: np.ndarray  # Shape: (4,)
    
    # Average speeds per approach (normalized)
    speeds: np.ndarray  # Shape: (4,)
    
    # Current phase (one-hot encoded)
    current_phase: np.ndarray  # Shape: (n_phases,)
    
    # Time since last phase switch (normalized)
    time_since_switch: float
    
    # Emergency status
    emergency_active: bool = False
    emergency_lane: int = -1
    
    # Weather state
    weather_state: float = 0.0  # 0=clear, 1=rain
    
    # Downstream congestion
    downstream_congestion: np.ndarray = field(default_factory=lambda: np.zeros(4))


@dataclass
class EnvConfig:
    """Environment configuration"""
    # SUMO settings
    sumo_config: str = ""
    sumo_gui: bool = False
    sumo_step_length: float = 1.0
    
    # Junction IDs
    junction_ids: List[str] = field(default_factory=list)
    
    # Observation settings
    n_phases: int = 4
    max_queue_length: float = 150.0
    max_speed: float = 15.0
    
    # Action settings
    min_green_time: float = 10.0
    max_green_time: float = 90.0
    yellow_time: float = 3.0
    all_red_time: float = 2.0
    
    # Reward weights
    reward_waiting_weight: float = -0.1
    reward_queue_weight: float = -0.5
    reward_throughput_weight: float = 1.0
    reward_switch_penalty: float = -10.0
    reward_emergency_weight: float = 100.0
    
    # Episode settings
    max_steps: int = 3600
    warmup_steps: int = 100
    
    # Queue mode
    queue_mode: str = "realistic_bangalore"
    
    # Weather
    enable_weather: bool = True
    rain_probability: float = 0.15


class MultiJunctionTrafficEnv(gym.Env):
    """
    Multi-Agent Traffic Signal Control Environment
    
    Observation Space (per junction):
        - Queue lengths: 4 values (N,S,E,W approaches)
        - Densities: 4 values
        - Speeds: 4 values
        - Current phase: n_phases values (one-hot)
        - Time since switch: 1 value
        - Emergency active: 1 value
        - Weather state: 1 value
        - Downstream congestion: 4 values
        Total: 19 + n_phases per junction
    
    Action Space (per junction):
        Discrete(2): 0=keep current phase, 1=switch to next phase
    
    Reward:
        Combination of waiting time, queue length, throughput,
        switching penalty, and emergency handling bonus/penalty.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the environment.
        
        Args:
            config: Environment configuration
            render_mode: 'human' for SUMO-GUI, None for headless
        """
        super().__init__()
        
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        
        # Update GUI setting based on render mode
        if render_mode == 'human':
            self.config.sumo_gui = True
        
        # Initialize components
        self.sumo: Optional[SUMOConnector] = None
        self.emergency_handler: Optional[EmergencyPriorityHandler] = None
        self.weather_model: Optional[WeatherModel] = None
        self.queue_config = QueueLengthConfig(mode=self.config.queue_mode)
        
        # Junction IDs (will be populated from SUMO or config)
        self.junction_ids = self.config.junction_ids or [
            'silk_board', 'tin_factory', 'hebbal', 'marathahalli'
        ]
        self.n_junctions = len(self.junction_ids)
        
        # Define observation and action spaces
        self._define_spaces()
        
        # State tracking
        self.current_step = 0
        self.episode_count = 0
        self.last_switch_time: Dict[str, int] = {}
        self.cumulative_reward = 0.0
        
        # Metrics for this episode
        self.episode_metrics = {
            'total_waiting_time': 0.0,
            'total_queue_length': 0.0,
            'throughput': 0,
            'phase_switches': 0,
            'emergency_delays': [],
        }
        
        logger.info(f"Environment initialized with {self.n_junctions} junctions")
    
    def _define_spaces(self) -> None:
        """Define observation and action spaces."""
        n_phases = self.config.n_phases
        
        # Observation dimension per junction:
        # 4 queues + 4 densities + 4 speeds + n_phases + 1 time + 1 emergency + 1 weather + 4 downstream
        obs_dim = 4 + 4 + 4 + n_phases + 1 + 1 + 1 + 4
        
        # Multi-agent observation space (Dict)
        self.observation_space = spaces.Dict({
            jid: spaces.Box(
                low=0.0,
                high=1.0,
                shape=(obs_dim,),
                dtype=np.float32
            )
            for jid in self.junction_ids
        })
        
        # Multi-agent action space (Dict)
        self.action_space = spaces.Dict({
            jid: spaces.Discrete(2)  # 0=keep, 1=switch
            for jid in self.junction_ids
        })
        
        # For single-agent wrappers
        self._flat_obs_dim = obs_dim * self.n_junctions
        self._flat_action_dim = 2 ** self.n_junctions
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed
            options: Additional options
        
        Returns:
            Tuple of (observations, info)
        """
        super().reset(seed=seed)
        
        # Close existing SUMO instance
        if self.sumo is not None:
            self.sumo.close()
        
        # Initialize SUMO connection
        self.sumo = SUMOConnector(
            config_file=self.config.sumo_config,
            gui=self.config.sumo_gui,
            step_length=self.config.sumo_step_length
        )
        self.sumo.start()
        
        # Initialize emergency handler
        self.emergency_handler = EmergencyPriorityHandler(
            max_override_duration=120.0,
            detection_radius=200.0
        )
        
        # Initialize weather model
        if self.config.enable_weather:
            self.weather_model = WeatherModel(
                rain_probability=self.config.rain_probability,
                seed=seed
            )
        
        # Reset state tracking
        self.current_step = 0
        self.episode_count += 1
        self.last_switch_time = {jid: 0 for jid in self.junction_ids}
        self.cumulative_reward = 0.0
        
        # Reset metrics
        self.episode_metrics = {
            'total_waiting_time': 0.0,
            'total_queue_length': 0.0,
            'throughput': 0,
            'phase_switches': 0,
            'emergency_delays': [],
        }
        
        # Warmup period
        for _ in range(self.config.warmup_steps):
            self.sumo.simulation_step()
        self.current_step = self.config.warmup_steps
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        logger.debug(f"Episode {self.episode_count} started")
        
        return obs, info
    
    def step(
        self,
        actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            actions: Dict mapping junction_id to action (0=keep, 1=switch)
        
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        # 1. Check for emergency vehicles (override actions)
        emergency_overrides = self._handle_emergencies()
        
        for jid, override_action in emergency_overrides.items():
            if override_action is not None:
                actions[jid] = override_action
        
        # 2. Apply weather effects
        if self.weather_model:
            self.weather_model.step()
            if self.weather_model.is_raining():
                self._apply_weather_effects()
        
        # 3. Apply actions (respecting minimum green time)
        for jid, action in actions.items():
            if jid not in emergency_overrides:  # Don't apply RL if emergency override
                self._apply_action(jid, action)
        
        # 4. Advance simulation
        self.sumo.simulation_step()
        self.current_step += 1
        
        # 5. Get new observation
        obs = self._get_observation()
        
        # 6. Calculate rewards
        rewards = self._calculate_rewards(actions, emergency_overrides)
        
        # 7. Update metrics
        self._update_metrics()
        
        # 8. Check termination
        terminated = self.sumo.is_finished()
        truncated = self.current_step >= self.config.max_steps
        
        # 9. Get info
        info = self._get_info()
        
        return obs, rewards, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get observation for all junctions."""
        observations = {}
        
        for jid in self.junction_ids:
            obs = self._get_junction_observation(jid)
            observations[jid] = obs
        
        return observations
    
    def _get_junction_observation(self, junction_id: str) -> np.ndarray:
        """Get observation vector for a single junction."""
        obs = []
        
        # Get controlled lanes for this junction
        lanes = self.sumo.get_controlled_lanes(junction_id)
        
        # Group lanes by direction (assume 4 directions, 2 lanes each)
        direction_lanes = self._group_lanes_by_direction(lanes)
        
        # 1. Queue lengths (normalized)
        for direction in ['north', 'south', 'east', 'west']:
            dir_lanes = direction_lanes.get(direction, [])
            if dir_lanes:
                queue = sum(self.sumo.get_lane_queue(l) for l in dir_lanes)
                queue_norm = min(queue / self.config.max_queue_length, 1.0)
            else:
                queue_norm = 0.0
            obs.append(queue_norm)
        
        # 2. Densities (vehicles per meter, normalized)
        for direction in ['north', 'south', 'east', 'west']:
            dir_lanes = direction_lanes.get(direction, [])
            if dir_lanes:
                density = sum(self.sumo.get_lane_density(l) for l in dir_lanes) / len(dir_lanes)
                density_norm = min(density * 100, 1.0)  # Assume max 0.01 veh/m
            else:
                density_norm = 0.0
            obs.append(density_norm)
        
        # 3. Average speeds (normalized)
        for direction in ['north', 'south', 'east', 'west']:
            dir_lanes = direction_lanes.get(direction, [])
            if dir_lanes:
                speeds = [self.sumo.get_lane_speed(l) for l in dir_lanes]
                avg_speed = sum(speeds) / len(speeds) if speeds else 0
                speed_norm = min(avg_speed / self.config.max_speed, 1.0)
            else:
                speed_norm = 0.0
            obs.append(speed_norm)
        
        # 4. Current phase (one-hot)
        current_phase = self.sumo.get_traffic_light_phase(junction_id)
        phase_one_hot = np.zeros(self.config.n_phases)
        if 0 <= current_phase < self.config.n_phases:
            phase_one_hot[current_phase] = 1.0
        obs.extend(phase_one_hot)
        
        # 5. Time since last switch (normalized)
        time_since_switch = self.current_step - self.last_switch_time.get(junction_id, 0)
        time_norm = min(time_since_switch / 120.0, 1.0)
        obs.append(time_norm)
        
        # 6. Emergency active
        emergency_active = 1.0 if self.emergency_handler.is_active else 0.0
        obs.append(emergency_active)
        
        # 7. Weather state
        weather_state = 0.0
        if self.weather_model:
            weather_state = 1.0 if self.weather_model.is_raining() else 0.0
        obs.append(weather_state)
        
        # 8. Downstream congestion (from neighbor junctions)
        neighbors = self._get_neighbor_junctions(junction_id)
        for i, neighbor_id in enumerate(neighbors[:4]):  # Max 4 neighbors
            if neighbor_id:
                neighbor_queue = self._get_total_queue(neighbor_id)
                obs.append(min(neighbor_queue / self.config.max_queue_length, 1.0))
            else:
                obs.append(0.0)
        # Pad if fewer than 4 neighbors
        while len(obs) < 19 + self.config.n_phases:
            obs.append(0.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _apply_action(self, junction_id: str, action: int) -> None:
        """Apply action to a junction."""
        if action == 0:
            # Keep current phase
            return
        
        elif action == 1:
            # Check minimum green time
            time_since_switch = self.current_step - self.last_switch_time.get(junction_id, 0)
            if time_since_switch < self.config.min_green_time:
                return  # Can't switch yet
            
            # Switch to next phase
            current_phase = self.sumo.get_traffic_light_phase(junction_id)
            next_phase = (current_phase + 1) % self.config.n_phases
            
            self.sumo.set_traffic_light_phase(junction_id, next_phase)
            self.last_switch_time[junction_id] = self.current_step
            self.episode_metrics['phase_switches'] += 1
    
    def _handle_emergencies(self) -> Dict[str, Optional[int]]:
        """
        Check for emergency vehicles and return action overrides.
        
        Returns:
            Dict mapping junction_id to override action (or None)
        """
        overrides = {}
        
        if not self.sumo:
            return overrides
        
        # Get all vehicles
        vehicles = self.sumo.get_vehicle_list()
        
        for veh_id in vehicles:
            vtype = self.sumo.get_vehicle_type(veh_id)
            
            if vtype == 'ambulance':
                # Find which junction this ambulance is approaching
                veh_pos = self.sumo.get_vehicle_position(veh_id)
                veh_lane = self.sumo.get_vehicle_lane(veh_id)
                
                for jid in self.junction_ids:
                    junction_pos = self.sumo.get_junction_position(jid)
                    distance = np.sqrt(
                        (veh_pos[0] - junction_pos[0])**2 +
                        (veh_pos[1] - junction_pos[1])**2
                    )
                    
                    if distance < self.emergency_handler.detection_radius:
                        # Emergency detected - trigger override
                        self.emergency_handler.detect_emergency({
                            'id': veh_id,
                            'lane': veh_lane,
                            'junction': jid,
                            'lane_index': self._get_lane_index(veh_lane),
                            'speed': self.sumo.get_vehicle_speed(veh_id),
                            'is_waiting': self.sumo.get_vehicle_speed(veh_id) < 0.1,
                        })
                        
                        # Get emergency phase
                        emergency_phase = self.emergency_handler.get_emergency_phase(jid)
                        current_phase = self.sumo.get_traffic_light_phase(jid)
                        
                        if current_phase != emergency_phase:
                            self.sumo.set_traffic_light_phase(jid, emergency_phase)
                            self.sumo.set_traffic_light_duration(jid, 9999)  # Hold green
                        
                        overrides[jid] = None  # Prevent RL override
        
        # Check for cleared emergencies
        current_vehicles = self.sumo.get_vehicle_list()
        self.emergency_handler.check_clearance(
            current_vehicles,
            self.current_step
        )
        
        return overrides
    
    def _apply_weather_effects(self) -> None:
        """Apply weather effects to vehicles."""
        if not self.weather_model or not self.weather_model.is_raining():
            return
        
        effects = self.weather_model.get_effects()
        
        # Apply speed reduction to all vehicles
        for veh_id in self.sumo.get_vehicle_list():
            current_speed = self.sumo.get_vehicle_speed(veh_id)
            max_speed = self.sumo.get_vehicle_max_speed(veh_id)
            
            # Reduce max speed during rain
            new_max_speed = max_speed * effects.speed_factor
            self.sumo.set_vehicle_max_speed(veh_id, new_max_speed)
    
    def _calculate_rewards(
        self,
        actions: Dict[str, int],
        emergency_overrides: Dict[str, Optional[int]]
    ) -> Dict[str, float]:
        """Calculate rewards for each junction."""
        rewards = {}
        
        for jid in self.junction_ids:
            reward = 0.0
            
            # 1. Waiting time penalty
            total_waiting = self._get_total_waiting_time(jid)
            reward += total_waiting * self.config.reward_waiting_weight
            
            # 2. Queue length penalty
            total_queue = self._get_total_queue(jid)
            reward += total_queue * self.config.reward_queue_weight
            
            # 3. Throughput bonus
            throughput = self.sumo.get_junction_throughput(jid)
            reward += throughput * self.config.reward_throughput_weight
            
            # 4. Switching penalty
            if actions.get(jid, 0) == 1:
                reward += self.config.reward_switch_penalty
                
                # Extra penalty during rain
                if self.weather_model and self.weather_model.is_raining():
                    reward += self.config.reward_switch_penalty  # Double penalty
            
            # 5. Emergency handling
            if jid in emergency_overrides:
                if self.emergency_handler.ambulance_waiting:
                    # Penalty for making ambulance wait
                    reward -= 50.0
                if self.emergency_handler.ambulance_just_cleared:
                    # Bonus for quick clearance
                    clearance_time = self.emergency_handler.get_clearance_time()
                    if clearance_time:
                        if clearance_time < 10:
                            reward += 1000.0
                        elif clearance_time < 30:
                            reward += 500.0
                        else:
                            reward -= clearance_time * 5
            
            rewards[jid] = reward
            self.cumulative_reward += reward
        
        return rewards
    
    def _update_metrics(self) -> None:
        """Update episode metrics."""
        total_waiting = sum(
            self._get_total_waiting_time(jid) for jid in self.junction_ids
        )
        total_queue = sum(
            self._get_total_queue(jid) for jid in self.junction_ids
        )
        
        self.episode_metrics['total_waiting_time'] += total_waiting
        self.episode_metrics['total_queue_length'] += total_queue
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info dict."""
        return {
            'step': self.current_step,
            'episode': self.episode_count,
            'cumulative_reward': self.cumulative_reward,
            'metrics': self.episode_metrics.copy(),
            'weather': self.weather_model.get_state_encoding() if self.weather_model else 0,
            'emergency_active': self.emergency_handler.is_active if self.emergency_handler else False,
        }
    
    def _get_total_waiting_time(self, junction_id: str) -> float:
        """Get total waiting time for all controlled lanes."""
        lanes = self.sumo.get_controlled_lanes(junction_id)
        return sum(self.sumo.get_lane_waiting_time(l) for l in lanes)
    
    def _get_total_queue(self, junction_id: str) -> float:
        """Get total queue length for all controlled lanes."""
        lanes = self.sumo.get_controlled_lanes(junction_id)
        return sum(self.sumo.get_lane_queue(l) for l in lanes)
    
    def _group_lanes_by_direction(self, lanes: List[str]) -> Dict[str, List[str]]:
        """Group lanes by cardinal direction."""
        groups = {'north': [], 'south': [], 'east': [], 'west': []}
        
        for lane in lanes:
            lane_lower = lane.lower()
            if '_n_' in lane_lower or 'north' in lane_lower:
                groups['north'].append(lane)
            elif '_s_' in lane_lower or 'south' in lane_lower:
                groups['south'].append(lane)
            elif '_e_' in lane_lower or 'east' in lane_lower:
                groups['east'].append(lane)
            elif '_w_' in lane_lower or 'west' in lane_lower:
                groups['west'].append(lane)
            else:
                # Distribute evenly if direction unknown
                idx = len(lanes) % 4
                dir_names = ['north', 'south', 'east', 'west']
                groups[dir_names[idx]].append(lane)
        
        return groups
    
    def _get_neighbor_junctions(self, junction_id: str) -> List[str]:
        """Get neighboring junction IDs."""
        neighbors_map = {
            'silk_board': ['tin_factory', 'marathahalli', '', ''],
            'tin_factory': ['silk_board', 'hebbal', '', ''],
            'hebbal': ['tin_factory', '', '', ''],
            'marathahalli': ['silk_board', '', '', ''],
        }
        return neighbors_map.get(junction_id, ['', '', '', ''])
    
    def _get_lane_index(self, lane_id: str) -> int:
        """Get lane index from lane ID."""
        try:
            return int(lane_id.split('_')[-1])
        except (ValueError, IndexError):
            return 0
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == 'human':
            # SUMO-GUI handles rendering
            pass
        elif self.render_mode == 'rgb_array':
            # Would need to capture SUMO screenshot
            pass
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        if self.sumo is not None:
            self.sumo.close()
            self.sumo = None
        logger.info("Environment closed")


# ============================================================================
# WRAPPERS FOR SINGLE-AGENT ALGORITHMS
# ============================================================================

class FlattenedTrafficEnv(gym.Wrapper):
    """
    Wrapper that flattens multi-agent obs/action for single-agent algorithms.
    """
    
    def __init__(self, env: MultiJunctionTrafficEnv):
        super().__init__(env)
        
        self.n_junctions = env.n_junctions
        self.junction_ids = env.junction_ids
        
        # Flatten observation space
        obs_dim = env._flat_obs_dim
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Flatten action space
        self.action_space = spaces.Discrete(env._flat_action_dim)
    
    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs_dict), info
    
    def step(self, action):
        # Convert flat action to dict
        actions_dict = self._unflatten_action(action)
        
        obs_dict, rewards_dict, terminated, truncated, info = self.env.step(actions_dict)
        
        # Sum rewards
        total_reward = sum(rewards_dict.values())
        
        return self._flatten_obs(obs_dict), total_reward, terminated, truncated, info
    
    def _flatten_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten observation dict to array."""
        return np.concatenate([
            obs_dict[jid] for jid in self.junction_ids
        ])
    
    def _unflatten_action(self, action: int) -> Dict[str, int]:
        """Convert flat action index to junction action dict."""
        actions = {}
        for i, jid in enumerate(self.junction_ids):
            # Extract bit i from action
            actions[jid] = (action >> i) & 1
        return actions
