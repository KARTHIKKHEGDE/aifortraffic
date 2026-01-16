"""
Configuration Management Module
Handles loading and managing YAML configuration files
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


def get_project_root() -> Path:
    """Get the project root directory"""
    current = Path(__file__).resolve()
    # Navigate up from src/utils/config.py to project root
    return current.parent.parent.parent


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file
    
    Args:
        config_name: Name of config file (without .yaml extension)
                    e.g., 'env_config', 'training_config', 'junctions'
    
    Returns:
        Dictionary containing configuration
    """
    config_path = get_project_root() / "configs" / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def load_all_configs() -> Dict[str, Dict[str, Any]]:
    """Load all configuration files"""
    configs = {}
    config_dir = get_project_root() / "configs"
    
    for config_file in config_dir.glob("*.yaml"):
        config_name = config_file.stem
        configs[config_name] = load_config(config_name)
    
    return configs


@dataclass
class SimulationConfig:
    """Simulation configuration dataclass"""
    step_length: float = 1.0
    episode_duration: int = 3600
    max_steps: int = 3600
    seed: int = 42
    gui: bool = False
    delay: int = 0


@dataclass
class RewardConfig:
    """Reward function configuration"""
    waiting_time_weight: float = 0.35
    queue_length_weight: float = 0.25
    throughput_weight: float = 0.15
    stops_weight: float = 0.10
    fuel_weight: float = 0.05
    emergency_weight: float = 0.10
    emergency_clearance_bonus: float = 100.0
    emergency_delay_penalty: float = 50.0
    rain_penalty_multiplier: float = 1.3
    unsafe_switch_penalty: float = 20.0


@dataclass  
class JunctionConfig:
    """Single junction configuration"""
    id: str
    name: str
    latitude: float
    longitude: float
    type: str = "signalized"
    geometry: str = "4-way"
    lanes_per_approach: int = 4
    phases: int = 4
    priority: int = 1


@dataclass
class EnvironmentConfig:
    """Complete environment configuration"""
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    junctions: list = field(default_factory=list)
    queue_mode: str = "realistic_bangalore"
    weather_enabled: bool = True
    emergency_enabled: bool = True
    
    @classmethod
    def from_yaml(cls) -> 'EnvironmentConfig':
        """Load configuration from YAML files"""
        env_config = load_config('env_config')
        junctions_config = load_config('junctions')
        
        # Parse simulation config
        sim_dict = env_config.get('simulation', {})
        simulation = SimulationConfig(
            step_length=sim_dict.get('step_length', 1.0),
            episode_duration=sim_dict.get('episode_duration', 3600),
            max_steps=sim_dict.get('max_steps', 3600),
            seed=sim_dict.get('seed', 42),
            gui=sim_dict.get('gui', False),
            delay=sim_dict.get('delay', 0)
        )
        
        # Parse reward config
        reward_dict = env_config.get('reward', {})
        weights = reward_dict.get('weights', {})
        emergency = reward_dict.get('emergency', {})
        weather = reward_dict.get('weather', {})
        safety = reward_dict.get('safety', {})
        
        reward = RewardConfig(
            waiting_time_weight=weights.get('waiting_time', 0.35),
            queue_length_weight=weights.get('queue_length', 0.25),
            throughput_weight=weights.get('throughput', 0.15),
            stops_weight=weights.get('stops', 0.10),
            fuel_weight=weights.get('fuel', 0.05),
            emergency_weight=weights.get('emergency', 0.10),
            emergency_clearance_bonus=emergency.get('clearance_bonus', 100.0),
            emergency_delay_penalty=emergency.get('delay_penalty', 50.0),
            rain_penalty_multiplier=weather.get('rain_penalty_multiplier', 1.3),
            unsafe_switch_penalty=safety.get('unsafe_switch_penalty', 20.0)
        )
        
        # Parse junction configs
        junctions = []
        for junction_id in ['silk_board', 'tin_factory', 'hebbal', 'marathahalli']:
            if junction_id in junctions_config:
                j = junctions_config[junction_id]
                junctions.append(JunctionConfig(
                    id=j.get('id', junction_id),
                    name=j.get('name', junction_id),
                    latitude=j.get('latitude', 0.0),
                    longitude=j.get('longitude', 0.0),
                    type=j.get('type', 'signalized'),
                    geometry=j.get('geometry', '4-way'),
                ))
        
        # Queue configuration
        queue_config = env_config.get('queue', {})
        queue_mode = queue_config.get('mode', 'realistic_bangalore')
        
        return cls(
            simulation=simulation,
            reward=reward,
            junctions=junctions,
            queue_mode=queue_mode,
            weather_enabled=True,
            emergency_enabled=True
        )


class ConfigManager:
    """
    Centralized configuration manager
    Provides easy access to all configuration parameters
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._configs = {}
        self._load_all_configs()
        self._initialized = True
    
    def _load_all_configs(self):
        """Load all configuration files"""
        try:
            self._configs['env'] = load_config('env_config')
            self._configs['training'] = load_config('training_config')
            self._configs['junctions'] = load_config('junctions')
        except FileNotFoundError as e:
            print(f"Warning: Some config files not found: {e}")
    
    @property
    def env(self) -> Dict[str, Any]:
        """Get environment configuration"""
        return self._configs.get('env', {})
    
    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self._configs.get('training', {})
    
    @property
    def junctions(self) -> Dict[str, Any]:
        """Get junctions configuration"""
        return self._configs.get('junctions', {})
    
    def get(self, config_name: str, key: str, default: Any = None) -> Any:
        """Get a specific configuration value"""
        config = self._configs.get(config_name, {})
        keys = key.split('.')
        
        for k in keys:
            if isinstance(config, dict):
                config = config.get(k, default)
            else:
                return default
        
        return config
    
    def get_junction_ids(self) -> list:
        """Get list of controlled junction IDs"""
        return ['silk_board', 'tin_factory', 'hebbal', 'marathahalli']
    
    def get_junction_config(self, junction_id: str) -> Dict[str, Any]:
        """Get configuration for a specific junction"""
        return self.junctions.get(junction_id, {})
    
    def get_osm_bbox(self) -> list:
        """Get OSM bounding box for download"""
        osm_config = self.junctions.get('osm', {})
        return osm_config.get('bounding_box', [77.55, 12.88, 77.72, 13.06])


# Global config manager instance
config_manager = ConfigManager()


if __name__ == "__main__":
    # Test configuration loading
    print("Testing configuration loading...")
    
    try:
        env_config = load_config('env_config')
        print(f"Environment config loaded: {list(env_config.keys())}")
        
        training_config = load_config('training_config')
        print(f"Training config loaded: {list(training_config.keys())}")
        
        junctions_config = load_config('junctions')
        print(f"Junctions config loaded: {list(junctions_config.keys())}")
        
        # Test ConfigManager
        cm = ConfigManager()
        print(f"Junction IDs: {cm.get_junction_ids()}")
        print(f"OSM BBox: {cm.get_osm_bbox()}")
        
        print("\nConfiguration loading successful!")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
