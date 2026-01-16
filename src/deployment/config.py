"""
Configuration Management System

Provides:
- YAML/JSON configuration loading
- Environment variable support
- Configuration validation
- Hierarchical configuration merging
- Runtime configuration updates
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Union, TypeVar, get_type_hints
from dataclasses import dataclass, field, fields, asdict
from copy import deepcopy

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


T = TypeVar('T')


@dataclass
class NetworkConfig:
    """Configuration for neural network"""
    hidden_layers: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    dropout: float = 0.0
    use_batch_norm: bool = False
    init_method: str = "xavier"


@dataclass
class TrainingConfig:
    """Configuration for training"""
    algorithm: str = "DQN"
    
    # Learning
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 100000
    
    # Training schedule
    total_timesteps: int = 1000000
    warmup_steps: int = 10000
    update_frequency: int = 4
    target_update_frequency: int = 1000
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 100000
    
    # PPO specific
    n_steps: int = 2048
    n_epochs: int = 10
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95
    
    # Checkpointing
    save_frequency: int = 10000
    eval_frequency: int = 5000
    log_frequency: int = 100
    
    # Network
    network: NetworkConfig = field(default_factory=NetworkConfig)


@dataclass
class EnvironmentConfig:
    """Configuration for environment"""
    # SUMO settings
    sumo_config: str = "data/bangalore.sumocfg"
    sumo_gui: bool = False
    sumo_step_length: float = 1.0
    sumo_port: int = 8813
    
    # Junctions
    junction_ids: List[str] = field(default_factory=lambda: [
        "silk_board", "tin_factory", "hebbal", "marathahalli"
    ])
    
    # Observation space
    observation_type: str = "full"  # 'full', 'compact', 'image'
    normalize_observations: bool = True
    
    # Action space
    action_type: str = "discrete"  # 'discrete', 'continuous'
    num_phases: int = 4
    min_green_time: float = 10.0
    max_green_time: float = 60.0
    
    # Episode settings
    max_steps: int = 3600
    warmup_steps: int = 100
    
    # Traffic generation
    vehicle_types: List[str] = field(default_factory=lambda: [
        "car", "bus", "truck", "auto", "motorcycle", "emergency"
    ])
    base_demand: float = 1000.0
    demand_variation: float = 0.3
    
    # Weather
    enable_weather: bool = True
    weather_update_interval: int = 300


@dataclass
class RewardConfig:
    """Configuration for reward shaping"""
    # Component weights
    waiting_time_weight: float = -0.4
    queue_length_weight: float = -0.2
    throughput_weight: float = 0.3
    emergency_weight: float = 0.5
    switch_penalty_weight: float = -0.1
    fairness_weight: float = 0.05
    fuel_consumption_weight: float = -0.02
    emissions_weight: float = -0.02
    coordination_weight: float = 0.1
    
    # Normalization
    normalize_rewards: bool = True
    reward_scale: float = 1.0
    clip_rewards: bool = True
    reward_clip_value: float = 10.0


@dataclass
class DeploymentConfig:
    """Configuration for deployment"""
    model_path: str = "models/best_agent.pt"
    device: str = "cpu"
    batch_size: int = 1
    max_latency_ms: float = 100.0
    
    # Control
    control_interval: float = 5.0
    enable_failsafe: bool = True
    min_green_time: float = 10.0
    max_red_time: float = 120.0
    
    # Monitoring
    enable_monitoring: bool = True
    log_predictions: bool = True
    health_check_interval: float = 60.0
    
    # Hot reload
    enable_hot_reload: bool = True
    reload_check_interval: float = 10.0


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    name: str = "traffic_control_experiment"
    description: str = ""
    seed: int = 42
    
    # Sub-configurations
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    
    # Paths
    output_dir: str = "outputs"
    log_dir: str = "logs"
    model_dir: str = "models"
    data_dir: str = "data"
    
    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "traffic-control"
    wandb_entity: str = ""


class ConfigLoader:
    """
    Loads and manages configuration
    """
    
    def __init__(self):
        self.env_prefix = "TRAFFIC_"
    
    def load_yaml(self, path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not HAS_YAML:
            raise ImportError("PyYAML is required for YAML config files")
        
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_json(self, path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def load(self, path: str) -> Dict[str, Any]:
        """Load configuration from file (auto-detect format)"""
        path = Path(path)
        
        if path.suffix in ['.yaml', '.yml']:
            return self.load_yaml(str(path))
        elif path.suffix == '.json':
            return self.load_json(str(path))
        else:
            # Try JSON first, then YAML
            try:
                return self.load_json(str(path))
            except json.JSONDecodeError:
                if HAS_YAML:
                    return self.load_yaml(str(path))
                raise
    
    def load_from_env(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override config values from environment variables
        
        Environment variables should be named like:
        TRAFFIC_TRAINING_LEARNING_RATE=0.001
        TRAFFIC_ENVIRONMENT_SUMO_GUI=true
        """
        result = deepcopy(config)
        
        for key, value in os.environ.items():
            if not key.startswith(self.env_prefix):
                continue
            
            # Parse key path
            path = key[len(self.env_prefix):].lower().split('_')
            
            # Navigate to nested location
            current = result
            for part in path[:-1]:
                if part in current:
                    current = current[part]
                else:
                    break
            else:
                # Set value with type conversion
                final_key = path[-1]
                if final_key in current:
                    original_value = current[final_key]
                    current[final_key] = self._convert_type(value, type(original_value))
        
        return result
    
    def _convert_type(self, value: str, target_type: Type) -> Any:
        """Convert string value to target type"""
        if target_type == bool:
            return value.lower() in ('true', '1', 'yes')
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == list:
            return json.loads(value)
        else:
            return value
    
    def merge_configs(
        self, 
        base: Dict[str, Any], 
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recursively merge two configs
        
        Args:
            base: Base configuration
            override: Override configuration (takes precedence)
            
        Returns:
            Merged configuration
        """
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def dict_to_dataclass(
        self, 
        data: Dict[str, Any], 
        dataclass_type: Type[T]
    ) -> T:
        """Convert dictionary to dataclass instance"""
        field_types = {f.name: f.type for f in fields(dataclass_type)}
        
        kwargs = {}
        for key, value in data.items():
            if key not in field_types:
                continue
            
            field_type = field_types[key]
            
            # Handle nested dataclasses
            if hasattr(field_type, '__dataclass_fields__'):
                if isinstance(value, dict):
                    value = self.dict_to_dataclass(value, field_type)
            
            kwargs[key] = value
        
        return dataclass_type(**kwargs)
    
    def load_experiment_config(
        self,
        config_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> ExperimentConfig:
        """
        Load complete experiment configuration
        
        Args:
            config_path: Path to config file
            overrides: Additional overrides
            
        Returns:
            ExperimentConfig instance
        """
        # Start with defaults
        config = asdict(ExperimentConfig())
        
        # Load from file if provided
        if config_path:
            file_config = self.load(config_path)
            config = self.merge_configs(config, file_config)
        
        # Apply environment variable overrides
        config = self.load_from_env(config)
        
        # Apply explicit overrides
        if overrides:
            config = self.merge_configs(config, overrides)
        
        # Convert to dataclass
        return self.dict_to_dataclass(config, ExperimentConfig)


class ConfigValidator:
    """
    Validates configuration values
    """
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, config: ExperimentConfig) -> bool:
        """
        Validate experiment configuration
        
        Returns:
            True if valid
        """
        self.errors = []
        self.warnings = []
        
        # Validate training config
        self._validate_training(config.training)
        
        # Validate environment config
        self._validate_environment(config.environment)
        
        # Validate reward config
        self._validate_reward(config.reward)
        
        # Validate paths
        self._validate_paths(config)
        
        return len(self.errors) == 0
    
    def _validate_training(self, config: TrainingConfig):
        """Validate training configuration"""
        if config.learning_rate <= 0:
            self.errors.append("learning_rate must be positive")
        
        if config.learning_rate > 1:
            self.warnings.append("learning_rate > 1 is unusual")
        
        if config.gamma < 0 or config.gamma > 1:
            self.errors.append("gamma must be in [0, 1]")
        
        if config.batch_size <= 0:
            self.errors.append("batch_size must be positive")
        
        if config.algorithm not in ['DQN', 'PPO', 'A2C', 'SAC', 'Q-Learning']:
            self.warnings.append(f"Unknown algorithm: {config.algorithm}")
        
        if config.epsilon_start < config.epsilon_end:
            self.warnings.append("epsilon_start < epsilon_end is unusual")
    
    def _validate_environment(self, config: EnvironmentConfig):
        """Validate environment configuration"""
        if config.max_steps <= 0:
            self.errors.append("max_steps must be positive")
        
        if config.min_green_time <= 0:
            self.errors.append("min_green_time must be positive")
        
        if config.min_green_time > config.max_green_time:
            self.errors.append("min_green_time must be <= max_green_time")
        
        if not config.junction_ids:
            self.errors.append("junction_ids cannot be empty")
        
        sumo_config = Path(config.sumo_config)
        if not sumo_config.exists():
            self.warnings.append(f"SUMO config not found: {config.sumo_config}")
    
    def _validate_reward(self, config: RewardConfig):
        """Validate reward configuration"""
        total_weight = abs(config.waiting_time_weight) + \
                      abs(config.queue_length_weight) + \
                      abs(config.throughput_weight)
        
        if total_weight == 0:
            self.errors.append("At least one reward weight must be non-zero")
    
    def _validate_paths(self, config: ExperimentConfig):
        """Validate path configuration"""
        # Check if directories can be created
        for path_name in ['output_dir', 'log_dir', 'model_dir']:
            path = Path(getattr(config, path_name))
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.errors.append(f"Cannot create {path_name}: {e}")
    
    def get_report(self) -> str:
        """Get validation report"""
        lines = []
        
        if self.errors:
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  - {error}")
        
        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        
        if not self.errors and not self.warnings:
            lines.append("Configuration is valid")
        
        return '\n'.join(lines)


class ConfigManager:
    """
    High-level configuration manager
    """
    
    def __init__(
        self,
        config_dir: str = "configs",
        default_config: str = "default.yaml"
    ):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_config_path = self.config_dir / default_config
        
        self.loader = ConfigLoader()
        self.validator = ConfigValidator()
        
        self._current_config: Optional[ExperimentConfig] = None
    
    def create_default_config(self) -> str:
        """Create default configuration file"""
        config = ExperimentConfig()
        config_dict = asdict(config)
        
        if HAS_YAML:
            path = self.config_dir / "default.yaml"
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        else:
            path = self.config_dir / "default.json"
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        return str(path)
    
    def load(
        self,
        config_name: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> ExperimentConfig:
        """
        Load configuration
        
        Args:
            config_name: Config file name (in config_dir)
            overrides: Additional overrides
            
        Returns:
            ExperimentConfig
        """
        if config_name:
            config_path = self.config_dir / config_name
            
            if not config_path.exists():
                raise FileNotFoundError(f"Config not found: {config_path}")
            
            config = self.loader.load_experiment_config(
                str(config_path),
                overrides
            )
        else:
            config = self.loader.load_experiment_config(
                overrides=overrides
            )
        
        # Validate
        if not self.validator.validate(config):
            print("Configuration validation failed:")
            print(self.validator.get_report())
        elif self.validator.warnings:
            print("Configuration warnings:")
            for warning in self.validator.warnings:
                print(f"  - {warning}")
        
        self._current_config = config
        return config
    
    def save(
        self,
        config: ExperimentConfig,
        name: str,
        format: str = "yaml"
    ) -> str:
        """
        Save configuration to file
        
        Args:
            config: Configuration to save
            name: File name (without extension)
            format: 'yaml' or 'json'
            
        Returns:
            Path to saved file
        """
        config_dict = asdict(config)
        
        if format == "yaml" and HAS_YAML:
            path = self.config_dir / f"{name}.yaml"
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        else:
            path = self.config_dir / f"{name}.json"
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        return str(path)
    
    def list_configs(self) -> List[str]:
        """List available configurations"""
        configs = []
        
        for ext in ['*.yaml', '*.yml', '*.json']:
            configs.extend(self.config_dir.glob(ext))
        
        return [c.name for c in configs]
    
    def get_current(self) -> Optional[ExperimentConfig]:
        """Get currently loaded configuration"""
        return self._current_config


def load_config(
    config_path: Optional[str] = None,
    **overrides
) -> ExperimentConfig:
    """
    Convenience function to load configuration
    
    Args:
        config_path: Path to config file
        **overrides: Override values
        
    Returns:
        ExperimentConfig
    """
    loader = ConfigLoader()
    
    override_dict = {}
    for key, value in overrides.items():
        # Handle nested keys like "training.learning_rate"
        parts = key.split('.')
        current = override_dict
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    
    return loader.load_experiment_config(config_path, override_dict)


if __name__ == '__main__':
    # Demo
    manager = ConfigManager()
    
    # Create default config
    default_path = manager.create_default_config()
    print(f"Created default config: {default_path}")
    
    # List configs
    print(f"Available configs: {manager.list_configs()}")
    
    # Load with overrides
    config = manager.load(overrides={
        'name': 'demo_experiment',
        'training': {
            'learning_rate': 0.0005,
            'algorithm': 'PPO'
        }
    })
    
    print(f"\nLoaded config:")
    print(f"  Name: {config.name}")
    print(f"  Algorithm: {config.training.algorithm}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Junctions: {config.environment.junction_ids}")
    
    # Validate
    validator = ConfigValidator()
    if validator.validate(config):
        print("\nConfiguration is valid!")
    
    print(validator.get_report())
