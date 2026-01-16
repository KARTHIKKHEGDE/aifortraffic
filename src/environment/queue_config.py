"""
Queue Length Configuration System
Provides configurable queue density for different simulation modes
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional


class QueueMode(Enum):
    """Queue density modes"""
    BASELINE = "baseline"
    REALISTIC_BANGALORE = "realistic_bangalore"
    CALIBRATED = "calibrated"


@dataclass
class QueueSettings:
    """Settings for a specific queue mode"""
    description: str
    peak_factor: float
    max_queue: int
    vehicle_period: float  # seconds between vehicle insertions
    fringe_factor: float = 5.0


class QueueLengthConfig:
    """
    Configurable queue length system
    
    Supports three modes:
    1. Baseline: Standard SUMO defaults for initial testing
    2. Realistic Bangalore: Scaled to match actual congestion patterns
    3. Calibrated: Future integration with real sensor data
    """
    
    # Predefined mode configurations
    MODE_CONFIGS: Dict[QueueMode, QueueSettings] = {
        QueueMode.BASELINE: QueueSettings(
            description="Standard SUMO defaults for initial testing",
            peak_factor=1.0,
            max_queue=30,
            vehicle_period=3.0,
            fringe_factor=5.0
        ),
        QueueMode.REALISTIC_BANGALORE: QueueSettings(
            description="Scaled to Bangalore traffic patterns - Silk Board reality",
            peak_factor=2.5,
            max_queue=100,
            vehicle_period=1.2,
            fringe_factor=10.0
        ),
        QueueMode.CALIBRATED: QueueSettings(
            description="Future calibration from real sensor data",
            peak_factor=2.0,  # Will be updated from real data
            max_queue=80,     # Will be updated from real data
            vehicle_period=1.5,
            fringe_factor=8.0
        ),
    }
    
    # Junction-specific multipliers (Bangalore hotspots)
    JUNCTION_MULTIPLIERS: Dict[str, float] = {
        'silk_board': 2.8,      # Most congested
        'marathahalli': 2.2,    # IT corridor
        'tin_factory': 1.8,     # Old Madras Road
        'hebbal': 1.5,          # Flyover reduces ground congestion
    }
    
    def __init__(self, mode: str = "realistic_bangalore"):
        """
        Initialize queue configuration
        
        Args:
            mode: Queue mode name ('baseline', 'realistic_bangalore', 'calibrated')
        """
        try:
            self.mode = QueueMode(mode)
        except ValueError:
            print(f"Unknown mode '{mode}', defaulting to realistic_bangalore")
            self.mode = QueueMode.REALISTIC_BANGALORE
        
        self.settings = self.MODE_CONFIGS[self.mode]
        self._junction_overrides: Dict[str, Dict[str, float]] = {}
        
    @property
    def peak_factor(self) -> float:
        """Get peak traffic factor"""
        return self.settings.peak_factor
    
    @property
    def max_queue(self) -> int:
        """Get maximum expected queue length"""
        return self.settings.max_queue
    
    @property
    def vehicle_period(self) -> float:
        """Get time between vehicle insertions"""
        return self.settings.vehicle_period
    
    @property
    def fringe_factor(self) -> float:
        """Get fringe network factor for trip generation"""
        return self.settings.fringe_factor
    
    def set_mode(self, mode: str) -> None:
        """
        Change queue mode
        
        Args:
            mode: New mode name
        """
        try:
            self.mode = QueueMode(mode)
            self.settings = self.MODE_CONFIGS[self.mode]
        except ValueError:
            raise ValueError(f"Unknown mode: {mode}. Valid modes: {[m.value for m in QueueMode]}")
    
    def set_junction_override(
        self,
        junction_id: str,
        peak_multiplier: Optional[float] = None,
        vehicle_period: Optional[float] = None
    ) -> None:
        """
        Set junction-specific overrides
        
        Args:
            junction_id: Junction identifier
            peak_multiplier: Override peak multiplier
            vehicle_period: Override vehicle insertion period
        """
        if junction_id not in self._junction_overrides:
            self._junction_overrides[junction_id] = {}
        
        if peak_multiplier is not None:
            self._junction_overrides[junction_id]['peak_multiplier'] = peak_multiplier
        
        if vehicle_period is not None:
            self._junction_overrides[junction_id]['vehicle_period'] = vehicle_period
    
    def get_junction_settings(self, junction_id: str) -> Dict[str, Any]:
        """
        Get settings for a specific junction
        
        Args:
            junction_id: Junction identifier
        
        Returns:
            Dictionary with junction-specific settings
        """
        # Start with base settings
        settings = {
            'peak_factor': self.settings.peak_factor,
            'max_queue': self.settings.max_queue,
            'vehicle_period': self.settings.vehicle_period,
        }
        
        # Apply junction-specific multiplier
        if junction_id in self.JUNCTION_MULTIPLIERS:
            multiplier = self.JUNCTION_MULTIPLIERS[junction_id]
            settings['peak_factor'] *= multiplier
            settings['max_queue'] = int(settings['max_queue'] * multiplier)
            settings['vehicle_period'] /= multiplier
        
        # Apply custom overrides
        if junction_id in self._junction_overrides:
            overrides = self._junction_overrides[junction_id]
            
            if 'peak_multiplier' in overrides:
                settings['peak_factor'] = overrides['peak_multiplier']
            
            if 'vehicle_period' in overrides:
                settings['vehicle_period'] = overrides['vehicle_period']
        
        return settings
    
    def get_trip_generation_params(self, junction_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get parameters for traffic generation
        
        Args:
            junction_id: Optional junction for specific settings
        
        Returns:
            Dictionary with trip generation parameters
        """
        if junction_id:
            settings = self.get_junction_settings(junction_id)
        else:
            settings = {
                'peak_factor': self.settings.peak_factor,
                'vehicle_period': self.settings.vehicle_period,
            }
        
        return {
            'period': settings['vehicle_period'],
            'fringe_factor': self.settings.fringe_factor,
            'binomial': 5 if self.mode == QueueMode.REALISTIC_BANGALORE else 1,
        }
    
    def describe(self) -> str:
        """Get human-readable description of current configuration"""
        return (
            f"Queue Mode: {self.mode.value}\n"
            f"Description: {self.settings.description}\n"
            f"Peak Factor: {self.settings.peak_factor}x\n"
            f"Max Queue: {self.settings.max_queue} vehicles\n"
            f"Vehicle Period: {self.settings.vehicle_period}s\n"
            f"Fringe Factor: {self.settings.fringe_factor}\n"
        )
    
    @classmethod
    def calibrate_from_data(
        cls,
        junction_id: str,
        observed_queue_lengths: list,
        observed_periods: list
    ) -> 'QueueLengthConfig':
        """
        Create calibrated config from real-world observations
        
        Args:
            junction_id: Junction being calibrated
            observed_queue_lengths: List of observed queue lengths
            observed_periods: List of observed vehicle arrival periods
        
        Returns:
            Calibrated QueueLengthConfig instance
        """
        import numpy as np
        
        config = cls(mode='calibrated')
        
        # Calculate statistics from observations
        avg_queue = np.mean(observed_queue_lengths)
        max_queue = np.max(observed_queue_lengths)
        avg_period = np.mean(observed_periods)
        
        # Update calibrated settings
        config.settings = QueueSettings(
            description=f"Calibrated from real data for {junction_id}",
            peak_factor=max_queue / 30,  # Relative to baseline max
            max_queue=int(max_queue * 1.2),  # Allow some headroom
            vehicle_period=avg_period,
            fringe_factor=8.0
        )
        
        return config


class TrafficScenario:
    """
    Traffic scenario configuration for different time periods
    """
    
    SCENARIOS = {
        'morning_peak': {
            'start_hour': 8,
            'end_hour': 10,
            'demand_multiplier': 2.5,
            'description': 'Morning rush hour - inbound traffic'
        },
        'evening_peak': {
            'start_hour': 18,
            'end_hour': 20,
            'demand_multiplier': 2.8,
            'description': 'Evening rush hour - outbound traffic'
        },
        'off_peak': {
            'start_hour': 10,
            'end_hour': 16,
            'demand_multiplier': 0.8,
            'description': 'Mid-day off-peak hours'
        },
        'night': {
            'start_hour': 22,
            'end_hour': 6,
            'demand_multiplier': 0.3,
            'description': 'Night hours - low traffic'
        },
        'weekend': {
            'demand_multiplier': 0.6,
            'description': 'Weekend traffic - shopping/leisure'
        },
        'emergency_test': {
            'demand_multiplier': 1.0,
            'emergency_probability': 0.1,
            'description': 'High emergency vehicle frequency for testing'
        }
    }
    
    def __init__(self, scenario_name: str = 'morning_peak'):
        """
        Initialize traffic scenario
        
        Args:
            scenario_name: Name of the scenario
        """
        if scenario_name not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        self.name = scenario_name
        self.config = self.SCENARIOS[scenario_name]
    
    @property
    def demand_multiplier(self) -> float:
        """Get demand multiplier for this scenario"""
        return self.config.get('demand_multiplier', 1.0)
    
    @property
    def emergency_probability(self) -> float:
        """Get emergency vehicle probability"""
        return self.config.get('emergency_probability', 0.01)
    
    def apply_to_queue_config(self, queue_config: QueueLengthConfig) -> None:
        """
        Apply scenario adjustments to queue configuration
        
        Args:
            queue_config: Queue configuration to modify
        """
        # Adjust vehicle period based on demand multiplier
        adjusted_period = queue_config.vehicle_period / self.demand_multiplier
        
        # This doesn't modify the original config permanently
        # It's used for trip generation parameters
        pass
    
    def get_adjusted_period(self, base_period: float) -> float:
        """Get adjusted vehicle period for this scenario"""
        return base_period / self.demand_multiplier


if __name__ == "__main__":
    # Test queue configuration
    print("Testing Queue Length Configuration...")
    print("=" * 50)
    
    # Test baseline mode
    config = QueueLengthConfig(mode='baseline')
    print("\nBaseline Mode:")
    print(config.describe())
    
    # Test realistic mode
    config.set_mode('realistic_bangalore')
    print("\nRealistic Bangalore Mode:")
    print(config.describe())
    
    # Test junction-specific settings
    print("\nJunction-specific settings (Silk Board):")
    silk_board_settings = config.get_junction_settings('silk_board')
    for key, value in silk_board_settings.items():
        print(f"  {key}: {value}")
    
    # Test scenario
    print("\n" + "=" * 50)
    print("\nTesting Traffic Scenarios...")
    
    scenario = TrafficScenario('morning_peak')
    print(f"\nScenario: {scenario.name}")
    print(f"Demand multiplier: {scenario.demand_multiplier}")
    print(f"Emergency probability: {scenario.emergency_probability}")
