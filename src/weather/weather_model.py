"""
Weather Model
Simulates weather conditions and their effects on traffic
"""

import random
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Dict, Any

from ..utils.logger import setup_logger

logger = setup_logger("weather_model")


class WeatherState(IntEnum):
    """Weather state enumeration"""
    CLEAR = 0
    RAIN = 1
    HEAVY_RAIN = 2  # Future expansion
    FOG = 3         # Future expansion


@dataclass
class WeatherEffects:
    """Effects of weather on traffic parameters"""
    speed_factor: float = 1.0          # Multiplier for max speed
    acceleration_factor: float = 1.0    # Multiplier for acceleration
    headway_factor: float = 1.0         # Multiplier for following distance
    min_green_time: float = 30.0        # Minimum green time in seconds
    switching_penalty: float = 0.0      # Penalty for signal switching


# Predefined weather effect configurations
WEATHER_EFFECTS = {
    WeatherState.CLEAR: WeatherEffects(
        speed_factor=1.0,
        acceleration_factor=1.0,
        headway_factor=1.0,
        min_green_time=30.0,
        switching_penalty=0.0
    ),
    WeatherState.RAIN: WeatherEffects(
        speed_factor=0.75,          # 25% slower
        acceleration_factor=0.70,    # 30% slower acceleration
        headway_factor=1.5,          # 50% larger following distance
        min_green_time=45.0,         # Longer greens for safety
        switching_penalty=10.0       # Discourage rapid switching
    ),
    WeatherState.HEAVY_RAIN: WeatherEffects(
        speed_factor=0.60,
        acceleration_factor=0.55,
        headway_factor=2.0,
        min_green_time=60.0,
        switching_penalty=20.0
    ),
    WeatherState.FOG: WeatherEffects(
        speed_factor=0.50,
        acceleration_factor=0.60,
        headway_factor=2.5,
        min_green_time=50.0,
        switching_penalty=15.0
    ),
}


class WeatherModel:
    """
    Weather Simulation Model for Traffic Control
    
    Features:
    - Stochastic weather transitions
    - Configurable rain probability
    - Weather effects on vehicle behavior
    - Integration with reward function
    
    Bangalore Weather Patterns:
    - Monsoon season: June-September (high rain probability)
    - Winter: December-February (occasional fog)
    - Summer: March-May (generally clear)
    """
    
    def __init__(
        self,
        rain_probability: float = 0.15,
        state_duration: int = 900,  # 15 minutes average
        seed: int = 42
    ):
        """
        Initialize Weather Model
        
        Args:
            rain_probability: Probability of rain starting (0-1)
            state_duration: Average duration of weather state in simulation seconds
            seed: Random seed for reproducibility
        """
        self.rain_probability = rain_probability
        self.state_duration = state_duration
        self.seed = seed
        
        # State tracking
        self.current_state = WeatherState.CLEAR
        self.state_start_time = 0
        self.last_check_time = 0
        
        # Statistics
        self.rain_time = 0
        self.clear_time = 0
        self.transitions = 0
        
        # Random generator
        self.rng = random.Random(seed)
    
    def reset(self) -> None:
        """Reset weather model for new episode"""
        # Random initial state
        if self.rng.random() < self.rain_probability * 0.5:
            self.current_state = WeatherState.RAIN
        else:
            self.current_state = WeatherState.CLEAR
        
        self.state_start_time = 0
        self.last_check_time = 0
        self.rain_time = 0
        self.clear_time = 0
        self.transitions = 0
        
        logger.info(f"Weather reset: Initial state = {self.get_state_name()}")
    
    def update(self, simulation_step: int) -> bool:
        """
        Update weather state
        
        Args:
            simulation_step: Current simulation step
        
        Returns:
            True if state changed
        """
        current_time = simulation_step
        
        # Check at regular intervals (every state_duration/10 steps)
        check_interval = max(self.state_duration // 10, 60)
        
        if current_time - self.last_check_time < check_interval:
            return False
        
        self.last_check_time = current_time
        old_state = self.current_state
        
        # Calculate time in current state
        time_in_state = current_time - self.state_start_time
        
        # State transition logic
        if self.current_state == WeatherState.CLEAR:
            # Probability increases over time to create realistic weather patterns
            base_prob = self.rain_probability
            time_factor = min(time_in_state / self.state_duration, 2.0)
            transition_prob = base_prob * time_factor
            
            if self.rng.random() < transition_prob:
                self.current_state = WeatherState.RAIN
                
        elif self.current_state == WeatherState.RAIN:
            # Rain duration typically 30 minutes - 2 hours in Bangalore
            rain_duration_factor = time_in_state / (self.state_duration * 2)
            clear_prob = min(0.5, 0.1 + rain_duration_factor * 0.4)
            
            if self.rng.random() < clear_prob:
                self.current_state = WeatherState.CLEAR
        
        # Track state change
        if self.current_state != old_state:
            self.state_start_time = current_time
            self.transitions += 1
            
            logger.info(
                f"☁️ Weather change: {WeatherState(old_state).name} → "
                f"{WeatherState(self.current_state).name}"
            )
            
            return True
        
        return False
    
    def is_raining(self) -> bool:
        """Check if it's currently raining"""
        return self.current_state in [WeatherState.RAIN, WeatherState.HEAVY_RAIN]
    
    def get_state_name(self) -> str:
        """Get human-readable state name"""
        return WeatherState(self.current_state).name.lower()
    
    def get_effects(self) -> WeatherEffects:
        """Get current weather effects"""
        return WEATHER_EFFECTS.get(self.current_state, WEATHER_EFFECTS[WeatherState.CLEAR])
    
    def get_speed_factor(self) -> float:
        """Get speed factor for current weather"""
        return self.get_effects().speed_factor
    
    def get_acceleration_factor(self) -> float:
        """Get acceleration factor for current weather"""
        return self.get_effects().acceleration_factor
    
    def get_min_green_time(self) -> float:
        """Get minimum green time for current weather"""
        return self.get_effects().min_green_time
    
    def get_switching_penalty(self) -> float:
        """Get signal switching penalty for current weather"""
        return self.get_effects().switching_penalty
    
    def get_state_encoding(self) -> int:
        """Get integer encoding of current state for RL"""
        return int(self.current_state)
    
    def get_state_one_hot(self) -> list:
        """Get one-hot encoding of current state"""
        encoding = [0] * len(WeatherState)
        encoding[self.current_state] = 1
        return encoding
    
    def apply_weather_adjustment(self, base_value: float, adjustment_type: str) -> float:
        """
        Apply weather adjustment to a base value
        
        Args:
            base_value: Original value
            adjustment_type: Type of adjustment ('speed', 'accel', 'headway')
        
        Returns:
            Adjusted value
        """
        effects = self.get_effects()
        
        if adjustment_type == 'speed':
            return base_value * effects.speed_factor
        elif adjustment_type == 'accel':
            return base_value * effects.acceleration_factor
        elif adjustment_type == 'headway':
            return base_value * effects.headway_factor
        else:
            return base_value
    
    def get_reward_multiplier(self) -> float:
        """Get reward multiplier for current weather"""
        if self.is_raining():
            # In rain, we're more lenient with metrics
            return 1.3  # 30% higher penalties
        return 1.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get weather statistics"""
        return {
            'current_state': self.get_state_name(),
            'is_raining': self.is_raining(),
            'transitions': self.transitions,
            'rain_time': self.rain_time,
            'clear_time': self.clear_time,
            'rain_probability': self.rain_probability,
        }


class BangaloreWeatherModel(WeatherModel):
    """
    Bangalore-specific Weather Model
    
    Incorporates Bangalore's weather patterns:
    - Monsoon: June-September (high probability)
    - Pre-monsoon showers: March-May (medium probability)
    - Post-monsoon: October-November (medium probability)
    - Winter: December-February (low probability)
    """
    
    # Monthly rain probabilities for Bangalore
    MONTHLY_RAIN_PROBABILITY = {
        1: 0.05,   # January
        2: 0.05,   # February
        3: 0.10,   # March
        4: 0.20,   # April
        5: 0.30,   # May
        6: 0.50,   # June - Monsoon starts
        7: 0.60,   # July - Peak monsoon
        8: 0.55,   # August
        9: 0.50,   # September
        10: 0.40,  # October
        11: 0.20,  # November
        12: 0.05,  # December
    }
    
    def __init__(
        self,
        month: int = 7,  # Default to July (monsoon)
        seed: int = 42
    ):
        """
        Initialize Bangalore Weather Model
        
        Args:
            month: Month of year (1-12)
            seed: Random seed
        """
        rain_prob = self.MONTHLY_RAIN_PROBABILITY.get(month, 0.15)
        super().__init__(rain_probability=rain_prob, seed=seed)
        
        self.month = month
        logger.info(f"Bangalore weather for month {month}: rain_prob={rain_prob:.2f}")


if __name__ == "__main__":
    # Test weather model
    print("Testing Weather Model...")
    print("=" * 50)
    
    model = WeatherModel(rain_probability=0.3, seed=42)
    
    print("\n1. Initial state:")
    model.reset()
    print(f"   State: {model.get_state_name()}")
    print(f"   Is raining: {model.is_raining()}")
    
    print("\n2. Simulating 1 hour of weather...")
    transitions = 0
    for step in range(3600):
        if model.update(step):
            transitions += 1
            print(f"   Step {step}: Weather changed to {model.get_state_name()}")
    
    print(f"\n   Total transitions: {transitions}")
    
    print("\n3. Weather effects:")
    effects = model.get_effects()
    print(f"   Speed factor: {effects.speed_factor}")
    print(f"   Min green time: {effects.min_green_time}s")
    print(f"   Switching penalty: {effects.switching_penalty}")
    
    print("\n4. Testing Bangalore-specific model:")
    for month in [1, 4, 7, 10]:
        blr_model = BangaloreWeatherModel(month=month)
        print(f"   Month {month}: rain_probability = {blr_model.rain_probability:.2f}")
    
    print("\n" + "=" * 50)
    print("Weather Model test complete!")
