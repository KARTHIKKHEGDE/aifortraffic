"""
Advanced Reward Shaping for Traffic Signal Control

Multi-objective reward function with:
- Waiting time optimization
- Queue length reduction
- Throughput maximization
- Emergency vehicle priority
- Fairness across approaches
- Environmental impact (fuel/emissions)
- Multi-junction coordination
"""

from collections import deque
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RewardWeights:
    """Configurable weights for reward components"""
    waiting_time: float = -0.1
    queue_length: float = -0.5
    throughput: float = 1.0
    emergency_delay: float = -100.0
    switch_penalty: float = -2.0
    fairness: float = 0.3
    fuel_consumption: float = -0.05
    emissions: float = -0.08
    coordination: float = 0.5
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'waiting_time': self.waiting_time,
            'queue_length': self.queue_length,
            'throughput': self.throughput,
            'emergency_delay': self.emergency_delay,
            'switch_penalty': self.switch_penalty,
            'fairness': self.fairness,
            'fuel_consumption': self.fuel_consumption,
            'emissions': self.emissions,
            'coordination': self.coordination
        }


@dataclass
class RewardHistory:
    """Historical data for reward normalization"""
    waiting_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    queue_lengths: deque = field(default_factory=lambda: deque(maxlen=1000))
    throughputs: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def update(self, waiting: float, queue: float, throughput: float):
        self.waiting_times.append(waiting)
        self.queue_lengths.append(queue)
        self.throughputs.append(throughput)
    
    @property
    def avg_waiting(self) -> float:
        return np.mean(self.waiting_times) if self.waiting_times else 100
    
    @property
    def avg_queue(self) -> float:
        return np.mean(self.queue_lengths) if self.queue_lengths else 20
    
    @property
    def avg_throughput(self) -> float:
        return np.mean(self.throughputs) if self.throughputs else 30


class AdvancedRewardCalculator:
    """
    Sophisticated reward calculation with multiple objectives
    
    Features:
    - Weighted multi-objective optimization
    - Historical normalization
    - Emergency vehicle priority handling
    - Fairness across traffic approaches
    - Environmental impact tracking
    - Green wave coordination bonuses
    """
    
    def __init__(
        self,
        weights: Optional[RewardWeights] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize reward calculator
        
        Args:
            weights: Custom reward weights
            config: Additional configuration
        """
        self.weights = weights or RewardWeights()
        self.config = config or {}
        
        # Historical data for normalization
        self.history = RewardHistory()
        
        # Phase compatibility for green wave detection
        # Assumes phases 0,2 are N-S and 1,3 are E-W
        self.ns_phases = self.config.get('ns_phases', [0, 2])
        self.ew_phases = self.config.get('ew_phases', [1, 3])
        
        # Thresholds
        self.congestion_threshold = self.config.get('congestion_threshold', 0.7)
        self.starvation_threshold = self.config.get('starvation_threshold', 120)  # seconds
        self.high_throughput_bonus = self.config.get('high_throughput_bonus', 50)
        
        # Episode statistics
        self.episode_rewards = []
        self.step_count = 0
    
    def calculate_reward(
        self,
        state: Dict[str, Any],
        action: Any,
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Comprehensive reward calculation
        
        Args:
            state: Current state before action
            action: Action taken (0=keep, 1=switch, or phase index)
            next_state: State after action
            info: Additional information from environment
            
        Returns:
            Tuple of (total_reward, component_breakdown)
        """
        rewards = {}
        
        # 1. Waiting time component
        rewards['waiting_time'] = self._calc_waiting_time_reward(state, next_state)
        
        # 2. Queue length component
        rewards['queue_length'] = self._calc_queue_reward(state, next_state)
        
        # 3. Throughput component
        rewards['throughput'] = self._calc_throughput_reward(info)
        
        # 4. Emergency handling
        rewards['emergency_delay'] = self._calc_emergency_reward(info)
        
        # 5. Switching penalty
        rewards['switch_penalty'] = self._calc_switch_penalty(action, state)
        
        # 6. Fairness across lanes
        rewards['fairness'] = self._calc_fairness_reward(next_state)
        
        # 7. Environmental impact
        rewards['fuel_consumption'] = self._calc_fuel_reward(info)
        rewards['emissions'] = self._calc_emissions_reward(info)
        
        # 8. Multi-junction coordination
        rewards['coordination'] = self._calc_coordination_reward(state, next_state, info)
        
        # Calculate weighted sum
        weights_dict = self.weights.to_dict()
        total_reward = sum(
            weights_dict.get(k, 0) * v 
            for k, v in rewards.items()
        )
        
        # Update history
        self._update_history(next_state, info)
        
        # Track for episode statistics
        self.episode_rewards.append(total_reward)
        self.step_count += 1
        
        return total_reward, rewards
    
    def _calc_waiting_time_reward(
        self, 
        state: Dict[str, Any], 
        next_state: Dict[str, Any]
    ) -> float:
        """
        Reward based on change in total waiting time
        """
        prev_waiting = sum(state.get('waiting_times', [0]))
        current_waiting = sum(next_state.get('waiting_times', [0]))
        
        # Normalize by historical average
        avg_waiting = self.history.avg_waiting
        
        # Calculate improvement
        delta = (prev_waiting - current_waiting) / max(avg_waiting, 1)
        
        # Non-linear: penalize high waiting times more
        if current_waiting > avg_waiting * 1.5:
            delta -= 0.5  # Extra penalty for congestion
        
        # Bonus for very low waiting
        if current_waiting < avg_waiting * 0.5:
            delta += 0.3
        
        return delta
    
    def _calc_queue_reward(
        self, 
        state: Dict[str, Any], 
        next_state: Dict[str, Any]
    ) -> float:
        """
        Reward for reducing queue lengths
        """
        prev_queue = sum(state.get('queue_lengths', [0]))
        current_queue = sum(next_state.get('queue_lengths', [0]))
        
        # Basic improvement reward
        delta = prev_queue - current_queue
        
        # Penalize rapid queue growth
        if prev_queue > 0:
            growth_rate = (current_queue - prev_queue) / prev_queue
            if growth_rate > 0.2:  # Growing by >20%
                delta -= growth_rate * 10
        
        # Bonus for clearing queues
        if current_queue == 0 and prev_queue > 0:
            delta += 5
        
        return delta
    
    def _calc_throughput_reward(self, info: Dict[str, Any]) -> float:
        """
        Reward for vehicles passing through intersection
        """
        throughput = info.get('vehicles_passed', 0)
        
        # Bonus for high throughput
        if throughput > self.high_throughput_bonus:
            return throughput + 20
        
        return throughput
    
    def _calc_emergency_reward(self, info: Dict[str, Any]) -> float:
        """
        Large reward/penalty for emergency vehicle handling
        """
        if 'emergency_delay' not in info:
            return 0
        
        delay = info['emergency_delay']
        
        if delay == 0:
            return 0
        elif delay < 10:  # Fast clearance (<10 seconds)
            return 1000
        elif delay < 30:  # Acceptable (<30 seconds)
            return 500
        else:  # Too slow
            return -100 * delay
    
    def _calc_switch_penalty(
        self, 
        action: Any, 
        state: Dict[str, Any]
    ) -> float:
        """
        Penalty for frequent phase switching
        """
        # Determine if action is a switch
        is_switch = False
        
        if isinstance(action, int):
            is_switch = action == 1  # Binary action
        elif hasattr(action, 'item'):
            is_switch = action.item() == 1
        
        if not is_switch:
            return 0
        
        # Higher penalty during high congestion (avoid disruption)
        congestion_level = np.mean(state.get('densities', [0]))
        
        if congestion_level > self.congestion_threshold:
            return -5  # Higher penalty in congestion
        else:
            return -2  # Normal penalty
    
    def _calc_fairness_reward(self, state: Dict[str, Any]) -> float:
        """
        Reward for fair treatment of all approaches
        """
        waiting_times = state.get('waiting_times', [])
        
        if not waiting_times or len(waiting_times) < 2:
            return 0
        
        # Calculate variance (unfairness)
        variance = np.var(waiting_times)
        
        # Lower variance = more fair (capped penalty)
        fairness_score = -min(variance / 100, 5)
        
        # Bonus if no approach is starved
        max_waiting = max(waiting_times)
        if max_waiting < self.starvation_threshold:
            fairness_score += 0.5
        
        # Penalty for starvation
        if max_waiting > self.starvation_threshold * 2:
            fairness_score -= 2
        
        return fairness_score
    
    def _calc_fuel_reward(self, info: Dict[str, Any]) -> float:
        """
        Reward for fuel efficiency (lower is better)
        """
        # SUMO can track fuel consumption via emission output
        # Approximation: fuel ~ acceleration^2 + idling_time
        
        fuel_consumption = info.get('fuel_consumption', 0)
        
        # Could also estimate from state
        if fuel_consumption == 0 and 'speeds' in info:
            speeds = info['speeds']
            # Stopped vehicles consume fuel idling
            n_stopped = sum(1 for s in speeds if s < 0.1)
            fuel_consumption = n_stopped * 0.1  # Rough estimate
        
        return -fuel_consumption
    
    def _calc_emissions_reward(self, info: Dict[str, Any]) -> float:
        """
        Reward for low emissions (CO2, NOx, PM)
        """
        # SUMO emission models can provide this
        emissions = info.get('co2_emissions', 0)
        
        # Scale down large values
        return -emissions / 1000
    
    def _calc_coordination_reward(
        self,
        state: Dict[str, Any],
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> float:
        """
        Reward for good multi-junction coordination (green waves)
        """
        green_wave_bonus = 0
        
        # Check neighbor states
        neighbors_state = info.get('neighbors_state', {})
        
        if not neighbors_state:
            return 0
        
        our_phase = state.get('current_phase', 0)
        
        for neighbor_id, neighbor_data in neighbors_state.items():
            their_phase = neighbor_data.get('phase', 0)
            
            # Bonus if phases are compatible (green wave)
            if self._phases_compatible(our_phase, their_phase):
                green_wave_bonus += 1
            
            # Penalty if dumping traffic into congested neighbor
            neighbor_congestion = neighbor_data.get('congestion', 0)
            if neighbor_congestion > self.congestion_threshold:
                # Check if we're sending traffic to them
                if self._sending_traffic_to(state, our_phase, neighbor_id):
                    green_wave_bonus -= 2
        
        return green_wave_bonus
    
    def _phases_compatible(self, phase1: int, phase2: int) -> bool:
        """
        Check if two signal phases create green wave
        """
        # Compatible if both allowing same direction
        if (phase1 in self.ns_phases and phase2 in self.ns_phases):
            return True
        if (phase1 in self.ew_phases and phase2 in self.ew_phases):
            return True
        
        return False
    
    def _sending_traffic_to(
        self, 
        state: Dict[str, Any], 
        phase: int, 
        neighbor_id: str
    ) -> bool:
        """
        Check if current phase sends traffic to neighbor
        """
        # This requires topology information
        # Map phases to outgoing directions
        phase_directions = state.get('phase_directions', {})
        
        if not phase_directions:
            return False
        
        outgoing_direction = phase_directions.get(phase)
        neighbor_direction = state.get('neighbor_directions', {}).get(neighbor_id)
        
        return outgoing_direction == neighbor_direction
    
    def _update_history(self, state: Dict[str, Any], info: Dict[str, Any]):
        """
        Track metrics for normalization
        """
        waiting = sum(state.get('waiting_times', [0]))
        queue = sum(state.get('queue_lengths', [0]))
        throughput = info.get('vehicles_passed', 0)
        
        self.history.update(waiting, queue, throughput)
    
    def reset_episode(self):
        """Reset episode statistics"""
        self.episode_rewards = []
        self.step_count = 0
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get episode statistics"""
        if not self.episode_rewards:
            return {}
        
        return {
            'total_reward': sum(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': min(self.episode_rewards),
            'max_reward': max(self.episode_rewards),
            'n_steps': self.step_count
        }
    
    def get_reward_summary(self, rewards_breakdown: Dict[str, float]) -> str:
        """
        Get human-readable reward breakdown
        """
        lines = [
            "",
            "=" * 50,
            "REWARD BREAKDOWN",
            "=" * 50
        ]
        
        weights_dict = self.weights.to_dict()
        
        for component, value in rewards_breakdown.items():
            weighted = weights_dict.get(component, 0) * value
            lines.append(f"{component:20s}: {value:8.2f} (weighted: {weighted:8.2f})")
        
        total = sum(
            weights_dict.get(k, 0) * v 
            for k, v in rewards_breakdown.items()
        )
        lines.append("-" * 50)
        lines.append(f"{'TOTAL':20s}: {total:8.2f}")
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def print_reward_summary(self, rewards_breakdown: Dict[str, float]):
        """Print reward breakdown to console"""
        print(self.get_reward_summary(rewards_breakdown))


class AdaptiveRewardCalculator(AdvancedRewardCalculator):
    """
    Adaptive reward calculator that adjusts weights based on performance
    """
    
    def __init__(
        self,
        weights: Optional[RewardWeights] = None,
        config: Optional[Dict[str, Any]] = None,
        adaptation_rate: float = 0.01
    ):
        super().__init__(weights, config)
        self.adaptation_rate = adaptation_rate
        
        # Performance tracking
        self.component_performance = {
            'waiting_time': deque(maxlen=100),
            'queue_length': deque(maxlen=100),
            'throughput': deque(maxlen=100),
            'emergency_delay': deque(maxlen=100),
            'fairness': deque(maxlen=100),
        }
    
    def calculate_reward(
        self,
        state: Dict[str, Any],
        action: Any,
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate reward with adaptive weight adjustment"""
        total, breakdown = super().calculate_reward(state, action, next_state, info)
        
        # Track component performance
        for component, value in breakdown.items():
            if component in self.component_performance:
                self.component_performance[component].append(value)
        
        return total, breakdown
    
    def adapt_weights(self, target_metrics: Dict[str, float]):
        """
        Adjust weights based on gap between current and target metrics
        
        Args:
            target_metrics: Target values for each metric
        """
        weights_dict = self.weights.to_dict()
        
        for metric, target in target_metrics.items():
            if metric not in self.component_performance:
                continue
            
            history = self.component_performance[metric]
            if not history:
                continue
            
            current_avg = np.mean(history)
            
            # Increase weight if below target, decrease if above
            gap = target - current_avg
            adjustment = self.adaptation_rate * gap
            
            # Update weight (keeping sign)
            current_weight = weights_dict.get(metric, 0)
            new_weight = current_weight + adjustment
            
            # Apply to weights object
            if hasattr(self.weights, metric):
                setattr(self.weights, metric, new_weight)
        
        print(f"Adapted weights: {self.weights}")


class CurriculumRewardCalculator(AdvancedRewardCalculator):
    """
    Reward calculator with curriculum learning support
    
    Adjusts reward complexity based on training stage
    """
    
    def __init__(
        self,
        weights: Optional[RewardWeights] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(weights, config)
        self.stage = 0  # 0=easy, 1=medium, 2=hard, 3=full
        
        # Stage-specific weight multipliers
        self.stage_multipliers = {
            0: {  # Easy: focus on basic metrics
                'waiting_time': 1.0,
                'queue_length': 1.0,
                'throughput': 1.0,
                'emergency_delay': 0.0,
                'switch_penalty': 0.5,
                'fairness': 0.0,
                'fuel_consumption': 0.0,
                'emissions': 0.0,
                'coordination': 0.0
            },
            1: {  # Medium: add fairness and switching
                'waiting_time': 1.0,
                'queue_length': 1.0,
                'throughput': 1.0,
                'emergency_delay': 0.0,
                'switch_penalty': 1.0,
                'fairness': 1.0,
                'fuel_consumption': 0.0,
                'emissions': 0.0,
                'coordination': 0.0
            },
            2: {  # Hard: add emergency handling
                'waiting_time': 1.0,
                'queue_length': 1.0,
                'throughput': 1.0,
                'emergency_delay': 1.0,
                'switch_penalty': 1.0,
                'fairness': 1.0,
                'fuel_consumption': 0.5,
                'emissions': 0.5,
                'coordination': 0.0
            },
            3: {  # Full: all objectives
                'waiting_time': 1.0,
                'queue_length': 1.0,
                'throughput': 1.0,
                'emergency_delay': 1.0,
                'switch_penalty': 1.0,
                'fairness': 1.0,
                'fuel_consumption': 1.0,
                'emissions': 1.0,
                'coordination': 1.0
            }
        }
    
    def set_stage(self, stage: int):
        """Set curriculum stage (0-3)"""
        self.stage = min(max(stage, 0), 3)
        print(f"Reward calculator set to stage {self.stage}")
    
    def calculate_reward(
        self,
        state: Dict[str, Any],
        action: Any,
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate reward with stage-appropriate complexity"""
        # Get base rewards
        _, base_breakdown = super().calculate_reward(state, action, next_state, info)
        
        # Apply stage multipliers
        multipliers = self.stage_multipliers[self.stage]
        adjusted_breakdown = {
            k: v * multipliers.get(k, 1.0)
            for k, v in base_breakdown.items()
        }
        
        # Calculate adjusted total
        weights_dict = self.weights.to_dict()
        total = sum(
            weights_dict.get(k, 0) * v
            for k, v in adjusted_breakdown.items()
        )
        
        return total, adjusted_breakdown


# Utility functions
def create_reward_calculator(
    reward_type: str = 'advanced',
    **kwargs
) -> AdvancedRewardCalculator:
    """
    Factory function for reward calculators
    
    Args:
        reward_type: 'advanced', 'adaptive', or 'curriculum'
        **kwargs: Additional arguments for calculator
        
    Returns:
        Reward calculator instance
    """
    calculators = {
        'advanced': AdvancedRewardCalculator,
        'adaptive': AdaptiveRewardCalculator,
        'curriculum': CurriculumRewardCalculator
    }
    
    calculator_class = calculators.get(reward_type, AdvancedRewardCalculator)
    return calculator_class(**kwargs)


# Usage example
if __name__ == '__main__':
    # Create calculator
    calc = AdvancedRewardCalculator()
    
    # Simulate state transition
    state = {
        'waiting_times': [30, 25, 40, 35],
        'queue_lengths': [5, 4, 8, 6],
        'densities': [0.3, 0.25, 0.4, 0.35],
        'current_phase': 0
    }
    
    next_state = {
        'waiting_times': [25, 30, 35, 40],
        'queue_lengths': [3, 5, 6, 7],
        'densities': [0.25, 0.3, 0.35, 0.4],
        'current_phase': 1
    }
    
    info = {
        'vehicles_passed': 45,
        'emergency_delay': 0,
        'neighbors_state': {}
    }
    
    action = 1  # Switch phase
    
    # Calculate reward
    reward, breakdown = calc.calculate_reward(state, action, next_state, info)
    
    print(f"Total Reward: {reward:.2f}")
    calc.print_reward_summary(breakdown)
