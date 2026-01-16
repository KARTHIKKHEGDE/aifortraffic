"""
Baseline Controllers
Fixed-time and actuated controllers for comparison
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..utils.logger import setup_logger

logger = setup_logger("baselines")


@dataclass
class PhaseConfig:
    """Traffic signal phase configuration"""
    phase_id: int
    green_time: int  # seconds
    yellow_time: int = 3
    all_red_time: int = 2
    min_green: int = 10
    max_green: int = 60


class FixedTimeController:
    """
    Fixed-time traffic signal controller
    
    Classic baseline using predetermined cycle timing
    Represents the most common controller in Bangalore
    """
    
    def __init__(
        self,
        phases: List[PhaseConfig] = None,
        cycle_length: int = 120,
        offset: int = 0,
        junction_id: str = "default"
    ):
        """
        Initialize fixed-time controller
        
        Args:
            phases: List of phase configurations
            cycle_length: Total cycle length in seconds
            offset: Phase offset for coordination (0-cycle_length)
            junction_id: Junction identifier
        """
        self.junction_id = junction_id
        self.offset = offset
        
        if phases is None:
            # Default 4-phase config (typical Bangalore intersection)
            # Phase 0: NS straight, Phase 1: NS left turn
            # Phase 2: EW straight, Phase 3: EW left turn
            self.phases = [
                PhaseConfig(phase_id=0, green_time=35),  # NS main
                PhaseConfig(phase_id=1, green_time=20),  # NS left
                PhaseConfig(phase_id=2, green_time=35),  # EW main
                PhaseConfig(phase_id=3, green_time=20),  # EW left
            ]
        else:
            self.phases = phases
        
        # Calculate actual cycle length
        self.cycle_length = sum(
            p.green_time + p.yellow_time + p.all_red_time 
            for p in self.phases
        )
        
        # Build phase schedule
        self._build_schedule()
        
        # State
        self.current_time = 0
        self.current_phase_idx = 0
        
        logger.info(f"FixedTimeController initialized with {len(self.phases)} phases, cycle={self.cycle_length}s")
    
    def _build_schedule(self):
        """Build the phase schedule lookup"""
        self.schedule = []  # List of (phase_idx, phase_type) for each second
        
        for phase_idx, phase in enumerate(self.phases):
            # Green phase
            for _ in range(phase.green_time):
                self.schedule.append((phase_idx, 'green'))
            # Yellow phase
            for _ in range(phase.yellow_time):
                self.schedule.append((phase_idx, 'yellow'))
            # All-red phase
            for _ in range(phase.all_red_time):
                self.schedule.append((phase_idx, 'red'))
    
    def reset(self):
        """Reset controller state"""
        self.current_time = self.offset
        self.current_phase_idx = 0
    
    def get_action(self, obs: np.ndarray = None, training: bool = False) -> int:
        """
        Get action based on fixed schedule
        
        Args:
            obs: Observation (ignored for fixed-time)
            training: Training flag (ignored)
        
        Returns:
            Action (phase index)
        """
        # Get current position in cycle
        cycle_time = self.current_time % self.cycle_length
        
        # Get phase from schedule
        phase_idx, phase_type = self.schedule[cycle_time]
        
        # Increment time
        self.current_time += 1
        
        # Return phase index as action
        # Action 0 = keep, 1-3 = switch to phase 1-3
        if phase_type == 'green':
            return phase_idx + 1  # Action to set this phase
        else:
            return 0  # Keep current (transition phase)
    
    def get_phase_info(self) -> Dict[str, Any]:
        """Get current phase information"""
        cycle_time = self.current_time % self.cycle_length
        phase_idx, phase_type = self.schedule[cycle_time]
        
        return {
            'phase_idx': phase_idx,
            'phase_type': phase_type,
            'cycle_time': cycle_time,
            'cycle_length': self.cycle_length
        }


class ActuatedController:
    """
    Actuated traffic signal controller
    
    Extends green phases based on vehicle detection
    More advanced baseline than fixed-time
    """
    
    def __init__(
        self,
        phases: List[PhaseConfig] = None,
        extension_time: int = 3,
        gap_threshold: float = 3.0,
        detector_threshold: int = 2,
        junction_id: str = "default"
    ):
        """
        Initialize actuated controller
        
        Args:
            phases: Phase configurations
            extension_time: Extension per detection (seconds)
            gap_threshold: Maximum gap to allow extension (seconds)
            detector_threshold: Minimum vehicles to extend
            junction_id: Junction identifier
        """
        self.junction_id = junction_id
        self.extension_time = extension_time
        self.gap_threshold = gap_threshold
        self.detector_threshold = detector_threshold
        
        if phases is None:
            self.phases = [
                PhaseConfig(phase_id=0, green_time=20, min_green=10, max_green=50),
                PhaseConfig(phase_id=1, green_time=15, min_green=8, max_green=35),
                PhaseConfig(phase_id=2, green_time=20, min_green=10, max_green=50),
                PhaseConfig(phase_id=3, green_time=15, min_green=8, max_green=35),
            ]
        else:
            self.phases = phases
        
        # State
        self.current_phase_idx = 0
        self.phase_timer = 0
        self.in_transition = False
        self.transition_timer = 0
        self.last_detection_time = 0
        
        logger.info(f"ActuatedController initialized with {len(self.phases)} phases")
    
    def reset(self):
        """Reset controller state"""
        self.current_phase_idx = 0
        self.phase_timer = 0
        self.in_transition = False
        self.transition_timer = 0
        self.last_detection_time = 0
    
    def get_action(self, obs: np.ndarray, training: bool = False) -> int:
        """
        Get action based on vehicle detection
        
        Args:
            obs: Observation vector containing queue/detection info
            training: Training flag (ignored)
        
        Returns:
            Action (0=keep, 1-3=switch to phase)
        """
        current_phase = self.phases[self.current_phase_idx]
        
        # Handle transition phases
        if self.in_transition:
            self.transition_timer += 1
            transition_time = current_phase.yellow_time + current_phase.all_red_time
            
            if self.transition_timer >= transition_time:
                self.in_transition = False
                self.transition_timer = 0
                self.phase_timer = 0
                self.current_phase_idx = (self.current_phase_idx + 1) % len(self.phases)
            
            return 0  # Keep during transition
        
        # Increment phase timer
        self.phase_timer += 1
        
        # Extract detection from observation
        # Assuming obs contains queue lengths for each lane
        # Map phase to lanes: phase 0,1 = lanes 0,1 (NS), phase 2,3 = lanes 2,3 (EW)
        if self.current_phase_idx in [0, 1]:
            current_queue = obs[0] + obs[1] if len(obs) >= 4 else obs[0]
        else:
            current_queue = obs[2] + obs[3] if len(obs) >= 4 else obs[0]
        
        # Check for extension
        if current_queue >= self.detector_threshold:
            self.last_detection_time = self.phase_timer
        
        gap = self.phase_timer - self.last_detection_time
        
        # Decision logic
        if self.phase_timer < current_phase.min_green:
            # Min green not reached
            return 0
        
        if self.phase_timer >= current_phase.max_green:
            # Max green reached, must switch
            self._start_transition()
            return 0
        
        if gap > self.gap_threshold:
            # Gap-out, switch phase
            self._start_transition()
            return 0
        
        # Extend green
        return 0
    
    def _start_transition(self):
        """Start phase transition"""
        self.in_transition = True
        self.transition_timer = 0
    
    def get_phase_info(self) -> Dict[str, Any]:
        """Get current phase information"""
        return {
            'phase_idx': self.current_phase_idx,
            'phase_timer': self.phase_timer,
            'in_transition': self.in_transition,
            'max_green': self.phases[self.current_phase_idx].max_green
        }


class WebsterOptimalController:
    """
    Webster's optimal timing controller
    
    Calculates optimal cycle length and phase splits
    based on traffic flow rates
    """
    
    def __init__(
        self,
        num_phases: int = 4,
        saturation_flow: float = 1800,  # vehicles/hour
        lost_time_per_phase: float = 4,  # seconds
        junction_id: str = "default"
    ):
        """
        Initialize Webster controller
        
        Args:
            num_phases: Number of signal phases
            saturation_flow: Saturation flow rate (veh/hr)
            lost_time_per_phase: Lost time per phase (seconds)
            junction_id: Junction identifier
        """
        self.num_phases = num_phases
        self.saturation_flow = saturation_flow
        self.lost_time_per_phase = lost_time_per_phase
        self.junction_id = junction_id
        
        # Total lost time
        self.total_lost_time = num_phases * lost_time_per_phase
        
        # Current timing (updated based on observed flows)
        self.cycle_length = 90  # Initial estimate
        self.phase_greens = [20] * num_phases
        
        # Flow observations
        self.observed_flows = [0] * num_phases
        self.observation_count = 0
        
        # Controller state
        self.current_time = 0
        
        logger.info(f"WebsterOptimalController initialized")
    
    def update_timing(self, flow_rates: List[float]):
        """
        Update timing based on observed flow rates
        
        Args:
            flow_rates: Flow rate for each phase (veh/hr)
        """
        if sum(flow_rates) == 0:
            return
        
        # Calculate y (flow ratio) for each phase
        y_values = [f / self.saturation_flow for f in flow_rates]
        Y = sum(y_values)  # Total flow ratio
        
        if Y >= 1.0:
            Y = 0.95  # Clamp to prevent infinite cycle
        
        # Webster's optimal cycle length
        # C_opt = (1.5 * L + 5) / (1 - Y)
        C_opt = (1.5 * self.total_lost_time + 5) / (1 - Y)
        
        # Clamp to reasonable range
        self.cycle_length = max(60, min(180, C_opt))
        
        # Effective green time
        effective_green = self.cycle_length - self.total_lost_time
        
        # Distribute green time proportionally
        for i, y in enumerate(y_values):
            if Y > 0:
                self.phase_greens[i] = effective_green * (y / Y)
            else:
                self.phase_greens[i] = effective_green / self.num_phases
        
        logger.debug(f"Updated timing: cycle={self.cycle_length:.1f}s, greens={[f'{g:.1f}' for g in self.phase_greens]}")
    
    def reset(self):
        """Reset controller"""
        self.current_time = 0
        self.observed_flows = [0] * self.num_phases
    
    def get_action(self, obs: np.ndarray, training: bool = False) -> int:
        """Get action from Webster timing"""
        # Build schedule from current timing
        total_time = 0
        phase_ranges = []
        
        for i, green in enumerate(self.phase_greens):
            start = total_time
            end = total_time + green + self.lost_time_per_phase
            phase_ranges.append((start, end, i))
            total_time = end
        
        # Find current phase
        cycle_time = self.current_time % self.cycle_length
        
        current_phase = 0
        for start, end, phase in phase_ranges:
            if start <= cycle_time < end:
                current_phase = phase
                break
        
        self.current_time += 1
        
        # Return phase action
        return current_phase + 1


def run_baseline(
    env,
    controller,
    num_episodes: int = 10,
    max_steps: int = 3600
) -> Dict[str, Any]:
    """
    Run baseline controller and collect metrics
    
    Args:
        env: Traffic environment
        controller: Baseline controller
        num_episodes: Number of evaluation episodes
        max_steps: Max steps per episode
    
    Returns:
        Dictionary with metrics
    """
    from .metrics import TrafficMetrics
    
    metrics = TrafficMetrics()
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        controller.reset()
        
        episode_reward = 0
        episode_steps = 0
        phase_switches = 0
        last_phase = None
        
        for step in range(max_steps):
            # Get action from baseline controller
            action = controller.get_action(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track phase changes
            current_phase = info.get('current_phase', 0)
            if last_phase is not None and current_phase != last_phase:
                phase_switches += 1
            last_phase = current_phase
            
            # Record step metrics
            metrics.add_step_metrics(
                queue_lengths=info.get('queue_lengths', {}),
                waiting_times=info.get('waiting_times', {}),
                speeds=info.get('speeds', {}),
                vehicles_passed=info.get('vehicles_passed', 0),
                phase_duration=info.get('phase_duration', 0)
            )
            
            episode_reward += reward
            episode_steps += 1
            obs = next_obs
            
            if done:
                break
        
        metrics.add_episode_metrics(
            episode_length=episode_steps,
            episode_reward=episode_reward,
            phase_switches=phase_switches
        )
        
        logger.info(f"Baseline Episode {episode + 1}/{num_episodes}: reward={episode_reward:.2f}, steps={episode_steps}")
    
    return {
        'controller': controller.__class__.__name__,
        'metrics': metrics,
        'summary': metrics.get_summary()
    }


if __name__ == "__main__":
    print("Testing Baseline Controllers...")
    print("=" * 50)
    
    # Test Fixed-Time Controller
    print("\n1. Fixed-Time Controller:")
    fixed = FixedTimeController()
    fixed.reset()
    
    for i in range(10):
        action = fixed.get_action(np.zeros(18))
        phase_info = fixed.get_phase_info()
        print(f"  Step {i}: action={action}, phase={phase_info['phase_idx']}, type={phase_info['phase_type']}")
    
    # Test Actuated Controller
    print("\n2. Actuated Controller:")
    actuated = ActuatedController()
    actuated.reset()
    
    for i in range(10):
        obs = np.random.rand(18) * 10  # Random observation
        action = actuated.get_action(obs)
        phase_info = actuated.get_phase_info()
        print(f"  Step {i}: action={action}, phase={phase_info['phase_idx']}, timer={phase_info['phase_timer']}")
    
    # Test Webster Controller
    print("\n3. Webster Optimal Controller:")
    webster = WebsterOptimalController()
    webster.update_timing([400, 200, 500, 150])  # Sample flows
    print(f"  Cycle length: {webster.cycle_length:.1f}s")
    print(f"  Phase greens: {[f'{g:.1f}' for g in webster.phase_greens]}")
    
    print("\nBaseline controllers test complete!")
