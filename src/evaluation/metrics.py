"""
Traffic Metrics
Comprehensive metrics for evaluating traffic signal control performance
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from collections import defaultdict


@dataclass
class TrafficMetrics:
    """
    Container for traffic performance metrics
    
    Tracks multiple metrics across simulation episodes
    """
    
    # Queue metrics
    avg_queue_length: List[float] = field(default_factory=list)
    max_queue_length: List[float] = field(default_factory=list)
    queue_lengths_by_lane: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    # Waiting time metrics
    avg_waiting_time: List[float] = field(default_factory=list)
    max_waiting_time: List[float] = field(default_factory=list)
    total_waiting_time: List[float] = field(default_factory=list)
    
    # Throughput metrics
    vehicles_passed: List[int] = field(default_factory=list)
    throughput_per_hour: List[float] = field(default_factory=list)
    
    # Speed metrics
    avg_speed: List[float] = field(default_factory=list)
    min_speed: List[float] = field(default_factory=list)
    speed_variance: List[float] = field(default_factory=list)
    
    # Emergency metrics
    emergency_clearance_times: List[float] = field(default_factory=list)
    emergency_violations: List[int] = field(default_factory=list)
    emergency_response_rate: List[float] = field(default_factory=list)
    
    # Phase metrics
    avg_phase_duration: List[float] = field(default_factory=list)
    phase_switches: List[int] = field(default_factory=list)
    green_utilization: List[float] = field(default_factory=list)
    
    # Episode info
    episode_lengths: List[int] = field(default_factory=list)
    episode_rewards: List[float] = field(default_factory=list)
    
    def add_step_metrics(
        self,
        queue_lengths: Dict[str, float],
        waiting_times: Dict[str, float],
        speeds: Dict[str, float],
        vehicles_passed: int = 0,
        phase_duration: float = 0,
        emergency_info: Optional[Dict] = None
    ) -> None:
        """Add metrics from single simulation step"""
        # Queue
        avg_q = np.mean(list(queue_lengths.values())) if queue_lengths else 0
        max_q = max(queue_lengths.values()) if queue_lengths else 0
        self.avg_queue_length.append(avg_q)
        self.max_queue_length.append(max_q)
        
        for lane, q in queue_lengths.items():
            self.queue_lengths_by_lane[lane].append(q)
        
        # Waiting time
        avg_w = np.mean(list(waiting_times.values())) if waiting_times else 0
        max_w = max(waiting_times.values()) if waiting_times else 0
        total_w = sum(waiting_times.values()) if waiting_times else 0
        self.avg_waiting_time.append(avg_w)
        self.max_waiting_time.append(max_w)
        self.total_waiting_time.append(total_w)
        
        # Speed
        if speeds:
            avg_s = np.mean(list(speeds.values()))
            min_s = min(speeds.values())
            var_s = np.var(list(speeds.values()))
        else:
            avg_s, min_s, var_s = 0, 0, 0
        self.avg_speed.append(avg_s)
        self.min_speed.append(min_s)
        self.speed_variance.append(var_s)
        
        # Throughput
        self.vehicles_passed.append(vehicles_passed)
        
        # Phase
        self.avg_phase_duration.append(phase_duration)
        
        # Emergency
        if emergency_info:
            if 'clearance_time' in emergency_info:
                self.emergency_clearance_times.append(emergency_info['clearance_time'])
            if 'violation' in emergency_info:
                self.emergency_violations.append(1 if emergency_info['violation'] else 0)
    
    def add_episode_metrics(
        self,
        episode_length: int,
        episode_reward: float,
        phase_switches: int = 0,
        emergency_count: int = 0,
        emergency_cleared: int = 0
    ) -> None:
        """Add metrics from completed episode"""
        self.episode_lengths.append(episode_length)
        self.episode_rewards.append(episode_reward)
        self.phase_switches.append(phase_switches)
        
        # Emergency response rate
        if emergency_count > 0:
            self.emergency_response_rate.append(emergency_cleared / emergency_count)
        
        # Throughput per hour
        if episode_length > 0:
            hourly_throughput = sum(self.vehicles_passed[-episode_length:]) * (3600 / episode_length)
            self.throughput_per_hour.append(hourly_throughput)
        
        # Green utilization (ratio of green time to total time)
        if episode_length > 0:
            avg_phase = np.mean(self.avg_phase_duration[-episode_length:])
            self.green_utilization.append(avg_phase / 60.0)  # Normalized to minute
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics"""
        summary = {}
        
        # Queue metrics
        if self.avg_queue_length:
            summary['avg_queue_length'] = np.mean(self.avg_queue_length)
            summary['max_queue_length'] = max(self.max_queue_length)
            summary['queue_std'] = np.std(self.avg_queue_length)
        
        # Waiting time metrics
        if self.avg_waiting_time:
            summary['avg_waiting_time'] = np.mean(self.avg_waiting_time)
            summary['max_waiting_time'] = max(self.max_waiting_time)
            summary['total_waiting_time'] = sum(self.total_waiting_time)
            summary['waiting_time_std'] = np.std(self.avg_waiting_time)
        
        # Speed metrics
        if self.avg_speed:
            summary['avg_speed'] = np.mean(self.avg_speed)
            summary['min_speed'] = min(self.min_speed) if self.min_speed else 0
            summary['speed_variance'] = np.mean(self.speed_variance)
        
        # Throughput metrics
        if self.throughput_per_hour:
            summary['avg_throughput_per_hour'] = np.mean(self.throughput_per_hour)
            summary['total_vehicles_passed'] = sum(self.vehicles_passed)
        
        # Emergency metrics
        if self.emergency_clearance_times:
            summary['avg_emergency_clearance'] = np.mean(self.emergency_clearance_times)
            summary['max_emergency_clearance'] = max(self.emergency_clearance_times)
            summary['emergency_count'] = len(self.emergency_clearance_times)
        
        if self.emergency_response_rate:
            summary['emergency_response_rate'] = np.mean(self.emergency_response_rate)
        
        if self.emergency_violations:
            summary['emergency_violations'] = sum(self.emergency_violations)
        
        # Episode metrics
        if self.episode_rewards:
            summary['avg_episode_reward'] = np.mean(self.episode_rewards)
            summary['total_episodes'] = len(self.episode_rewards)
        
        if self.phase_switches:
            summary['avg_phase_switches'] = np.mean(self.phase_switches)
        
        if self.green_utilization:
            summary['avg_green_utilization'] = np.mean(self.green_utilization)
        
        return summary
    
    def get_per_lane_summary(self) -> Dict[str, Dict[str, float]]:
        """Get per-lane statistics"""
        return {
            lane: {
                'avg_queue': np.mean(queues),
                'max_queue': max(queues),
                'std_queue': np.std(queues)
            }
            for lane, queues in self.queue_lengths_by_lane.items()
        }
    
    def compare_to_baseline(self, baseline_metrics: 'TrafficMetrics') -> Dict[str, float]:
        """Compare metrics to baseline"""
        own_summary = self.get_summary()
        baseline_summary = baseline_metrics.get_summary()
        
        comparison = {}
        
        # Calculate improvement percentages
        metrics_to_compare = [
            ('avg_waiting_time', True),   # Lower is better
            ('avg_queue_length', True),   # Lower is better
            ('avg_speed', False),          # Higher is better
            ('avg_throughput_per_hour', False),  # Higher is better
            ('avg_emergency_clearance', True),   # Lower is better
        ]
        
        for metric, lower_is_better in metrics_to_compare:
            if metric in own_summary and metric in baseline_summary:
                own_val = own_summary[metric]
                baseline_val = baseline_summary[metric]
                
                if baseline_val != 0:
                    if lower_is_better:
                        improvement = (baseline_val - own_val) / baseline_val * 100
                    else:
                        improvement = (own_val - baseline_val) / baseline_val * 100
                    
                    comparison[f'{metric}_improvement_%'] = improvement
        
        return comparison
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'summary': self.get_summary(),
            'per_lane': self.get_per_lane_summary(),
            'raw_data': {
                'avg_queue_length': list(self.avg_queue_length),
                'avg_waiting_time': list(self.avg_waiting_time),
                'avg_speed': list(self.avg_speed),
                'episode_rewards': list(self.episode_rewards),
                'emergency_clearance_times': list(self.emergency_clearance_times),
            }
        }


def compute_metrics(
    env,
    agent,
    num_episodes: int = 10,
    max_steps: int = 3600,
    deterministic: bool = True
) -> TrafficMetrics:
    """
    Compute comprehensive metrics for agent performance
    
    Args:
        env: Traffic environment
        agent: Trained agent
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        deterministic: Use deterministic policy
    
    Returns:
        TrafficMetrics object with all metrics
    """
    metrics = TrafficMetrics()
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        phase_switches = 0
        last_phase = None
        emergency_count = 0
        emergency_cleared = 0
        
        for step in range(max_steps):
            # Get action from agent
            if hasattr(agent, 'predict'):
                action, _ = agent.predict(obs, deterministic=deterministic)
                action = int(action[0]) if isinstance(action, np.ndarray) else int(action)
            else:
                action = agent.get_action(obs, training=False)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track phase switches
            current_phase = info.get('current_phase', 0)
            if last_phase is not None and current_phase != last_phase:
                phase_switches += 1
            last_phase = current_phase
            
            # Track emergency
            if info.get('emergency_active', False):
                emergency_count += 1
            if info.get('emergency_cleared', False):
                emergency_cleared += 1
            
            # Add step metrics
            metrics.add_step_metrics(
                queue_lengths=info.get('queue_lengths', {}),
                waiting_times=info.get('waiting_times', {}),
                speeds=info.get('speeds', {}),
                vehicles_passed=info.get('vehicles_passed', 0),
                phase_duration=info.get('phase_duration', 0),
                emergency_info=info.get('emergency_info', None)
            )
            
            episode_reward += reward
            episode_steps += 1
            obs = next_obs
            
            if done:
                break
        
        # Add episode metrics
        metrics.add_episode_metrics(
            episode_length=episode_steps,
            episode_reward=episode_reward,
            phase_switches=phase_switches,
            emergency_count=emergency_count,
            emergency_cleared=emergency_cleared
        )
    
    return metrics


def compute_target_achievements(
    metrics: TrafficMetrics,
    baseline_metrics: TrafficMetrics,
    targets: Optional[Dict[str, float]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Check achievement of target improvements
    
    Args:
        metrics: RL agent metrics
        baseline_metrics: Baseline controller metrics
        targets: Target improvement percentages
    
    Returns:
        Achievement status for each target
    """
    if targets is None:
        targets = {
            'waiting_time_improvement': 33.0,    # 33% reduction
            'queue_length_improvement': 33.0,    # 33% reduction
            'emergency_clearance_improvement': 67.0,  # 67% faster
        }
    
    comparison = metrics.compare_to_baseline(baseline_metrics)
    own_summary = metrics.get_summary()
    baseline_summary = baseline_metrics.get_summary()
    
    achievements = {}
    
    # Waiting time improvement
    if 'avg_waiting_time_improvement_%' in comparison:
        improvement = comparison['avg_waiting_time_improvement_%']
        target = targets['waiting_time_improvement']
        achievements['waiting_time'] = {
            'target': target,
            'achieved': improvement,
            'met': improvement >= target,
            'baseline_value': baseline_summary.get('avg_waiting_time', 0),
            'agent_value': own_summary.get('avg_waiting_time', 0)
        }
    
    # Queue length improvement
    if 'avg_queue_length_improvement_%' in comparison:
        improvement = comparison['avg_queue_length_improvement_%']
        target = targets['queue_length_improvement']
        achievements['queue_length'] = {
            'target': target,
            'achieved': improvement,
            'met': improvement >= target,
            'baseline_value': baseline_summary.get('avg_queue_length', 0),
            'agent_value': own_summary.get('avg_queue_length', 0)
        }
    
    # Emergency clearance improvement
    if 'avg_emergency_clearance_improvement_%' in comparison:
        improvement = comparison['avg_emergency_clearance_improvement_%']
        target = targets['emergency_clearance_improvement']
        achievements['emergency_clearance'] = {
            'target': target,
            'achieved': improvement,
            'met': improvement >= target,
            'baseline_value': baseline_summary.get('avg_emergency_clearance', 0),
            'agent_value': own_summary.get('avg_emergency_clearance', 0)
        }
    
    return achievements


if __name__ == "__main__":
    # Test metrics
    print("Testing TrafficMetrics...")
    print("=" * 50)
    
    metrics = TrafficMetrics()
    
    # Simulate some data
    for step in range(100):
        metrics.add_step_metrics(
            queue_lengths={'lane_1': np.random.randint(0, 20), 'lane_2': np.random.randint(0, 15)},
            waiting_times={'lane_1': np.random.uniform(0, 60), 'lane_2': np.random.uniform(0, 50)},
            speeds={'lane_1': np.random.uniform(5, 15), 'lane_2': np.random.uniform(5, 15)},
            vehicles_passed=np.random.randint(0, 5),
            phase_duration=np.random.uniform(15, 45)
        )
    
    metrics.add_episode_metrics(
        episode_length=100,
        episode_reward=np.random.uniform(50, 100),
        phase_switches=np.random.randint(10, 30),
        emergency_count=5,
        emergency_cleared=4
    )
    
    print("Summary:")
    summary = metrics.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}")
    
    print("\nPer-lane summary:")
    per_lane = metrics.get_per_lane_summary()
    for lane, stats in per_lane.items():
        print(f"  {lane}: avg={stats['avg_queue']:.1f}, max={stats['max_queue']}")
    
    print("\nMetrics test complete!")
