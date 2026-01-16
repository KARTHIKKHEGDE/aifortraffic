"""
Test suite for Bangalore RL Traffic Control
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestQueueConfig:
    """Test queue length configuration"""
    
    def test_baseline_mode(self):
        from src.environment.queue_config import QueueLengthConfig
        
        config = QueueLengthConfig(mode="baseline")
        
        assert config.peak_factor == 1.0
        assert config.max_queue == 30
    
    def test_realistic_bangalore_mode(self):
        from src.environment.queue_config import QueueLengthConfig
        
        config = QueueLengthConfig(mode="realistic_bangalore")
        
        # Realistic mode should have higher values
        assert config.peak_factor > 1.0
        assert config.max_queue > 50
    
    def test_mode_switch(self):
        from src.environment.queue_config import QueueLengthConfig
        
        config = QueueLengthConfig(mode="baseline")
        assert config.peak_factor == 1.0
        
        config.set_mode("realistic_bangalore")
        assert config.peak_factor > 1.0


class TestWeatherModel:
    """Test weather model"""
    
    def test_weather_initialization(self):
        from src.weather.weather_model import WeatherModel
        
        model = WeatherModel(seed=42)
        
        # Should have valid initial state
        assert model.current_state is not None
        assert model.rain_probability == 0.15
    
    def test_weather_effects(self):
        from src.weather.weather_model import WeatherModel
        
        model = WeatherModel(seed=42)
        effects = model.get_effects()
        
        # Effects should have required attributes
        assert hasattr(effects, 'speed_factor')
        assert hasattr(effects, 'headway_factor')
        assert 0 < effects.speed_factor <= 1.0
    
    def test_weather_is_raining(self):
        from src.weather.weather_model import WeatherModel, WeatherState
        
        model = WeatherModel(seed=42)
        
        # Test clear weather
        model.current_state = WeatherState.CLEAR
        assert not model.is_raining()
        
        # Test rain
        model.current_state = WeatherState.RAIN
        assert model.is_raining()


class TestQLearningAgent:
    """Test Q-Learning agent"""
    
    def test_initialization(self):
        from src.agents.qlearning import QLearningAgent
        
        agent = QLearningAgent(state_size=18, action_size=4)
        
        assert agent.state_size == 18
        assert agent.action_size == 4
        assert agent.epsilon == 1.0
    
    def test_action_selection(self):
        from src.agents.qlearning import QLearningAgent
        
        agent = QLearningAgent(state_size=18, action_size=4, epsilon=0.0)  # No exploration
        
        state = np.zeros(18)
        action = agent.get_action(state, training=False)
        
        assert 0 <= action < 4
    
    def test_learning(self):
        from src.agents.qlearning import QLearningAgent
        
        agent = QLearningAgent(state_size=18, action_size=4)
        
        state = np.random.rand(18)
        action = 1
        reward = 10.0
        next_state = np.random.rand(18)
        
        # Update Q-values (the actual method name)
        agent.update(state, action, reward, next_state, done=False)
        
        # Q-table should be updated
        state_tuple = agent._discretize_state(state)
        assert state_tuple in agent.q_table


class TestDQNAgent:
    """Test DQN agent"""
    
    def test_initialization(self):
        from src.agents.dqn_agent import DQNAgent
        
        agent = DQNAgent(state_size=18, action_size=4, device='cpu')
        
        assert agent.state_size == 18
        assert agent.action_size == 4
    
    def test_forward_pass(self):
        from src.agents.dqn_agent import DQNNetwork
        import torch
        
        network = DQNNetwork(state_size=18, action_size=4)
        state = torch.randn(1, 18)
        
        q_values = network(state)
        
        assert q_values.shape == (1, 4)
    
    def test_action_selection(self):
        from src.agents.dqn_agent import DQNAgent
        
        agent = DQNAgent(state_size=18, action_size=4, device='cpu')
        agent.epsilon = 0.0  # No exploration
        
        state = np.random.rand(18).astype(np.float32)
        action = agent.get_action(state, training=False)
        
        assert 0 <= action < 4
    
    def test_experience_replay(self):
        from src.agents.dqn_agent import ReplayBuffer
        
        buffer = ReplayBuffer(capacity=100)
        
        for i in range(50):
            buffer.push(
                state=np.random.rand(18),
                action=np.random.randint(0, 4),
                reward=np.random.randn(),
                next_state=np.random.rand(18),
                done=False
            )
        
        assert len(buffer) == 50
        
        batch = buffer.sample(32)
        assert len(batch) == 32


class TestMetrics:
    """Test metrics calculation"""
    
    def test_traffic_metrics(self):
        from src.evaluation.metrics import TrafficMetrics
        
        metrics = TrafficMetrics()
        
        # Add some step metrics
        for _ in range(10):
            metrics.add_step_metrics(
                queue_lengths={'lane_1': np.random.randint(5, 20)},
                waiting_times={'lane_1': np.random.uniform(10, 50)},
                speeds={'lane_1': np.random.uniform(5, 15)}
            )
        
        summary = metrics.get_summary()
        
        assert 'avg_queue_length' in summary
        assert 'avg_waiting_time' in summary
        assert 'avg_speed' in summary
    
    def test_baseline_comparison(self):
        from src.evaluation.metrics import TrafficMetrics
        
        # Agent metrics (better performance) - use add_step_metrics for proper initialization
        agent_metrics = TrafficMetrics()
        for val in [30, 35, 32, 28, 30]:
            agent_metrics.add_step_metrics(
                queue_lengths={'lane_1': val / 4},  # Roughly 7-9 queue length
                waiting_times={'lane_1': val},
                speeds={'lane_1': 10.0}
            )
        
        # Baseline metrics (worse performance)
        baseline_metrics = TrafficMetrics()
        for val in [60, 65, 62, 58, 60]:
            baseline_metrics.add_step_metrics(
                queue_lengths={'lane_1': val / 4},  # Roughly 14-16 queue length
                waiting_times={'lane_1': val},
                speeds={'lane_1': 8.0}
            )
        
        comparison = agent_metrics.compare_to_baseline(baseline_metrics)
        
        # Agent should show improvement
        assert comparison.get('avg_waiting_time_improvement_%', 0) > 0


class TestBaselines:
    """Test baseline controllers"""
    
    def test_fixed_time_controller(self):
        from src.evaluation.baselines import FixedTimeController
        
        controller = FixedTimeController()
        controller.reset()
        
        actions = []
        for _ in range(10):
            action = controller.get_action(np.zeros(18))
            actions.append(action)
        
        # Should produce valid actions
        assert all(0 <= a <= 3 for a in actions)
    
    def test_actuated_controller(self):
        from src.evaluation.baselines import ActuatedController
        
        controller = ActuatedController()
        controller.reset()
        
        obs = np.random.rand(18) * 10  # Simulated observation
        action = controller.get_action(obs)
        
        assert 0 <= action <= 3


class TestEmergencyHandler:
    """Test emergency priority handler"""
    
    def test_initialization(self):
        from src.emergency.priority_handler import EmergencyPriorityHandler
        
        handler = EmergencyPriorityHandler()
        
        assert handler.max_override_duration == 120.0
        assert handler.is_active is False
    
    def test_emergency_detection(self):
        from src.emergency.priority_handler import EmergencyPriorityHandler
        
        handler = EmergencyPriorityHandler()
        
        # Simulate emergency vehicle with the correct API
        emergency_info = {
            'id': 'ambulance_1',
            'lane': 'north_lane_0',
            'junction': 'silk_board',
            'lane_index': 0,
            'speed': 10.0,
            'is_waiting': False
        }
        
        result = handler.detect_emergency(emergency_info)
        
        assert handler.is_active is True
        assert result is True
    
    def test_action_override(self):
        from src.emergency.priority_handler import EmergencyPriorityHandler, EmergencyVehicle
        
        handler = EmergencyPriorityHandler()
        handler.is_active = True
        handler.current_emergency = EmergencyVehicle(
            vehicle_id='ambulance_1',
            lane='north_lane_0',
            junction='silk_board',
            lane_index=0,
            speed=10.0
        )
        
        original_actions = {'silk_board': 2}
        overridden_actions = handler.override_actions(original_actions, {})
        
        # Should override to emergency action
        assert overridden_actions['silk_board'] == 3  # Emergency override action


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
