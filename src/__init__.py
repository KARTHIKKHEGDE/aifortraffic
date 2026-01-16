"""
Bangalore RL Traffic Control - Source Package
Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control
"""

__version__ = "1.0.0"
__author__ = "Bangalore Traffic RL Team"

from .utils.config import ConfigManager, load_config
from .utils.logger import setup_logger, TrainingLogger

# Environment
from .environment import BangaloreTrafficEnv, SUMOConnector, QueueLengthConfig

# Agents
from .agents import QLearningAgent, DQNAgent, PPOMultiAgent

# Emergency
from .emergency import EmergencyPriorityHandler

# Weather
from .weather import WeatherModel, BangaloreWeatherModel

# Evaluation
from .evaluation import TrafficMetrics, FixedTimeController, ActuatedController, ResultsAnalyzer

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Utils
    "ConfigManager",
    "load_config",
    "setup_logger",
    "TrainingLogger",
    # Environment
    "BangaloreTrafficEnv",
    "SUMOConnector",
    "QueueLengthConfig",
    # Agents
    "QLearningAgent",
    "DQNAgent",
    "PPOMultiAgent",
    # Emergency
    "EmergencyPriorityHandler",
    # Weather
    "WeatherModel",
    "BangaloreWeatherModel",
    # Evaluation
    "TrafficMetrics",
    "FixedTimeController",
    "ActuatedController",
    "ResultsAnalyzer",
]

