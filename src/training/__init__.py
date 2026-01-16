"""
Training module for traffic control agents.

Includes:
- Curriculum learning
- Hyperparameter tuning
- Distributed training
"""

from .curriculum import CurriculumTrainer, TrainingStage, DEFAULT_CURRICULUM
from .hyperparameter_tuning import (
    HyperparameterTuner,
    HyperparameterSpace,
    TrialResult,
    tune_dqn_agent,
)
from .distributed_training import (
    DistributedTrainer,
    DistributedConfig,
    VectorizedEnvManager,
    RolloutBuffer,
    create_distributed_trainer,
)

__all__ = [
    # Curriculum
    'CurriculumTrainer',
    'TrainingStage',
    'DEFAULT_CURRICULUM',
    # Hyperparameter tuning
    'HyperparameterTuner',
    'HyperparameterSpace',
    'TrialResult',
    'tune_dqn_agent',
    # Distributed training
    'DistributedTrainer',
    'DistributedConfig',
    'VectorizedEnvManager',
    'RolloutBuffer',
    'create_distributed_trainer',
]
