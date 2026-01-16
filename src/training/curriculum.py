"""
Curriculum Learning Trainer
Implements progressive difficulty training for traffic control agents.

Training Stages:
1. Easy: Low traffic, no weather, no emergencies
2. Weather: Medium traffic with rain effects
3. Emergency: Medium traffic with ambulance priority
4. Full: High traffic, weather, and emergencies

This approach helps agents learn basic skills before tackling complex scenarios.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time

from ..utils.logger import setup_logger

logger = setup_logger("curriculum")


@dataclass
class TrainingStage:
    """Configuration for a curriculum training stage"""
    name: str
    episodes: int
    
    # Environment configuration
    traffic_level: str  # 'low', 'medium', 'high'
    enable_weather: bool
    enable_emergency: bool
    
    # Learning parameters
    learning_rate: Optional[float] = None
    epsilon: Optional[float] = None
    epsilon_decay: Optional[float] = None
    
    # Success criteria
    target_reward: Optional[float] = None
    max_queue_threshold: Optional[float] = None
    
    # Stage-specific reward weights
    reward_weights: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# DEFAULT CURRICULUM STAGES
# ============================================================================

DEFAULT_CURRICULUM = [
    TrainingStage(
        name="easy",
        episodes=500,
        traffic_level="low",
        enable_weather=False,
        enable_emergency=False,
        learning_rate=0.001,
        epsilon=1.0,
        epsilon_decay=0.995,
        target_reward=-500,  # Low negative (good)
        max_queue_threshold=30,
        reward_weights={
            'waiting': -0.1,
            'queue': -0.5,
            'throughput': 1.0,
            'switch': -5.0,
        }
    ),
    
    TrainingStage(
        name="weather",
        episodes=300,
        traffic_level="medium",
        enable_weather=True,
        enable_emergency=False,
        learning_rate=0.0005,
        epsilon=0.5,  # Start with learned policy
        epsilon_decay=0.99,
        target_reward=-800,
        max_queue_threshold=50,
        reward_weights={
            'waiting': -0.15,
            'queue': -0.6,
            'throughput': 1.0,
            'switch': -15.0,  # Higher penalty during rain
        }
    ),
    
    TrainingStage(
        name="emergency",
        episodes=300,
        traffic_level="medium",
        enable_weather=False,
        enable_emergency=True,
        learning_rate=0.0005,
        epsilon=0.4,
        epsilon_decay=0.99,
        target_reward=-700,
        max_queue_threshold=50,
        reward_weights={
            'waiting': -0.1,
            'queue': -0.5,
            'throughput': 1.0,
            'switch': -10.0,
            'emergency': 100.0,  # Big reward for handling emergencies
        }
    ),
    
    TrainingStage(
        name="full",
        episodes=900,
        traffic_level="high",
        enable_weather=True,
        enable_emergency=True,
        learning_rate=0.0003,
        epsilon=0.3,
        epsilon_decay=0.995,
        target_reward=-1200,
        max_queue_threshold=80,
        reward_weights={
            'waiting': -0.15,
            'queue': -0.7,
            'throughput': 1.2,
            'switch': -15.0,
            'emergency': 150.0,
        }
    ),
]


class CurriculumTrainer:
    """
    Manages curriculum-based training for traffic control agents.
    
    Features:
    - Progressive difficulty stages
    - Automatic stage advancement based on performance
    - Checkpoint saving between stages
    - Performance tracking and logging
    """
    
    def __init__(
        self,
        env_class: type,
        agent_class: type,
        stages: Optional[List[TrainingStage]] = None,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs"
    ):
        """
        Initialize curriculum trainer.
        
        Args:
            env_class: Environment class to instantiate
            agent_class: Agent class to instantiate
            stages: List of training stages (uses default if None)
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Directory for training logs
        """
        self.env_class = env_class
        self.agent_class = agent_class
        self.stages = stages or DEFAULT_CURRICULUM
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_stage_idx = 0
        self.total_episodes = 0
        self.training_history: List[Dict] = []
        
        # Environment and agent
        self.env = None
        self.agent = None
    
    def configure_environment(self, stage: TrainingStage) -> Dict[str, Any]:
        """
        Create environment configuration for a stage.
        
        Args:
            stage: Current training stage
        
        Returns:
            Environment config dictionary
        """
        # Traffic level to queue mode mapping
        traffic_map = {
            'low': 'baseline',
            'medium': 'realistic_bangalore',
            'high': 'realistic_bangalore',  # With higher multiplier
        }
        
        return {
            'queue_mode': traffic_map[stage.traffic_level],
            'enable_weather': stage.enable_weather,
            'rain_probability': 0.15 if stage.enable_weather else 0.0,
            'reward_waiting_weight': stage.reward_weights.get('waiting', -0.1),
            'reward_queue_weight': stage.reward_weights.get('queue', -0.5),
            'reward_throughput_weight': stage.reward_weights.get('throughput', 1.0),
            'reward_switch_penalty': stage.reward_weights.get('switch', -10.0),
            'reward_emergency_weight': stage.reward_weights.get('emergency', 100.0),
        }
    
    def configure_agent(self, stage: TrainingStage) -> Dict[str, Any]:
        """
        Create agent configuration for a stage.
        
        Args:
            stage: Current training stage
        
        Returns:
            Agent config dictionary
        """
        config = {}
        
        if stage.learning_rate is not None:
            config['learning_rate'] = stage.learning_rate
        
        if stage.epsilon is not None:
            config['epsilon'] = stage.epsilon
        
        if stage.epsilon_decay is not None:
            config['epsilon_decay'] = stage.epsilon_decay
        
        return config
    
    def train_stage(
        self,
        stage: TrainingStage,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train agent on a single stage.
        
        Args:
            stage: Training stage configuration
            progress_callback: Optional callback for progress updates
        
        Returns:
            Stage training results
        """
        logger.info(f"Starting stage: {stage.name}")
        logger.info(f"  Episodes: {stage.episodes}")
        logger.info(f"  Traffic: {stage.traffic_level}")
        logger.info(f"  Weather: {stage.enable_weather}")
        logger.info(f"  Emergency: {stage.enable_emergency}")
        
        # Configure environment
        env_config = self.configure_environment(stage)
        
        # Update agent parameters
        agent_config = self.configure_agent(stage)
        if hasattr(self.agent, 'update_config'):
            self.agent.update_config(agent_config)
        
        # Training metrics
        stage_rewards = []
        stage_queues = []
        stage_emergencies = []
        best_reward = float('-inf')
        
        start_time = time.time()
        
        for episode in range(stage.episodes):
            # Reset environment
            obs, info = self.env.reset()
            
            episode_reward = 0
            max_queue = 0
            emergency_cleared = 0
            done = False
            
            while not done:
                # Get action from agent
                if hasattr(self.agent, 'predict'):
                    # Stable-Baselines3 style
                    action, _ = self.agent.predict(obs)
                elif hasattr(self.agent, 'get_action'):
                    # Custom agent style
                    action = self.agent.get_action(obs, training=True)
                else:
                    action = self.env.action_space.sample()
                
                # Step environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Handle reward (may be dict for multi-agent)
                if isinstance(reward, dict):
                    step_reward = sum(reward.values())
                else:
                    step_reward = reward
                
                # Update agent
                if hasattr(self.agent, 'update'):
                    self.agent.update(obs, action, step_reward, next_obs, done)
                elif hasattr(self.agent, 'store_transition'):
                    self.agent.store_transition(obs, action, step_reward, next_obs, done)
                    if hasattr(self.agent, 'train_step'):
                        self.agent.train_step()
                
                # Track metrics
                episode_reward += step_reward
                if 'metrics' in info:
                    max_queue = max(max_queue, info['metrics'].get('total_queue_length', 0))
                if info.get('emergency_cleared', False):
                    emergency_cleared += 1
                
                obs = next_obs
            
            # End of episode
            if hasattr(self.agent, 'end_episode'):
                self.agent.end_episode()
            
            # Record metrics
            stage_rewards.append(episode_reward)
            stage_queues.append(max_queue)
            stage_emergencies.append(emergency_cleared)
            self.total_episodes += 1
            
            # Update best
            if episode_reward > best_reward:
                best_reward = episode_reward
                self._save_checkpoint(f"best_{stage.name}")
            
            # Progress callback
            if progress_callback:
                progress_callback({
                    'stage': stage.name,
                    'episode': episode,
                    'total_episodes': stage.episodes,
                    'reward': episode_reward,
                    'avg_reward': np.mean(stage_rewards[-100:]),
                    'max_queue': max_queue,
                })
            
            # Logging
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(stage_rewards[-50:])
                avg_queue = np.mean(stage_queues[-50:])
                logger.info(
                    f"  Episode {episode + 1}/{stage.episodes}: "
                    f"Avg Reward={avg_reward:.1f}, "
                    f"Avg Max Queue={avg_queue:.1f}"
                )
        
        # Stage complete
        elapsed = time.time() - start_time
        
        results = {
            'stage': stage.name,
            'episodes': stage.episodes,
            'total_time': elapsed,
            'avg_reward': np.mean(stage_rewards),
            'best_reward': best_reward,
            'final_avg_reward': np.mean(stage_rewards[-100:]),
            'avg_max_queue': np.mean(stage_queues),
            'final_avg_queue': np.mean(stage_queues[-100:]),
            'total_emergencies': sum(stage_emergencies),
        }
        
        # Check success criteria
        if stage.target_reward is not None:
            results['target_achieved'] = results['final_avg_reward'] >= stage.target_reward
        
        if stage.max_queue_threshold is not None:
            results['queue_target_achieved'] = results['final_avg_queue'] <= stage.max_queue_threshold
        
        logger.info(f"Stage {stage.name} complete!")
        logger.info(f"  Average Reward: {results['avg_reward']:.1f}")
        logger.info(f"  Best Reward: {results['best_reward']:.1f}")
        logger.info(f"  Average Max Queue: {results['avg_max_queue']:.1f}")
        
        return results
    
    def train(
        self,
        agent_kwargs: Optional[Dict] = None,
        env_kwargs: Optional[Dict] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Run full curriculum training.
        
        Args:
            agent_kwargs: Additional agent initialization arguments
            env_kwargs: Additional environment initialization arguments
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of results for each stage
        """
        logger.info("="*60)
        logger.info("CURRICULUM TRAINING")
        logger.info("="*60)
        logger.info(f"Stages: {[s.name for s in self.stages]}")
        logger.info(f"Total episodes: {sum(s.episodes for s in self.stages)}")
        
        # Initialize environment
        self.env = self.env_class(**(env_kwargs or {}))
        
        # Get observation/action dimensions
        if hasattr(self.env, 'observation_space'):
            if hasattr(self.env.observation_space, 'shape'):
                state_size = self.env.observation_space.shape[0]
            else:
                # Dict observation space
                first_space = list(self.env.observation_space.spaces.values())[0]
                state_size = first_space.shape[0]
        else:
            state_size = 33  # Default
        
        if hasattr(self.env, 'action_space'):
            if hasattr(self.env.action_space, 'n'):
                action_size = self.env.action_space.n
            else:
                # Dict action space
                action_size = 2  # Binary per junction
        else:
            action_size = 2  # Default
        
        # Initialize agent
        default_agent_kwargs = {
            'state_size': state_size,
            'action_size': action_size,
        }
        default_agent_kwargs.update(agent_kwargs or {})
        self.agent = self.agent_class(**default_agent_kwargs)
        
        # Train each stage
        all_results = []
        
        for i, stage in enumerate(self.stages):
            self.current_stage_idx = i
            
            logger.info(f"\n{'='*60}")
            logger.info(f"STAGE {i+1}/{len(self.stages)}: {stage.name.upper()}")
            logger.info(f"{'='*60}")
            
            results = self.train_stage(stage, progress_callback)
            all_results.append(results)
            self.training_history.append(results)
            
            # Save stage checkpoint
            self._save_checkpoint(f"stage_{stage.name}")
            
            # Save training history
            self._save_history()
        
        # Final save
        self._save_checkpoint("final")
        
        logger.info("\n" + "="*60)
        logger.info("CURRICULUM TRAINING COMPLETE")
        logger.info("="*60)
        
        # Summary
        for results in all_results:
            logger.info(
                f"  {results['stage']}: "
                f"Reward={results['final_avg_reward']:.1f}, "
                f"Queue={results['final_avg_queue']:.1f}"
            )
        
        return all_results
    
    def _save_checkpoint(self, name: str) -> None:
        """Save agent checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        
        if hasattr(self.agent, 'save'):
            self.agent.save(str(checkpoint_path))
            logger.debug(f"Checkpoint saved: {checkpoint_path}")
        elif hasattr(self.agent, 'q_table'):
            # Q-learning agent
            import pickle
            with open(checkpoint_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(self.agent.q_table, f)
    
    def load_checkpoint(self, name: str) -> bool:
        """Load agent checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        
        if hasattr(self.agent, 'load') and checkpoint_path.exists():
            self.agent.load(str(checkpoint_path))
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return True
        
        pkl_path = checkpoint_path.with_suffix('.pkl')
        if pkl_path.exists():
            import pickle
            with open(pkl_path, 'rb') as f:
                self.agent.q_table = pickle.load(f)
            logger.info(f"Checkpoint loaded: {pkl_path}")
            return True
        
        return False
    
    def _save_history(self) -> None:
        """Save training history to JSON."""
        history_path = self.log_dir / "training_history.json"
        
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def evaluate_stage(
        self,
        stage_name: str,
        num_episodes: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate current agent on a specific stage.
        
        Args:
            stage_name: Name of stage to evaluate on
            num_episodes: Number of evaluation episodes
        
        Returns:
            Evaluation results
        """
        stage = next((s for s in self.stages if s.name == stage_name), None)
        if stage is None:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        logger.info(f"Evaluating on stage: {stage_name}")
        
        # Configure environment for evaluation
        env_config = self.configure_environment(stage)
        
        rewards = []
        queues = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            max_queue = 0
            done = False
            
            while not done:
                # Deterministic action
                if hasattr(self.agent, 'predict'):
                    action, _ = self.agent.predict(obs, deterministic=True)
                else:
                    action = self.agent.get_action(obs, training=False)
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                if isinstance(reward, dict):
                    episode_reward += sum(reward.values())
                else:
                    episode_reward += reward
                
                if 'metrics' in info:
                    max_queue = max(max_queue, info['metrics'].get('total_queue_length', 0))
            
            rewards.append(episode_reward)
            queues.append(max_queue)
        
        results = {
            'stage': stage_name,
            'episodes': num_episodes,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_max_queue': np.mean(queues),
            'std_max_queue': np.std(queues),
        }
        
        logger.info(
            f"Evaluation results: "
            f"Reward={results['avg_reward']:.1f}±{results['std_reward']:.1f}, "
            f"Queue={results['avg_max_queue']:.1f}±{results['std_max_queue']:.1f}"
        )
        
        return results
