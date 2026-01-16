"""
Curriculum Learning Training Script
Multi-stage training for adaptive traffic signal control

Stages:
1. Basic: Single junction, no weather, no emergency
2. Weather: Add weather variation
3. Emergency: Add emergency vehicle priority
4. Multi-Junction: Enable coordination
5. Full: Complete system with all features
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import ConfigManager
from src.utils.logger import setup_logger, TrainingLogger
from src.environment import BangaloreTrafficEnv
from src.agents import QLearningAgent, DQNAgent, PPOMultiAgent, train_qlearning, train_dqn

logger = setup_logger("curriculum_training")


class CurriculumTrainer:
    """
    Curriculum Learning Trainer
    
    Progressively increases task difficulty through stages
    """
    
    def __init__(
        self,
        env_config_path: str,
        training_config_path: str,
        agent_type: str = 'dqn',
        output_dir: str = 'runs',
        device: str = 'auto'
    ):
        """
        Initialize curriculum trainer
        
        Args:
            env_config_path: Path to environment config
            training_config_path: Path to training config
            agent_type: Agent type ('qlearning', 'dqn', 'ppo')
            output_dir: Output directory for models and logs
            device: PyTorch device
        """
        self.config_manager = ConfigManager()
        self.env_config = self.config_manager.load_yaml(env_config_path)
        self.training_config = self.config_manager.load_yaml(training_config_path)
        
        self.agent_type = agent_type
        self.device = device
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"{agent_type}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training logger
        self.training_logger = TrainingLogger(
            log_dir=str(self.output_dir / "logs"),
            experiment_name=f"curriculum_{agent_type}"
        )
        
        # Curriculum stages from config
        self.stages = self.training_config.get('curriculum', {}).get('stages', [])
        
        if not self.stages:
            # Default stages
            self.stages = [
                {
                    'name': 'basic',
                    'episodes': 100,
                    'weather_enabled': False,
                    'emergency_enabled': False,
                    'multi_junction': False
                },
                {
                    'name': 'weather',
                    'episodes': 150,
                    'weather_enabled': True,
                    'emergency_enabled': False,
                    'multi_junction': False
                },
                {
                    'name': 'emergency',
                    'episodes': 200,
                    'weather_enabled': True,
                    'emergency_enabled': True,
                    'multi_junction': False
                },
                {
                    'name': 'full',
                    'episodes': 300,
                    'weather_enabled': True,
                    'emergency_enabled': True,
                    'multi_junction': True
                }
            ]
        
        self.agent = None
        self.env = None
        self.current_stage = 0
        self.total_episodes = 0
        self.training_history = []
        
        logger.info(f"Curriculum trainer initialized with {len(self.stages)} stages")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _create_stage_config(self, stage: Dict[str, Any]) -> Dict[str, Any]:
        """Create environment config for specific stage"""
        config = self.env_config.copy()
        
        # Update config based on stage
        config['weather'] = config.get('weather', {})
        config['weather']['enabled'] = stage.get('weather_enabled', False)
        
        config['emergency'] = config.get('emergency', {})
        config['emergency']['enabled'] = stage.get('emergency_enabled', False)
        
        config['multi_junction'] = stage.get('multi_junction', False)
        
        # Traffic scenario based on stage
        if stage['name'] == 'basic':
            config['traffic_scenario'] = 'low'
        elif stage['name'] == 'weather':
            config['traffic_scenario'] = 'medium'
        elif stage['name'] == 'emergency':
            config['traffic_scenario'] = 'high'
        else:
            config['traffic_scenario'] = 'peak_hour'
        
        return config
    
    def _create_env(self, stage_config: Dict[str, Any]) -> BangaloreTrafficEnv:
        """Create environment for stage"""
        # Close existing environment
        if self.env is not None:
            try:
                self.env.close()
            except:
                pass
        
        return BangaloreTrafficEnv(stage_config)
    
    def _create_agent(self, env: BangaloreTrafficEnv, from_checkpoint: Optional[str] = None):
        """Create or load agent"""
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        agent_config = self.training_config.get(self.agent_type, {})
        
        if self.agent_type == 'qlearning':
            agent = QLearningAgent(
                state_size=state_size,
                action_size=action_size,
                learning_rate=agent_config.get('learning_rate', 0.1),
                gamma=agent_config.get('gamma', 0.95),
                epsilon=agent_config.get('epsilon', 1.0),
                epsilon_decay=agent_config.get('epsilon_decay', 0.995),
                epsilon_min=agent_config.get('epsilon_min', 0.01)
            )
        
        elif self.agent_type == 'dqn':
            agent = DQNAgent(
                state_size=state_size,
                action_size=action_size,
                hidden_layers=agent_config.get('hidden_layers', [256, 256, 128]),
                learning_rate=agent_config.get('learning_rate', 0.001),
                gamma=agent_config.get('gamma', 0.95),
                epsilon=agent_config.get('epsilon', 1.0),
                epsilon_decay=agent_config.get('epsilon_decay', 0.995),
                epsilon_min=agent_config.get('epsilon_min', 0.01),
                buffer_size=agent_config.get('buffer_size', 100000),
                batch_size=agent_config.get('batch_size', 64),
                target_update_freq=agent_config.get('target_update_freq', 500),
                dueling=agent_config.get('dueling', False),
                double=agent_config.get('double', True),
                prioritized=agent_config.get('prioritized', False),
                device=self.device
            )
        
        elif self.agent_type == 'ppo':
            ppo_config = self.training_config.get('ppo', {})
            agent = PPOMultiAgent(
                env=env,
                learning_rate=ppo_config.get('learning_rate', 3e-4),
                n_steps=ppo_config.get('n_steps', 2048),
                batch_size=ppo_config.get('batch_size', 64),
                n_epochs=ppo_config.get('n_epochs', 10),
                gamma=ppo_config.get('gamma', 0.99),
                device=self.device
            )
        
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")
        
        # Load from checkpoint if provided
        if from_checkpoint and Path(from_checkpoint).exists():
            try:
                agent.load(from_checkpoint)
                logger.info(f"Loaded agent from checkpoint: {from_checkpoint}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        
        return agent
    
    def train_stage(
        self,
        stage_idx: int,
        from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train single curriculum stage
        
        Args:
            stage_idx: Stage index
            from_checkpoint: Optional checkpoint to resume from
        
        Returns:
            Stage training results
        """
        stage = self.stages[stage_idx]
        stage_name = stage['name']
        num_episodes = stage['episodes']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Stage {stage_idx + 1}/{len(self.stages)}: {stage_name}")
        logger.info(f"Episodes: {num_episodes}")
        logger.info(f"Weather: {stage.get('weather_enabled', False)}")
        logger.info(f"Emergency: {stage.get('emergency_enabled', False)}")
        logger.info(f"Multi-Junction: {stage.get('multi_junction', False)}")
        logger.info(f"{'='*60}\n")
        
        # Create stage config and environment
        stage_config = self._create_stage_config(stage)
        self.env = self._create_env(stage_config)
        
        # Create or continue agent
        if self.agent is None or from_checkpoint:
            self.agent = self._create_agent(self.env, from_checkpoint)
        
        # Training
        max_steps = stage_config.get('simulation', {}).get('max_steps', 3600)
        
        stage_results = {
            'stage_name': stage_name,
            'episodes': [],
            'rewards': [],
            'waiting_times': [],
            'queue_lengths': [],
            'emergency_metrics': []
        }
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_waiting_time = 0
            episode_queue_length = 0
            
            for step in range(max_steps):
                # Get action
                if self.agent_type == 'ppo':
                    action, _ = self.agent.predict(obs, deterministic=False)
                    action = int(action[0]) if isinstance(action, np.ndarray) else int(action)
                else:
                    action = self.agent.get_action(obs, training=True)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store experience (for value-based methods)
                if self.agent_type in ['qlearning', 'dqn']:
                    self.agent.store_experience(obs, action, reward, next_obs, done)
                    
                    # Train step (DQN)
                    if self.agent_type == 'dqn':
                        self.agent.train_step()
                
                episode_reward += reward
                episode_steps += 1
                episode_waiting_time += info.get('avg_waiting_time', 0)
                episode_queue_length += info.get('avg_queue_length', 0)
                
                obs = next_obs
                
                if done:
                    break
            
            # End episode
            if hasattr(self.agent, 'end_episode'):
                self.agent.end_episode()
            
            # Record metrics
            avg_waiting_time = episode_waiting_time / max(episode_steps, 1)
            avg_queue_length = episode_queue_length / max(episode_steps, 1)
            
            stage_results['episodes'].append(self.total_episodes)
            stage_results['rewards'].append(episode_reward)
            stage_results['waiting_times'].append(avg_waiting_time)
            stage_results['queue_lengths'].append(avg_queue_length)
            
            if 'emergency_clearance_time' in info:
                stage_results['emergency_metrics'].append(info['emergency_clearance_time'])
            
            # Log training metrics
            self.training_logger.log_step({
                'episode': self.total_episodes,
                'stage': stage_name,
                'reward': episode_reward,
                'steps': episode_steps,
                'avg_waiting_time': avg_waiting_time,
                'avg_queue_length': avg_queue_length,
                'epsilon': getattr(self.agent, 'epsilon', 0)
            })
            
            self.total_episodes += 1
            
            # Periodic logging
            if (episode + 1) % 10 == 0:
                recent_reward = np.mean(stage_results['rewards'][-10:])
                recent_wait = np.mean(stage_results['waiting_times'][-10:])
                
                logger.info(
                    f"Stage {stage_name} | Episode {episode + 1}/{num_episodes} | "
                    f"Reward: {recent_reward:.2f} | Wait: {recent_wait:.1f}s"
                )
        
        # Save stage checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_stage_{stage_idx}_{stage_name}.pt"
        if self.agent_type == 'ppo':
            self.agent.save(str(checkpoint_path).replace('.pt', '.zip'))
        else:
            self.agent.save(str(checkpoint_path))
        
        logger.info(f"Stage {stage_name} complete. Checkpoint saved.")
        
        return stage_results
    
    def train_full_curriculum(self) -> Dict[str, Any]:
        """
        Train through all curriculum stages
        
        Returns:
            Full training results
        """
        logger.info("Starting Full Curriculum Training")
        logger.info(f"Total stages: {len(self.stages)}")
        
        all_results = {
            'agent_type': self.agent_type,
            'stages': [],
            'total_episodes': 0,
            'start_time': datetime.now().isoformat()
        }
        
        previous_checkpoint = None
        
        for stage_idx in range(len(self.stages)):
            stage_results = self.train_stage(stage_idx, previous_checkpoint)
            all_results['stages'].append(stage_results)
            all_results['total_episodes'] = self.total_episodes
            
            # Save intermediate results
            results_path = self.output_dir / "training_results.json"
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)
        
        all_results['end_time'] = datetime.now().isoformat()
        
        # Save final model
        final_model_path = self.output_dir / f"final_model_{self.agent_type}"
        if self.agent_type == 'ppo':
            self.agent.save(str(final_model_path) + '.zip')
        else:
            self.agent.save(str(final_model_path) + '.pt')
        
        # Save final results
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"\nCurriculum training complete!")
        logger.info(f"Total episodes: {self.total_episodes}")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"Final model saved to: {final_model_path}")
        
        return all_results
    
    def cleanup(self):
        """Clean up resources"""
        if self.env is not None:
            try:
                self.env.close()
            except:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum Learning Training for Bangalore Traffic Control"
    )
    
    parser.add_argument(
        '--agent', '-a',
        type=str,
        default='dqn',
        choices=['qlearning', 'dqn', 'ppo'],
        help='Agent type (default: dqn)'
    )
    
    parser.add_argument(
        '--env-config', '-e',
        type=str,
        default='configs/env_config.yaml',
        help='Environment config path'
    )
    
    parser.add_argument(
        '--training-config', '-t',
        type=str,
        default='configs/training_config.yaml',
        help='Training config path'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='runs',
        help='Output directory'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='auto',
        help='Device (auto, cuda, cpu)'
    )
    
    parser.add_argument(
        '--stage', '-s',
        type=int,
        default=None,
        help='Train only specific stage (0-indexed)'
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = CurriculumTrainer(
        env_config_path=args.env_config,
        training_config_path=args.training_config,
        agent_type=args.agent,
        output_dir=args.output,
        device=args.device
    )
    
    try:
        if args.stage is not None:
            # Train single stage
            results = trainer.train_stage(args.stage, args.checkpoint)
            logger.info(f"Stage {args.stage} training complete")
        else:
            # Full curriculum
            results = trainer.train_full_curriculum()
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Agent: {args.agent}")
        print(f"Total Episodes: {trainer.total_episodes}")
        print(f"Output: {trainer.output_dir}")
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
