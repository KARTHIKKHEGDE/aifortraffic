#!/usr/bin/env python3
"""
Main Training Script
Trains traffic control agents using curriculum learning.

Supports:
- Q-Learning (tabular, for proof of concept)
- DQN (Deep Q-Network with experience replay)
- PPO (Proximal Policy Optimization via Stable-Baselines3)

Usage:
    python 04_train_agents.py --algorithm dqn --episodes 2000
    python 04_train_agents.py --algorithm ppo --curriculum
    python 04_train_agents.py --algorithm qlearning --mode baseline
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("training")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train traffic control agents"
    )
    
    parser.add_argument(
        '--algorithm', '-a',
        type=str,
        choices=['qlearning', 'dqn', 'ppo'],
        default='dqn',
        help='Algorithm to use (default: dqn)'
    )
    
    parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=1000,
        help='Number of training episodes (default: 1000)'
    )
    
    parser.add_argument(
        '--curriculum',
        action='store_true',
        help='Use curriculum learning (progressive difficulty)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['baseline', 'realistic_bangalore', 'calibrated'],
        default='realistic_bangalore',
        help='Traffic mode (default: realistic_bangalore)'
    )
    
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Use SUMO-GUI for visualization'
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='models',
        help='Output directory for saved models (default: models)'
    )
    
    parser.add_argument(
        '--allow-mock',
        action='store_true',
        dest='allow_mock',
        help='Allow using MockTrafficEnv if SUMO not available (DEVELOPMENT ONLY)'
    )
    
    return parser.parse_args()


def create_environment(args):
    """Create the traffic environment.
    
    Uses REAL SUMO simulation with actual Bangalore OSM data.
    Falls back to MockTrafficEnv only if SUMO is not installed.
    """
    import os
    
    # Check if SUMO is available
    sumo_home = os.environ.get('SUMO_HOME')
    if not sumo_home:
        # Try to find SUMO
        possible_paths = [
            r"C:\Program Files (x86)\Eclipse\Sumo",
            r"C:\Program Files\Eclipse\Sumo",
            r"/usr/share/sumo",
            r"/opt/sumo",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                os.environ['SUMO_HOME'] = path
                sumo_home = path
                break
    
    if sumo_home:
        logger.info(f"✓ SUMO found at: {sumo_home}")
        logger.info("  Using REAL Bangalore traffic simulation")
    else:
        logger.warning("✗ SUMO not found. Install SUMO for real simulation.")
        raise EnvironmentError("SUMO not installed. Please install SUMO for real simulation.")
    
    from src.environment.multi_junction_env import MultiJunctionTrafficEnv, EnvConfig, FlattenedTrafficEnv
    
    config = EnvConfig(
        junction_ids=['silk_board', 'tin_factory', 'hebbal', 'marathahalli'],
        queue_mode=args.mode,
        enable_weather=True,
        max_steps=3600,
        sumo_gui=args.gui,
    )
    
    env = MultiJunctionTrafficEnv(config=config)
    
    # Flatten for single-agent algorithms
    return FlattenedTrafficEnv(env)


def create_agent(algorithm: str, env, args):
    """Create the RL agent."""
    
    if algorithm == 'qlearning':
        from src.agents.qlearning import QLearningAgent
        
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        agent = QLearningAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            discretization_bins=10,
            seed=args.seed
        )
        return agent
    
    elif algorithm == 'dqn':
        from src.agents.dqn_agent import DQNAgent
        
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            buffer_size=100000,
            batch_size=64,
            update_frequency=4,
            target_update_frequency=1000,
            device='auto',
            seed=args.seed
        )
        return agent
    
    elif algorithm == 'ppo':
        from src.agents.ppo_agent import PPOMultiAgent
        
        agent = PPOMultiAgent(
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=args.seed
        )
        return agent
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def train_simple(env, agent, args):
    """Simple training loop without curriculum."""
    import numpy as np
    
    logger.info(f"Training {args.algorithm.upper()} for {args.episodes} episodes")
    
    # Tracking
    episode_rewards = []
    episode_queues = []
    best_reward = float('-inf')
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        episode_reward = 0
        max_queue = 0
        done = False
        step = 0
        
        while not done:
            # Get action
            if hasattr(agent, 'get_action'):
                action = agent.get_action(obs, training=True)
            elif hasattr(agent, 'predict'):
                action, _ = agent.predict(obs)
                action = int(action)
            else:
                action = env.action_space.sample()
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update agent
            if hasattr(agent, 'update'):
                agent.update(obs, action, reward, next_obs, done)
            elif hasattr(agent, 'store_experience'):
                agent.store_experience(obs, action, reward, next_obs, done)
                if hasattr(agent, 'train_step'):
                    agent.train_step()
            
            episode_reward += reward
            if 'metrics' in info:
                max_queue = max(max_queue, info['metrics'].get('total_queue_length', 0))
            
            obs = next_obs
            step += 1
        
        # End of episode
        if hasattr(agent, 'end_episode'):
            agent.end_episode()
        
        episode_rewards.append(episode_reward)
        episode_queues.append(max_queue)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            save_path = output_dir / f"{args.algorithm}_best.pt"
            if hasattr(agent, 'save'):
                agent.save(str(save_path))
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_queue = np.mean(episode_queues[-10:])
            
            if hasattr(agent, 'epsilon'):
                eps = agent.epsilon
            else:
                eps = 0.0
            
            logger.info(
                f"Episode {episode + 1}/{args.episodes} | "
                f"Reward: {episode_reward:.1f} | "
                f"Avg(10): {avg_reward:.1f} | "
                f"Queue: {max_queue:.0f} | "
                f"Epsilon: {eps:.3f}"
            )
        
        # Periodic checkpoint
        if (episode + 1) % 100 == 0:
            save_path = output_dir / f"{args.algorithm}_ep{episode+1}.pt"
            if hasattr(agent, 'save'):
                agent.save(str(save_path))
    
    # Final save
    save_path = output_dir / f"{args.algorithm}_final.pt"
    if hasattr(agent, 'save'):
        agent.save(str(save_path))
    
    logger.info(f"Training complete. Best reward: {best_reward:.1f}")
    logger.info(f"Models saved to: {output_dir}")
    
    return episode_rewards


def train_with_curriculum(env, agent_class, args):
    """Training with curriculum learning."""
    from src.training.curriculum import CurriculumTrainer
    
    logger.info("Starting curriculum training...")
    
    trainer = CurriculumTrainer(
        env_class=type(env),
        agent_class=agent_class,
        checkpoint_dir=Path(args.output) / "checkpoints",
        log_dir=Path(args.output) / "logs"
    )
    
    # Override with custom environment
    trainer.env = env
    
    results = trainer.train(
        progress_callback=lambda p: logger.debug(
            f"Stage {p['stage']}: {p['episode']}/{p['total_episodes']}"
        )
    )
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    # Logging setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*60)
    print("BANGALORE TRAFFIC RL TRAINING")
    print("="*60)
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Episodes: {args.episodes}")
    print(f"Mode: {args.mode}")
    print(f"Curriculum: {args.curriculum}")
    print(f"Output: {args.output}")
    print("="*60)
    
    # Create environment
    logger.info("Creating environment...")
    try:
        env = create_environment(args)
        logger.info(f"✓ Real SUMO environment created: obs={env.observation_space}, act={env.action_space}")
        using_real_sumo = True
    except Exception as e:
        logger.warning(f"Could not create SUMO environment: {e}")
        
        # Check if user wants to proceed with mock
        if args.allow_mock:
            logger.warning("⚠ Using MockTrafficEnv - FOR DEVELOPMENT ONLY, NOT REAL DATA!")
            from src.environment.mock_env import MockTrafficEnv
            env = MockTrafficEnv()
            using_real_sumo = False
        else:
            logger.error("✗ SUMO not available. Use --allow-mock flag for development mode.")
            return 1
    
    # Create agent
    logger.info(f"Creating {args.algorithm.upper()} agent...")
    agent = create_agent(args.algorithm, env, args)
    
    # Load checkpoint if specified
    if args.checkpoint:
        if hasattr(agent, 'load'):
            agent.load(args.checkpoint)
            logger.info(f"Loaded checkpoint: {args.checkpoint}")
    
    # Train
    if args.curriculum and args.algorithm != 'qlearning':
        # Curriculum learning (not suitable for Q-learning due to discretization)
        results = train_with_curriculum(env, type(agent), args)
    else:
        # Simple training
        results = train_simple(env, agent, args)
    
    # Cleanup
    env.close()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Models saved to: {args.output}")
    print("\nNext steps:")
    print("  python scripts/05_evaluate_models.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
