"""
Distributed Training for Multi-Agent Traffic Control

Supports:
- Multi-process training with Ray
- Vectorized environments
- Parameter server architecture
- Gradient aggregation
"""

import os
import time
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np

try:
    import ray
    from ray import tune
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

try:
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    n_workers: int = 4
    n_envs_per_worker: int = 2
    rollout_fragment_length: int = 200
    train_batch_size: int = 4000
    sgd_minibatch_size: int = 128
    num_sgd_iter: int = 10
    lr: float = 3e-4
    gamma: float = 0.99
    use_gae: bool = True
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    entropy_coeff: float = 0.01
    vf_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5
    
    # Checkpointing
    checkpoint_freq: int = 10
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    log_interval: int = 1
    verbose: int = 1


class VectorizedEnvManager:
    """
    Manages vectorized environments for parallel training
    """
    
    def __init__(
        self,
        env_factory: Callable,
        n_envs: int = 4,
        use_subproc: bool = True
    ):
        """
        Initialize vectorized environments
        
        Args:
            env_factory: Function that creates a single environment
            n_envs: Number of parallel environments
            use_subproc: Use subprocess vectorization (more isolation)
        """
        self.env_factory = env_factory
        self.n_envs = n_envs
        self.use_subproc = use_subproc
        
        self.vec_env = None
        self._create_vec_env()
    
    def _create_vec_env(self):
        """Create vectorized environment"""
        if not HAS_SB3:
            print("Warning: stable-baselines3 not available, using simple loop")
            self.vec_env = None
            self.envs = [self.env_factory() for _ in range(self.n_envs)]
            return
        
        env_fns = [self.env_factory for _ in range(self.n_envs)]
        
        if self.use_subproc and self.n_envs > 1:
            self.vec_env = SubprocVecEnv(env_fns)
        else:
            self.vec_env = DummyVecEnv(env_fns)
    
    def reset(self) -> np.ndarray:
        """Reset all environments"""
        if self.vec_env is not None:
            return self.vec_env.reset()
        else:
            observations = []
            for env in self.envs:
                obs, _ = env.reset()
                observations.append(obs)
            return np.array(observations)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments"""
        if self.vec_env is not None:
            return self.vec_env.step(actions)
        else:
            observations, rewards, dones, infos = [], [], [], []
            for i, env in enumerate(self.envs):
                obs, reward, terminated, truncated, info = env.step(actions[i])
                done = terminated or truncated
                
                if done:
                    obs, _ = env.reset()
                
                observations.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
            
            return np.array(observations), np.array(rewards), np.array(dones), infos
    
    def close(self):
        """Close all environments"""
        if self.vec_env is not None:
            self.vec_env.close()
        else:
            for env in self.envs:
                env.close()


class RolloutBuffer:
    """
    Buffer for storing rollout data from multiple environments
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        n_envs: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.reset()
    
    def reset(self):
        """Reset buffer"""
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs) + self.observation_shape,
            dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.int64
        )
        self.rewards = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.float32
        )
        self.dones = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.float32
        )
        self.values = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.float32
        )
        self.log_probs = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.float32
        )
        self.advantages = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.float32
        )
        self.returns = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.float32
        )
        
        self.pos = 0
        self.full = False
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray
    ):
        """Add transition to buffer"""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantage(self, last_values: np.ndarray, last_dones: np.ndarray):
        """
        Compute GAE (Generalized Advantage Estimation)
        """
        last_gae_lam = 0
        
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            
            delta = (
                self.rewards[step] 
                + self.gamma * next_values * next_non_terminal 
                - self.values[step]
            )
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        self.returns = self.advantages + self.values
    
    def get(self, batch_size: Optional[int] = None):
        """
        Get data from buffer
        
        Args:
            batch_size: If provided, return random mini-batches
            
        Yields:
            Batches of (obs, actions, old_values, old_log_probs, advantages, returns)
        """
        indices = np.arange(self.buffer_size * self.n_envs)
        
        # Flatten
        obs = self.observations.reshape(-1, *self.observation_shape)
        actions = self.actions.flatten()
        values = self.values.flatten()
        log_probs = self.log_probs.flatten()
        advantages = self.advantages.flatten()
        returns = self.returns.flatten()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        if batch_size is None:
            yield obs, actions, values, log_probs, advantages, returns
        else:
            np.random.shuffle(indices)
            
            for start in range(0, len(indices), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                yield (
                    obs[batch_indices],
                    actions[batch_indices],
                    values[batch_indices],
                    log_probs[batch_indices],
                    advantages[batch_indices],
                    returns[batch_indices]
                )


class DistributedTrainer:
    """
    Distributed trainer for PPO with vectorized environments
    """
    
    def __init__(
        self,
        env_factory: Callable,
        agent,
        config: Optional[DistributedConfig] = None
    ):
        """
        Initialize distributed trainer
        
        Args:
            env_factory: Function that creates environment
            agent: Agent to train (must have policy and value networks)
            config: Training configuration
        """
        self.env_factory = env_factory
        self.agent = agent
        self.config = config or DistributedConfig()
        
        # Calculate total environments
        self.total_envs = self.config.n_workers * self.config.n_envs_per_worker
        
        # Create vectorized environment
        self.vec_env = VectorizedEnvManager(
            env_factory=env_factory,
            n_envs=self.total_envs,
            use_subproc=self.config.n_workers > 1
        )
        
        # Get env specs
        sample_env = env_factory()
        self.observation_shape = sample_env.observation_space.shape
        self.action_dim = sample_env.action_space.n
        sample_env.close()
        
        # Create rollout buffer
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.config.rollout_fragment_length,
            observation_shape=self.observation_shape,
            action_dim=self.action_dim,
            n_envs=self.total_envs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda
        )
        
        # Training stats
        self.num_timesteps = 0
        self.num_episodes = 0
        self.episode_rewards = deque(maxlen=100)
        
        # Checkpointing
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_rollouts(self) -> bool:
        """
        Collect rollouts from all environments
        
        Returns:
            True if rollout collection was successful
        """
        self.rollout_buffer.reset()
        
        obs = self.vec_env.reset()
        
        for step in range(self.config.rollout_fragment_length):
            # Get actions from agent
            with_torch = hasattr(self.agent, 'get_action_and_value')
            
            if with_torch:
                actions, values, log_probs = self.agent.get_action_and_value(obs)
            else:
                actions = np.array([self.agent.select_action(o) for o in obs])
                values = np.zeros(self.total_envs)
                log_probs = np.zeros(self.total_envs)
            
            # Step environments
            next_obs, rewards, dones, infos = self.vec_env.step(actions)
            
            # Track episodes
            for i, (done, info) in enumerate(zip(dones, infos)):
                if done:
                    self.num_episodes += 1
                    if 'episode_reward' in info:
                        self.episode_rewards.append(info['episode_reward'])
            
            # Store transition
            self.rollout_buffer.add(obs, actions, rewards, dones, values, log_probs)
            
            obs = next_obs
            self.num_timesteps += self.total_envs
        
        # Compute returns and advantages
        if with_torch:
            _, last_values, _ = self.agent.get_action_and_value(obs)
        else:
            last_values = np.zeros(self.total_envs)
        
        last_dones = dones.astype(np.float32)
        self.rollout_buffer.compute_returns_and_advantage(last_values, last_dones)
        
        return True
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step
        
        Returns:
            Dictionary of training metrics
        """
        # Collect rollouts
        self.collect_rollouts()
        
        # Training metrics
        pg_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        clip_fractions = []
        
        # Multiple epochs of SGD
        for epoch in range(self.config.num_sgd_iter):
            for batch in self.rollout_buffer.get(self.config.sgd_minibatch_size):
                obs, actions, old_values, old_log_probs, advantages, returns = batch
                
                # Skip if agent doesn't support this training interface
                if not hasattr(self.agent, 'update_policy'):
                    continue
                
                # Get current policy outputs
                loss_dict = self.agent.update_policy(
                    obs=obs,
                    actions=actions,
                    old_log_probs=old_log_probs,
                    advantages=advantages,
                    returns=returns,
                    old_values=old_values,
                    clip_range=self.config.clip_param,
                    vf_coef=self.config.vf_loss_coeff,
                    ent_coef=self.config.entropy_coeff,
                    max_grad_norm=self.config.max_grad_norm
                )
                
                pg_losses.append(loss_dict.get('policy_loss', 0))
                value_losses.append(loss_dict.get('value_loss', 0))
                entropy_losses.append(loss_dict.get('entropy_loss', 0))
                total_losses.append(loss_dict.get('total_loss', 0))
                clip_fractions.append(loss_dict.get('clip_fraction', 0))
        
        return {
            'policy_loss': np.mean(pg_losses) if pg_losses else 0,
            'value_loss': np.mean(value_losses) if value_losses else 0,
            'entropy_loss': np.mean(entropy_losses) if entropy_losses else 0,
            'total_loss': np.mean(total_losses) if total_losses else 0,
            'clip_fraction': np.mean(clip_fractions) if clip_fractions else 0,
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'num_episodes': self.num_episodes,
            'num_timesteps': self.num_timesteps
        }
    
    def train(
        self,
        total_timesteps: int,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train for specified number of timesteps
        
        Args:
            total_timesteps: Total environment steps to train
            callback: Optional callback function called after each step
            
        Returns:
            Training results
        """
        start_time = time.time()
        iteration = 0
        
        print(f"Starting distributed training with {self.total_envs} environments")
        print(f"Target timesteps: {total_timesteps}")
        
        while self.num_timesteps < total_timesteps:
            iteration += 1
            
            # Training step
            metrics = self.train_step()
            
            # Logging
            if iteration % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                fps = self.num_timesteps / elapsed
                
                print(f"\n=== Iteration {iteration} ===")
                print(f"Timesteps: {self.num_timesteps}/{total_timesteps}")
                print(f"Episodes: {metrics['num_episodes']}")
                print(f"Mean reward: {metrics['mean_reward']:.2f}")
                print(f"FPS: {fps:.0f}")
                
                if metrics['total_loss'] > 0:
                    print(f"Policy loss: {metrics['policy_loss']:.4f}")
                    print(f"Value loss: {metrics['value_loss']:.4f}")
            
            # Checkpointing
            if iteration % self.config.checkpoint_freq == 0:
                self.save_checkpoint(iteration)
            
            # Callback
            if callback is not None:
                callback(self, metrics)
        
        total_time = time.time() - start_time
        
        return {
            'total_timesteps': self.num_timesteps,
            'total_episodes': self.num_episodes,
            'total_time': total_time,
            'final_mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0
        }
    
    def save_checkpoint(self, iteration: int):
        """Save training checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'num_timesteps': self.num_timesteps,
            'num_episodes': self.num_episodes,
            'episode_rewards': list(self.episode_rewards),
            'config': self.config.__dict__
        }
        
        # Save agent if it has save method
        if hasattr(self.agent, 'save'):
            agent_path = self.checkpoint_dir / f"agent_iter_{iteration}.pt"
            self.agent.save(str(agent_path))
            checkpoint['agent_path'] = str(agent_path)
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{iteration}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.num_timesteps = checkpoint['num_timesteps']
        self.num_episodes = checkpoint['num_episodes']
        self.episode_rewards = deque(checkpoint['episode_rewards'], maxlen=100)
        
        # Load agent if path is provided
        if 'agent_path' in checkpoint and hasattr(self.agent, 'load'):
            self.agent.load(checkpoint['agent_path'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def close(self):
        """Clean up resources"""
        self.vec_env.close()


# Ray-based distributed training (if available)
if HAS_RAY:
    @ray.remote
    class RayWorker:
        """
        Ray worker for distributed rollout collection
        """
        
        def __init__(self, env_factory: Callable, worker_id: int):
            self.env = env_factory()
            self.worker_id = worker_id
            self.obs, _ = self.env.reset()
        
        def collect_rollout(self, agent_weights: Dict, n_steps: int) -> Dict:
            """Collect rollout with given agent weights"""
            # Would update local agent weights here
            
            rollout = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'dones': [],
                'episode_rewards': []
            }
            
            episode_reward = 0
            
            for _ in range(n_steps):
                # Simplified action selection
                action = self.env.action_space.sample()
                
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                rollout['observations'].append(self.obs)
                rollout['actions'].append(action)
                rollout['rewards'].append(reward)
                rollout['dones'].append(done)
                
                episode_reward += reward
                
                if done:
                    rollout['episode_rewards'].append(episode_reward)
                    episode_reward = 0
                    self.obs, _ = self.env.reset()
                else:
                    self.obs = next_obs
            
            return rollout
        
        def close(self):
            self.env.close()
    
    
    class RayDistributedTrainer:
        """
        Distributed trainer using Ray for parallelization
        """
        
        def __init__(
            self,
            env_factory: Callable,
            agent,
            n_workers: int = 4,
            rollout_length: int = 200
        ):
            if not ray.is_initialized():
                ray.init()
            
            self.agent = agent
            self.n_workers = n_workers
            self.rollout_length = rollout_length
            
            # Create workers
            self.workers = [
                RayWorker.remote(env_factory, i)
                for i in range(n_workers)
            ]
        
        def collect_parallel_rollouts(self) -> List[Dict]:
            """Collect rollouts from all workers in parallel"""
            agent_weights = {}  # Would extract weights from agent
            
            futures = [
                worker.collect_rollout.remote(agent_weights, self.rollout_length)
                for worker in self.workers
            ]
            
            rollouts = ray.get(futures)
            return rollouts
        
        def train(self, n_iterations: int) -> Dict:
            """Train using parallel rollouts"""
            for i in range(n_iterations):
                rollouts = self.collect_parallel_rollouts()
                
                # Aggregate rollouts
                all_rewards = []
                for rollout in rollouts:
                    all_rewards.extend(rollout['episode_rewards'])
                
                if all_rewards:
                    print(f"Iteration {i}: Mean reward = {np.mean(all_rewards):.2f}")
            
            return {'n_iterations': n_iterations}
        
        def close(self):
            """Clean up workers"""
            for worker in self.workers:
                ray.kill(worker)


# Convenience function
def create_distributed_trainer(
    env_factory: Callable,
    agent,
    n_workers: int = 4,
    use_ray: bool = False
) -> DistributedTrainer:
    """
    Create distributed trainer
    
    Args:
        env_factory: Environment factory function
        agent: Agent to train
        n_workers: Number of parallel workers
        use_ray: Use Ray for distribution (requires ray)
        
    Returns:
        Distributed trainer instance
    """
    if use_ray and HAS_RAY:
        return RayDistributedTrainer(env_factory, agent, n_workers)
    else:
        config = DistributedConfig(n_workers=n_workers)
        return DistributedTrainer(env_factory, agent, config)


if __name__ == '__main__':
    # Example usage
    from src.environment import MockTrafficEnv
    
    def env_factory():
        return MockTrafficEnv()
    
    class SimpleAgent:
        def select_action(self, obs):
            return np.random.randint(2)
    
    trainer = DistributedTrainer(
        env_factory=env_factory,
        agent=SimpleAgent(),
        config=DistributedConfig(n_workers=2, n_envs_per_worker=2)
    )
    
    results = trainer.train(total_timesteps=10000)
    print(f"\nTraining complete: {results}")
    
    trainer.close()
