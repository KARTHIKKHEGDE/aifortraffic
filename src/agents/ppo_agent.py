"""
PPO Multi-Agent System
Proximal Policy Optimization for multi-junction traffic control
Uses Stable-Baselines3 with custom multi-agent wrapper
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces

from ..utils.logger import setup_logger

logger = setup_logger("ppo_agent")


# Try to import Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("Stable-Baselines3 not available. Install with: pip install stable-baselines3")


class TrafficFeaturesExtractor(nn.Module if not SB3_AVAILABLE else BaseFeaturesExtractor):
    """
    Custom feature extractor for traffic state
    
    Processes per-junction features and combines them
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        if SB3_AVAILABLE:
            super().__init__(observation_space, features_dim)
        else:
            super().__init__()
        
        self.features_dim = features_dim
        obs_size = observation_space.shape[0]
        
        # Feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.feature_net(observations)


class MultiAgentTrafficEnv(gym.Env):
    """
    Multi-Agent Traffic Environment Wrapper
    
    Wraps multiple junction environments into a single environment
    for centralized training with shared policy
    """
    
    def __init__(
        self,
        junction_ids: List[str],
        config: Dict[str, Any],
        shared_state: bool = True
    ):
        """
        Initialize multi-agent environment
        
        Args:
            junction_ids: List of junction IDs to control
            config: Environment configuration
            shared_state: Whether to share state information between agents
        """
        super().__init__()
        
        self.junction_ids = junction_ids
        self.num_agents = len(junction_ids)
        self.config = config
        self.shared_state = shared_state
        
        # State size per junction
        self.state_size_per_junction = config.get('state_size', 18)
        
        # Total state size (with neighbor info if shared)
        if shared_state:
            # Each junction sees own state + summary of neighbors
            self.neighbor_summary_size = 4  # Avg queue, density, phase, emergency flag
            total_state_size = self.state_size_per_junction + (self.num_agents - 1) * self.neighbor_summary_size
        else:
            total_state_size = self.state_size_per_junction
        
        # Multi-agent state (concatenated for all agents)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_agents * total_state_size,),
            dtype=np.float32
        )
        
        # Multi-agent action (one action per junction)
        self.action_size = config.get('action_size', 4)
        self.action_space = spaces.MultiDiscrete([self.action_size] * self.num_agents)
        
        # Lazy initialization
        self._base_env = None
    
    def _get_base_env(self):
        """Lazy initialization of base environment"""
        if self._base_env is None:
            # Import here to avoid circular imports
            from ..environment import BangaloreTrafficEnv
            self._base_env = BangaloreTrafficEnv(self.config)
        return self._base_env
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        base_env = self._get_base_env()
        obs, info = base_env.reset(seed=seed, options=options)
        
        # Get multi-agent observation
        multi_obs = self._get_multi_agent_obs(obs)
        
        return multi_obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step environment with multi-agent action
        
        Args:
            action: Array of actions, one per junction
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        base_env = self._get_base_env()
        
        # For now, use primary junction action
        # Full multi-junction control requires SUMO configuration with multiple TLS
        primary_action = int(action[0]) if isinstance(action, np.ndarray) else action
        
        obs, reward, terminated, truncated, info = base_env.step(primary_action)
        
        # Get multi-agent observation
        multi_obs = self._get_multi_agent_obs(obs)
        
        # Compute multi-agent reward
        multi_reward = self._compute_multi_agent_reward(reward, info)
        
        return multi_obs, multi_reward, terminated, truncated, info
    
    def _get_multi_agent_obs(self, single_obs: np.ndarray) -> np.ndarray:
        """Convert single observation to multi-agent observation"""
        # For now, replicate observation for all agents
        # In full implementation, each agent would have its own state from SUMO
        if self.shared_state:
            # Add neighbor summaries
            base_env = self._get_base_env()
            multi_obs = []
            
            for i in range(self.num_agents):
                # Own state
                agent_obs = single_obs.copy()
                
                # Neighbor summaries (simplified)
                for j in range(self.num_agents):
                    if i != j:
                        # Summary: avg_queue, avg_density, phase, emergency
                        summary = np.array([
                            np.mean(single_obs[0:4]),  # Avg queue
                            np.mean(single_obs[4:8]),  # Avg density
                            single_obs[12],            # Phase
                            single_obs[14]             # Emergency
                        ])
                        agent_obs = np.concatenate([agent_obs, summary])
                
                multi_obs.append(agent_obs)
            
            return np.concatenate(multi_obs).astype(np.float32)
        else:
            # Just replicate
            return np.tile(single_obs, self.num_agents).astype(np.float32)
    
    def _compute_multi_agent_reward(self, base_reward: float, info: Dict) -> float:
        """Compute multi-agent reward with coordination bonus"""
        # Base reward
        reward = base_reward
        
        # Coordination bonus (encourage synchronized behavior)
        # In full implementation, this would consider inter-junction queue balancing
        coordination_bonus = 0.0
        
        return reward + coordination_bonus
    
    def close(self):
        """Close environment"""
        if self._base_env is not None:
            self._base_env.close()
    
    def render(self):
        """Render environment"""
        if self._base_env is not None:
            return self._base_env.render()


class PPOTrainingCallback(BaseCallback if SB3_AVAILABLE else object):
    """
    Custom callback for PPO training
    
    Logs metrics and handles curriculum learning
    """
    
    def __init__(
        self,
        log_interval: int = 100,
        curriculum_stages: Optional[List[Dict]] = None,
        verbose: int = 1
    ):
        if SB3_AVAILABLE:
            super().__init__(verbose)
        
        self.log_interval = log_interval
        self.curriculum_stages = curriculum_stages or []
        self.current_stage = 0
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.waiting_times = []
    
    def _on_step(self) -> bool:
        """Called at each step"""
        # Log metrics
        if self.n_calls % self.log_interval == 0:
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_wait = np.mean(self.waiting_times[-100:]) if self.waiting_times else 0
                
                logger.info(
                    f"Step {self.n_calls} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Avg Wait: {avg_wait:.1f}s"
                )
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at end of rollout"""
        # Get episode info from buffer
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
            if 'avg_waiting_time' in info:
                self.waiting_times.append(info['avg_waiting_time'])


class PPOMultiAgent:
    """
    PPO Multi-Agent Controller
    
    Centralized training with shared policy for multiple junctions
    """
    
    def __init__(
        self,
        env: gym.Env,
        policy_type: str = 'MlpPolicy',
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy_kwargs: Optional[Dict] = None,
        device: str = 'auto',
        seed: int = 42,
        verbose: int = 1
    ):
        """
        Initialize PPO Multi-Agent
        
        Args:
            env: Gymnasium environment
            policy_type: Policy network type
            learning_rate: Learning rate
            n_steps: Steps per rollout
            batch_size: Minibatch size
            n_epochs: Epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm
            policy_kwargs: Additional policy arguments
            device: Device for PyTorch
            seed: Random seed
            verbose: Verbosity level
        """
        if not SB3_AVAILABLE:
            raise ImportError("Stable-Baselines3 required for PPO. Install with: pip install stable-baselines3")
        
        self.env = env
        self.seed = seed
        self.verbose = verbose
        
        # Wrap environment for SB3
        vec_env = DummyVecEnv([lambda: env])
        
        # Policy kwargs with custom feature extractor
        if policy_kwargs is None:
            policy_kwargs = {}
        
        # Create PPO model
        self.model = PPO(
            policy=policy_type,
            env=vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            device=device,
            seed=seed,
            verbose=verbose
        )
        
        logger.info(f"PPO Multi-Agent initialized with device: {self.model.device}")
    
    def train(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        log_interval: int = 10,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_path: Optional[str] = None,
        save_freq: int = 50000
    ) -> Dict[str, list]:
        """
        Train PPO agent
        
        Args:
            total_timesteps: Total training timesteps
            callback: Custom callback
            log_interval: Logging interval
            eval_env: Evaluation environment
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            save_path: Path to save checkpoints
            save_freq: Checkpoint save frequency
        
        Returns:
            Training history
        """
        callbacks = []
        
        # Add custom callback
        training_callback = PPOTrainingCallback(log_interval=100)
        callbacks.append(training_callback)
        
        # Add evaluation callback
        if eval_env is not None:
            eval_callback = EvalCallback(
                DummyVecEnv([lambda: eval_env]),
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True
            )
            callbacks.append(eval_callback)
        
        # Add checkpoint callback
        if save_path is not None:
            checkpoint_callback = CheckpointCallback(
                save_freq=save_freq,
                save_path=save_path,
                name_prefix='ppo_traffic'
            )
            callbacks.append(checkpoint_callback)
        
        # Add user callback
        if callback is not None:
            callbacks.append(callback)
        
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=log_interval
        )
        
        logger.info("PPO training complete")
        
        # Return training metrics
        return {
            'episode_rewards': training_callback.episode_rewards,
            'episode_lengths': training_callback.episode_lengths,
            'waiting_times': training_callback.waiting_times
        }
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict action for observation
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy
        
        Returns:
            action, states (None for non-recurrent)
        """
        action, states = self.model.predict(observation, deterministic=deterministic)
        return action, states
    
    def save(self, path: str) -> None:
        """Save model to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info(f"PPO model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model from file"""
        self.model = PPO.load(path, env=self.env)
        logger.info(f"PPO model loaded from {path}")
    
    @classmethod
    def from_pretrained(cls, path: str, env: gym.Env) -> 'PPOMultiAgent':
        """Load pretrained model"""
        instance = cls.__new__(cls)
        instance.env = env
        instance.model = PPO.load(path, env=DummyVecEnv([lambda: env]))
        return instance


class IndependentPPOAgents:
    """
    Independent PPO Agents for each junction
    
    Each junction has its own PPO policy (decentralized)
    """
    
    def __init__(
        self,
        junction_ids: List[str],
        state_size: int,
        action_size: int,
        **ppo_kwargs
    ):
        """
        Initialize independent agents
        
        Args:
            junction_ids: List of junction IDs
            state_size: State dimension per junction
            action_size: Number of actions
            **ppo_kwargs: Arguments for PPO
        """
        if not SB3_AVAILABLE:
            raise ImportError("Stable-Baselines3 required")
        
        self.junction_ids = junction_ids
        self.state_size = state_size
        self.action_size = action_size
        self.ppo_kwargs = ppo_kwargs
        
        # Create separate agent for each junction
        self.agents: Dict[str, Any] = {}
        
        # Agents are created lazily when environments are provided
    
    def create_agent(self, junction_id: str, env: gym.Env) -> None:
        """Create PPO agent for junction"""
        self.agents[junction_id] = PPO(
            policy='MlpPolicy',
            env=DummyVecEnv([lambda: env]),
            **self.ppo_kwargs
        )
    
    def predict(
        self,
        junction_id: str,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> int:
        """Predict action for specific junction"""
        if junction_id not in self.agents:
            raise ValueError(f"Agent for {junction_id} not created")
        
        action, _ = self.agents[junction_id].predict(observation, deterministic=deterministic)
        return int(action[0])
    
    def save_all(self, directory: str) -> None:
        """Save all agents"""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        for junction_id, agent in self.agents.items():
            agent.save(dir_path / f"ppo_{junction_id}.zip")
    
    def load_all(self, directory: str) -> None:
        """Load all agents"""
        dir_path = Path(directory)
        
        for junction_id in self.junction_ids:
            model_path = dir_path / f"ppo_{junction_id}.zip"
            if model_path.exists():
                self.agents[junction_id] = PPO.load(model_path)


def create_ppo_agent(
    env: gym.Env,
    config: Dict[str, Any]
) -> PPOMultiAgent:
    """
    Factory function to create PPO agent from config
    
    Args:
        env: Gymnasium environment
        config: Training configuration
    
    Returns:
        PPOMultiAgent instance
    """
    ppo_config = config.get('ppo', {})
    
    return PPOMultiAgent(
        env=env,
        learning_rate=ppo_config.get('learning_rate', 3e-4),
        n_steps=ppo_config.get('n_steps', 2048),
        batch_size=ppo_config.get('batch_size', 64),
        n_epochs=ppo_config.get('n_epochs', 10),
        gamma=ppo_config.get('gamma', 0.99),
        gae_lambda=ppo_config.get('gae_lambda', 0.95),
        clip_range=ppo_config.get('clip_range', 0.2),
        ent_coef=ppo_config.get('ent_coef', 0.01),
        device=config.get('device', 'auto'),
        seed=config.get('seed', 42)
    )


if __name__ == "__main__":
    print("Testing PPO Agent...")
    print("=" * 50)
    
    if not SB3_AVAILABLE:
        print("Stable-Baselines3 not available. Install with: pip install stable-baselines3")
    else:
        # Create simple test environment
        from gymnasium.envs.classic_control import CartPoleEnv
        
        test_env = CartPoleEnv()
        
        # Create PPO agent
        agent = PPOMultiAgent(
            env=test_env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=32,
            verbose=0
        )
        
        print(f"PPO model created with device: {agent.model.device}")
        
        # Quick training test
        agent.train(total_timesteps=1000, log_interval=10)
        
        # Test prediction
        obs, _ = test_env.reset()
        action, _ = agent.predict(obs)
        print(f"Test prediction: action = {action}")
        
        # Test save/load
        agent.save("test_ppo.zip")
        agent.load("test_ppo.zip")
        
        # Clean up
        import os
        os.remove("test_ppo.zip")
        
        print("\nPPO Agent test complete!")
