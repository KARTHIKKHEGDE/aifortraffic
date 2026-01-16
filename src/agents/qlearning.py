"""
Q-Learning Agent
Tabular Q-Learning for single junction traffic control (proof of concept)
"""

import numpy as np
import pickle
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

from ..utils.logger import setup_logger

logger = setup_logger("qlearning")


class QLearningAgent:
    """
    Tabular Q-Learning Agent for Traffic Signal Control
    
    This is a baseline agent for proof-of-concept on single junctions.
    Uses state discretization to handle continuous state space.
    
    Features:
    - Epsilon-greedy exploration
    - State discretization
    - Learning rate decay
    - Save/load functionality
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        discretization_bins: int = 10,
        seed: int = 42
    ):
        """
        Initialize Q-Learning Agent
        
        Args:
            state_size: Dimension of state space
            action_size: Number of discrete actions
            learning_rate: Alpha - learning rate
            discount_factor: Gamma - discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate per episode
            epsilon_min: Minimum exploration rate
            discretization_bins: Number of bins for state discretization
            seed: Random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.discretization_bins = discretization_bins
        
        # Q-table: maps discretized state to action values
        self.q_table: Dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(action_size)
        )
        
        # Random number generator
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
        
        # Training statistics
        self.episode_count = 0
        self.update_count = 0
        self.total_reward = 0.0
        
        logger.info(
            f"Q-Learning Agent initialized: "
            f"state_size={state_size}, action_size={action_size}, "
            f"bins={discretization_bins}"
        )
    
    def _discretize_state(self, state: np.ndarray) -> tuple:
        """
        Discretize continuous state into bins
        
        Args:
            state: Continuous state vector
        
        Returns:
            Tuple of discretized values
        """
        # Clip state values to [0, 1] range (assuming normalized state)
        clipped = np.clip(state, 0.0, 1.0)
        
        # Discretize to bins
        discretized = (clipped * (self.discretization_bins - 1)).astype(int)
        
        return tuple(discretized)
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (use epsilon-greedy)
        
        Returns:
            Selected action
        """
        if training and self.rng.random() < self.epsilon:
            # Exploration: random action
            return self.rng.randint(0, self.action_size)
        
        # Exploitation: best action from Q-table
        state_key = self._discretize_state(state)
        q_values = self.q_table[state_key]
        
        # Break ties randomly
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        
        return self.rng.choice(best_actions)
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> float:
        """
        Update Q-value using TD learning
        
        Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        
        Returns:
            TD error
        """
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Target Q-value
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.discount_factor * max_next_q
        
        # TD error
        td_error = target_q - current_q
        
        # Update Q-value
        self.q_table[state_key][action] += self.learning_rate * td_error
        
        self.update_count += 1
        self.total_reward += reward
        
        return td_error
    
    def end_episode(self) -> None:
        """Called at the end of each episode"""
        self.episode_count += 1
        
        # Decay epsilon
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )
        
        logger.debug(
            f"Episode {self.episode_count}: "
            f"epsilon={self.epsilon:.4f}, "
            f"q_table_size={len(self.q_table)}"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'avg_q_value': np.mean([np.mean(v) for v in self.q_table.values()]) if self.q_table else 0,
        }
    
    def save(self, path: str) -> None:
        """
        Save agent to file
        
        Args:
            path: Path to save file
        """
        save_data = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'discretization_bins': self.discretization_bins,
            'q_table': dict(self.q_table),
            'episode_count': self.episode_count,
            'update_count': self.update_count,
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Agent saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'QLearningAgent':
        """
        Load agent from file
        
        Args:
            path: Path to saved file
        
        Returns:
            Loaded agent
        """
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        agent = cls(
            state_size=save_data['state_size'],
            action_size=save_data['action_size'],
            learning_rate=save_data['learning_rate'],
            discount_factor=save_data['discount_factor'],
            epsilon=save_data['epsilon'],
            epsilon_decay=save_data['epsilon_decay'],
            epsilon_min=save_data['epsilon_min'],
            discretization_bins=save_data['discretization_bins'],
        )
        
        agent.q_table = defaultdict(
            lambda: np.zeros(agent.action_size),
            save_data['q_table']
        )
        agent.episode_count = save_data['episode_count']
        agent.update_count = save_data['update_count']
        
        logger.info(f"Agent loaded from {path}")
        
        return agent


def train_qlearning(
    env,
    agent: QLearningAgent,
    num_episodes: int = 500,
    max_steps: int = 3600,
    log_interval: int = 10
) -> Dict[str, list]:
    """
    Train Q-Learning agent
    
    Args:
        env: Gymnasium environment
        agent: Q-Learning agent
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        log_interval: Episode interval for logging
    
    Returns:
        Training history dictionary
    """
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'avg_waiting_times': [],
        'epsilons': [],
    }
    
    logger.info(f"Starting Q-Learning training for {num_episodes} episodes")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.get_action(obs, training=True)
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update Q-values
            agent.update(obs, action, reward, next_obs, done)
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            if done:
                break
        
        # End episode
        agent.end_episode()
        
        # Record history
        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(episode_length)
        history['avg_waiting_times'].append(info.get('avg_waiting_time', 0))
        history['epsilons'].append(agent.epsilon)
        
        # Logging
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(history['episode_rewards'][-log_interval:])
            avg_wait = np.mean(history['avg_waiting_times'][-log_interval:])
            
            logger.info(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Avg Wait: {avg_wait:.1f}s | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Q-table: {len(agent.q_table)} states"
            )
    
    logger.info("Q-Learning training complete")
    
    return history


if __name__ == "__main__":
    # Test Q-Learning agent
    print("Testing Q-Learning Agent...")
    print("=" * 50)
    
    # Create agent
    agent = QLearningAgent(
        state_size=18,
        action_size=4,
        discretization_bins=10
    )
    
    # Test with random states
    for i in range(100):
        state = np.random.rand(18)
        action = agent.get_action(state)
        next_state = np.random.rand(18)
        reward = np.random.randn()
        
        td_error = agent.update(state, action, reward, next_state, done=False)
    
    agent.end_episode()
    
    print(f"Agent stats: {agent.get_stats()}")
    
    # Test save/load
    agent.save("test_qlearning.pkl")
    loaded_agent = QLearningAgent.load("test_qlearning.pkl")
    print(f"Loaded agent stats: {loaded_agent.get_stats()}")
    
    # Clean up
    import os
    os.remove("test_qlearning.pkl")
    
    print("\nQ-Learning Agent test complete!")
