"""
Deep Q-Network (DQN) Agent
Neural network-based Q-learning for traffic signal control
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path

from ..utils.logger import setup_logger

logger = setup_logger("dqn_agent")

# Experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNNetwork(nn.Module):
    """
    Deep Q-Network architecture
    
    Multi-layer perceptron with configurable hidden layers
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layers: List[int] = [256, 256, 128],
        activation: str = 'relu'
    ):
        """
        Initialize DQN Network
        
        Args:
            state_size: Dimension of state space
            action_size: Number of discrete actions
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'leaky_relu')
        """
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Build network layers
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(state)


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN Network architecture
    
    Separates value and advantage streams for better learning
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layers: List[int] = [256, 256]
    ):
        super(DuelingDQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Shared feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_layers[1], 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_layers[1], 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass with value-advantage decomposition"""
        features = self.feature_layer(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_values


class ReplayBuffer:
    """
    Experience Replay Buffer
    
    Stores and samples experiences for training
    """
    
    def __init__(self, capacity: int = 100000, seed: int = 42):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum buffer size
            seed: Random seed
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        random.seed(seed)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add experience to buffer"""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences"""
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    
    Samples experiences based on TD error priority
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        seed: int = 42
    ):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
        random.seed(seed)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add experience with max priority"""
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities.append(max_priority)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, List[int]]:
        """Sample batch with importance sampling weights"""
        if len(self.buffer) == 0:
            return [], np.array([]), []
        
        priorities = np.array(list(self.priorities))
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probs, replace=False)
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        experiences = [self.buffer[i] for i in indices]
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, weights, list(indices)
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """Update priorities based on TD errors"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small constant to avoid zero
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent
    
    Features:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration
    - Optional: Double DQN, Dueling DQN, Prioritized Replay
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layers: List[int] = [256, 256, 128],
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        dueling: bool = False,
        double: bool = True,
        prioritized: bool = False,
        device: str = 'auto',
        seed: int = 42
    ):
        """
        Initialize DQN Agent
        
        Args:
            state_size: Dimension of state space
            action_size: Number of discrete actions
            hidden_layers: Hidden layer sizes
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
            buffer_size: Replay buffer size
            batch_size: Training batch size
            target_update_freq: Steps between target network updates
            dueling: Use Dueling DQN architecture
            double: Use Double DQN
            prioritized: Use Prioritized Experience Replay
            device: Device for PyTorch ('auto', 'cuda', 'cpu')
            seed: Random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double = double
        self.prioritized = prioritized
        
        # Device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
        
        # Networks
        NetworkClass = DuelingDQNNetwork if dueling else DQNNetwork
        
        self.policy_network = NetworkClass(state_size, action_size, hidden_layers).to(self.device)
        self.target_network = NetworkClass(state_size, action_size, hidden_layers).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        if prioritized:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size, seed=seed)
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_size, seed=seed)
        
        # Training stats
        self.step_count = 0
        self.episode_count = 0
        self.training_losses = []
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode
        
        Returns:
            Selected action
        """
        if training and self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.action_size)
        
        # Use network for action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step
        
        Returns:
            Loss value or None if not enough samples
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        if self.prioritized:
            experiences, weights, indices = self.replay_buffer.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            experiences = self.replay_buffer.sample(self.batch_size)
            weights = None
            indices = None
        
        # Convert to tensors
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)
        
        # Current Q values
        current_q = self.policy_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            if self.double:
                # Double DQN: use policy network to select actions, target network to evaluate
                next_actions = self.policy_network(next_states).argmax(dim=1)
                next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                # Standard DQN
                next_q = self.target_network(next_states).max(dim=1)[0]
            
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss
        td_errors = current_q.squeeze() - target_q
        
        if self.prioritized and weights is not None:
            loss = (weights * td_errors.pow(2)).mean()
            # Update priorities
            self.replay_buffer.update_priorities(indices, td_errors.abs().detach().cpu().numpy())
        else:
            loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
        
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        
        return loss_value
    
    def end_episode(self) -> None:
        """Called at end of episode"""
        self.episode_count += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0,
        }
    
    def save(self, path: str) -> None:
        """Save agent to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'state_size': self.state_size,
            'action_size': self.action_size,
        }
        
        torch.save(save_dict, path)
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: str) -> None:
        """Load agent from file"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        
        logger.info(f"Agent loaded from {path}")


def train_dqn(
    env,
    agent: DQNAgent,
    num_episodes: int = 500,
    max_steps: int = 3600,
    train_freq: int = 4,
    learning_starts: int = 1000,
    log_interval: int = 10
) -> Dict[str, list]:
    """
    Train DQN agent
    
    Args:
        env: Gymnasium environment
        agent: DQN agent
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        train_freq: Steps between training updates
        learning_starts: Steps before training starts
        log_interval: Episodes between logging
    
    Returns:
        Training history
    """
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'avg_waiting_times': [],
        'epsilons': [],
        'losses': [],
    }
    
    total_steps = 0
    
    logger.info(f"Starting DQN training for {num_episodes} episodes")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        for step in range(max_steps):
            # Select action
            action = agent.get_action(obs, training=True)
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.store_experience(obs, action, reward, next_obs, done)
            
            # Train
            total_steps += 1
            if total_steps > learning_starts and total_steps % train_freq == 0:
                loss = agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)
            
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
        history['losses'].append(np.mean(episode_losses) if episode_losses else 0)
        
        # Logging
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(history['episode_rewards'][-log_interval:])
            avg_wait = np.mean(history['avg_waiting_times'][-log_interval:])
            avg_loss = np.mean(history['losses'][-log_interval:])
            
            logger.info(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Avg Wait: {avg_wait:.1f}s | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Loss: {avg_loss:.4f}"
            )
    
    logger.info("DQN training complete")
    
    return history


if __name__ == "__main__":
    # Test DQN agent
    print("Testing DQN Agent...")
    print("=" * 50)
    
    # Create agent
    agent = DQNAgent(
        state_size=18,
        action_size=4,
        hidden_layers=[128, 128],
        buffer_size=1000,
        batch_size=32
    )
    
    print(f"Device: {agent.device}")
    print(f"Policy network: {agent.policy_network}")
    
    # Test with random experiences
    for i in range(200):
        state = np.random.rand(18).astype(np.float32)
        action = agent.get_action(state)
        next_state = np.random.rand(18).astype(np.float32)
        reward = np.random.randn()
        
        agent.store_experience(state, action, reward, next_state, done=False)
        
        if i > 50:
            loss = agent.train_step()
    
    agent.end_episode()
    
    print(f"\nAgent stats: {agent.get_stats()}")
    
    # Test save/load
    agent.save("test_dqn.pt")
    agent.load("test_dqn.pt")
    print(f"Loaded agent stats: {agent.get_stats()}")
    
    # Clean up
    import os
    os.remove("test_dqn.pt")
    
    print("\nDQN Agent test complete!")
