"""
Hyperparameter Tuning for Traffic Signal Control RL Agents

Supports:
- Grid search
- Random search
- Bayesian optimization (with optuna)
- Population-based training
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from itertools import product
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


@dataclass
class HyperparameterSpace:
    """Define hyperparameter search space"""
    
    # Learning parameters
    learning_rate: Tuple[float, float] = (1e-5, 1e-2)
    gamma: Tuple[float, float] = (0.9, 0.999)
    
    # Network architecture (for DQN/PPO)
    hidden_layers: List[List[int]] = field(default_factory=lambda: [
        [64, 64], [128, 128], [256, 256], [64, 64, 64], [128, 64]
    ])
    
    # Training parameters
    batch_size: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    buffer_size: List[int] = field(default_factory=lambda: [10000, 50000, 100000])
    
    # Exploration (for DQN)
    epsilon_start: Tuple[float, float] = (0.8, 1.0)
    epsilon_end: Tuple[float, float] = (0.01, 0.1)
    epsilon_decay: Tuple[float, float] = (0.99, 0.9999)
    
    # PPO specific
    clip_range: Tuple[float, float] = (0.1, 0.3)
    n_steps: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    n_epochs: List[int] = field(default_factory=lambda: [3, 5, 10])
    gae_lambda: Tuple[float, float] = (0.9, 0.99)
    
    # Reward shaping weights
    reward_waiting_weight: Tuple[float, float] = (-0.5, -0.01)
    reward_queue_weight: Tuple[float, float] = (-1.0, -0.1)
    reward_throughput_weight: Tuple[float, float] = (0.5, 2.0)


@dataclass
class TrialResult:
    """Result from a single hyperparameter trial"""
    trial_id: int
    params: Dict[str, Any]
    mean_reward: float
    std_reward: float
    best_reward: float
    training_time: float
    n_episodes: int
    converged: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HyperparameterTuner:
    """
    Hyperparameter tuning for RL agents
    
    Supports multiple search strategies and parallel evaluation.
    """
    
    def __init__(
        self,
        env_factory: Callable,
        agent_factory: Callable,
        search_space: Optional[HyperparameterSpace] = None,
        n_eval_episodes: int = 10,
        n_training_episodes: int = 100,
        output_dir: str = "tuning_results",
        n_parallel: int = 1
    ):
        """
        Initialize tuner
        
        Args:
            env_factory: Function that creates environment
            agent_factory: Function that creates agent given params
            search_space: Hyperparameter search space
            n_eval_episodes: Episodes for evaluation
            n_training_episodes: Episodes for training each trial
            output_dir: Directory for results
            n_parallel: Number of parallel trials
        """
        self.env_factory = env_factory
        self.agent_factory = agent_factory
        self.search_space = search_space or HyperparameterSpace()
        self.n_eval_episodes = n_eval_episodes
        self.n_training_episodes = n_training_episodes
        self.output_dir = Path(output_dir)
        self.n_parallel = n_parallel
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results tracking
        self.trials: List[TrialResult] = []
        self.best_trial: Optional[TrialResult] = None
    
    def sample_random_params(self) -> Dict[str, Any]:
        """Sample random hyperparameters from search space"""
        space = self.search_space
        
        params = {
            'learning_rate': np.random.uniform(*space.learning_rate),
            'gamma': np.random.uniform(*space.gamma),
            'hidden_layers': space.hidden_layers[np.random.randint(len(space.hidden_layers))],
            'batch_size': np.random.choice(space.batch_size),
            'buffer_size': np.random.choice(space.buffer_size),
            'epsilon_start': np.random.uniform(*space.epsilon_start),
            'epsilon_end': np.random.uniform(*space.epsilon_end),
            'epsilon_decay': np.random.uniform(*space.epsilon_decay),
            'clip_range': np.random.uniform(*space.clip_range),
            'n_steps': np.random.choice(space.n_steps),
            'n_epochs': np.random.choice(space.n_epochs),
            'gae_lambda': np.random.uniform(*space.gae_lambda),
        }
        
        return params
    
    def generate_grid_params(self, n_samples_per_param: int = 3) -> List[Dict[str, Any]]:
        """Generate grid of hyperparameters"""
        space = self.search_space
        
        # Sample continuous parameters
        lr_values = np.logspace(
            np.log10(space.learning_rate[0]),
            np.log10(space.learning_rate[1]),
            n_samples_per_param
        )
        gamma_values = np.linspace(space.gamma[0], space.gamma[1], n_samples_per_param)
        
        # Create grid
        grid = []
        for lr, gamma, hidden, batch in product(
            lr_values, gamma_values, 
            space.hidden_layers[:3],  # Limit combinations
            space.batch_size[:2]
        ):
            grid.append({
                'learning_rate': float(lr),
                'gamma': float(gamma),
                'hidden_layers': hidden,
                'batch_size': int(batch),
                'buffer_size': 50000,  # Fixed for grid
                'epsilon_start': 1.0,
                'epsilon_end': 0.05,
                'epsilon_decay': 0.995,
            })
        
        return grid
    
    def evaluate_params(
        self, 
        params: Dict[str, Any], 
        trial_id: int
    ) -> TrialResult:
        """
        Evaluate a single hyperparameter configuration
        
        Args:
            params: Hyperparameters to evaluate
            trial_id: Trial identifier
            
        Returns:
            TrialResult with performance metrics
        """
        start_time = time.time()
        
        # Create environment and agent
        env = self.env_factory()
        agent = self.agent_factory(params)
        
        # Training
        episode_rewards = []
        
        for episode in range(self.n_training_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Update agent
                if hasattr(agent, 'update'):
                    agent.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
            
            episode_rewards.append(total_reward)
        
        # Evaluation
        eval_rewards = []
        for _ in range(self.n_eval_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward
            
            eval_rewards.append(total_reward)
        
        training_time = time.time() - start_time
        
        # Check convergence
        if len(episode_rewards) >= 20:
            early_mean = np.mean(episode_rewards[:10])
            late_mean = np.mean(episode_rewards[-10:])
            converged = late_mean > early_mean * 1.1  # 10% improvement
        else:
            converged = False
        
        result = TrialResult(
            trial_id=trial_id,
            params=params,
            mean_reward=float(np.mean(eval_rewards)),
            std_reward=float(np.std(eval_rewards)),
            best_reward=float(max(eval_rewards)),
            training_time=training_time,
            n_episodes=self.n_training_episodes,
            converged=converged,
            metrics={
                'training_mean': float(np.mean(episode_rewards)),
                'training_final': float(np.mean(episode_rewards[-10:])),
                'improvement': float(np.mean(episode_rewards[-10:]) - np.mean(episode_rewards[:10]))
            }
        )
        
        env.close()
        
        return result
    
    def grid_search(self, n_samples_per_param: int = 3) -> TrialResult:
        """
        Perform grid search over hyperparameters
        
        Args:
            n_samples_per_param: Number of samples per continuous parameter
            
        Returns:
            Best trial result
        """
        print("Starting Grid Search...")
        
        param_grid = self.generate_grid_params(n_samples_per_param)
        print(f"Total configurations: {len(param_grid)}")
        
        for i, params in enumerate(param_grid):
            print(f"\nTrial {i+1}/{len(param_grid)}")
            print(f"Params: lr={params['learning_rate']:.6f}, gamma={params['gamma']:.4f}")
            
            result = self.evaluate_params(params, i)
            self.trials.append(result)
            
            print(f"Mean reward: {result.mean_reward:.2f} ± {result.std_reward:.2f}")
            
            if self.best_trial is None or result.mean_reward > self.best_trial.mean_reward:
                self.best_trial = result
                print("  *** New best! ***")
        
        self._save_results("grid_search")
        return self.best_trial
    
    def random_search(self, n_trials: int = 50) -> TrialResult:
        """
        Perform random search over hyperparameters
        
        Args:
            n_trials: Number of random configurations to try
            
        Returns:
            Best trial result
        """
        print(f"Starting Random Search ({n_trials} trials)...")
        
        for i in range(n_trials):
            params = self.sample_random_params()
            
            print(f"\nTrial {i+1}/{n_trials}")
            print(f"Params: lr={params['learning_rate']:.6f}, gamma={params['gamma']:.4f}")
            
            result = self.evaluate_params(params, i)
            self.trials.append(result)
            
            print(f"Mean reward: {result.mean_reward:.2f} ± {result.std_reward:.2f}")
            
            if self.best_trial is None or result.mean_reward > self.best_trial.mean_reward:
                self.best_trial = result
                print("  *** New best! ***")
        
        self._save_results("random_search")
        return self.best_trial
    
    def bayesian_optimization(
        self, 
        n_trials: int = 100,
        n_startup_trials: int = 10
    ) -> TrialResult:
        """
        Perform Bayesian optimization using Optuna
        
        Args:
            n_trials: Total number of trials
            n_startup_trials: Random trials before optimization
            
        Returns:
            Best trial result
        """
        if not HAS_OPTUNA:
            raise ImportError("optuna required for Bayesian optimization. Install with: pip install optuna")
        
        print(f"Starting Bayesian Optimization ({n_trials} trials)...")
        
        space = self.search_space
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                'learning_rate': trial.suggest_float('learning_rate', *space.learning_rate, log=True),
                'gamma': trial.suggest_float('gamma', *space.gamma),
                'hidden_layers': trial.suggest_categorical('hidden_layers', [str(h) for h in space.hidden_layers]),
                'batch_size': trial.suggest_categorical('batch_size', space.batch_size),
                'buffer_size': trial.suggest_categorical('buffer_size', space.buffer_size),
                'epsilon_start': trial.suggest_float('epsilon_start', *space.epsilon_start),
                'epsilon_end': trial.suggest_float('epsilon_end', *space.epsilon_end),
                'epsilon_decay': trial.suggest_float('epsilon_decay', *space.epsilon_decay),
            }
            
            # Convert hidden_layers back from string
            params['hidden_layers'] = eval(params['hidden_layers'])
            
            result = self.evaluate_params(params, trial.number)
            self.trials.append(result)
            
            if self.best_trial is None or result.mean_reward > self.best_trial.mean_reward:
                self.best_trial = result
            
            return result.mean_reward
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value:.2f}")
        print(f"Best params: {study.best_params}")
        
        self._save_results("bayesian_optimization")
        
        # Save optuna study
        study_path = self.output_dir / "optuna_study.pkl"
        import pickle
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        
        return self.best_trial
    
    def population_based_training(
        self,
        population_size: int = 8,
        n_generations: int = 10,
        exploit_fraction: float = 0.2,
        explore_fraction: float = 0.2
    ) -> TrialResult:
        """
        Population-Based Training (PBT)
        
        Evolves a population of agents, exploiting good performers
        and exploring new hyperparameters.
        
        Args:
            population_size: Number of agents in population
            n_generations: Number of generations
            exploit_fraction: Fraction of population to exploit from
            explore_fraction: Fraction to randomly perturb
            
        Returns:
            Best trial result
        """
        print(f"Starting Population-Based Training...")
        print(f"Population: {population_size}, Generations: {n_generations}")
        
        # Initialize population with random params
        population = []
        for i in range(population_size):
            params = self.sample_random_params()
            population.append({
                'params': params,
                'agent': None,  # Would be actual agent in full implementation
                'score': float('-inf'),
                'generation': 0
            })
        
        trial_id = 0
        
        for gen in range(n_generations):
            print(f"\n=== Generation {gen + 1}/{n_generations} ===")
            
            # Evaluate all members
            for i, member in enumerate(population):
                result = self.evaluate_params(member['params'], trial_id)
                member['score'] = result.mean_reward
                member['generation'] = gen
                self.trials.append(result)
                trial_id += 1
                
                print(f"  Member {i}: score={member['score']:.2f}")
            
            # Sort by performance
            population.sort(key=lambda x: x['score'], reverse=True)
            
            # Best of generation
            best_score = population[0]['score']
            print(f"  Best score: {best_score:.2f}")
            
            if self.best_trial is None or best_score > self.best_trial.mean_reward:
                self.best_trial = self.trials[-population_size]
            
            # Exploit and explore for bottom performers
            n_exploit = int(population_size * exploit_fraction)
            n_explore = int(population_size * explore_fraction)
            
            for i in range(population_size - n_exploit, population_size):
                # Exploit: copy params from top performer
                top_idx = np.random.randint(n_exploit)
                population[i]['params'] = population[top_idx]['params'].copy()
                
                # Explore: perturb some parameters
                if np.random.random() < explore_fraction:
                    population[i]['params'] = self._perturb_params(population[i]['params'])
        
        self._save_results("pbt")
        return self.best_trial
    
    def _perturb_params(
        self, 
        params: Dict[str, Any], 
        perturb_factor: float = 0.2
    ) -> Dict[str, Any]:
        """Randomly perturb parameters"""
        perturbed = params.copy()
        
        # Perturb continuous params
        for key in ['learning_rate', 'gamma', 'epsilon_decay']:
            if key in perturbed and np.random.random() < 0.5:
                factor = 1 + np.random.uniform(-perturb_factor, perturb_factor)
                perturbed[key] = perturbed[key] * factor
        
        # Clamp to valid ranges
        perturbed['learning_rate'] = np.clip(perturbed['learning_rate'], 1e-6, 1e-1)
        perturbed['gamma'] = np.clip(perturbed['gamma'], 0.8, 0.9999)
        
        return perturbed
    
    def _save_results(self, search_type: str):
        """Save tuning results to file"""
        results = {
            'search_type': search_type,
            'n_trials': len(self.trials),
            'best_trial': self.best_trial.to_dict() if self.best_trial else None,
            'all_trials': [t.to_dict() for t in self.trials]
        }
        
        output_file = self.output_dir / f"{search_type}_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best hyperparameters found"""
        if self.best_trial is None:
            return {}
        return self.best_trial.params
    
    def plot_results(self, output_file: str = "tuning_results.png"):
        """Visualize tuning results"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return
        
        if not self.trials:
            print("No trials to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Reward over trials
        ax1 = axes[0, 0]
        rewards = [t.mean_reward for t in self.trials]
        ax1.plot(rewards, 'b-', alpha=0.7)
        ax1.axhline(y=max(rewards), color='r', linestyle='--', label=f'Best: {max(rewards):.2f}')
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('Mean Reward')
        ax1.set_title('Reward over Trials')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Learning rate vs reward
        ax2 = axes[0, 1]
        lrs = [t.params.get('learning_rate', 0) for t in self.trials]
        ax2.scatter(lrs, rewards, alpha=0.6)
        ax2.set_xscale('log')
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('Mean Reward')
        ax2.set_title('Learning Rate vs Reward')
        ax2.grid(True, alpha=0.3)
        
        # 3. Gamma vs reward
        ax3 = axes[1, 0]
        gammas = [t.params.get('gamma', 0) for t in self.trials]
        ax3.scatter(gammas, rewards, alpha=0.6)
        ax3.set_xlabel('Gamma')
        ax3.set_ylabel('Mean Reward')
        ax3.set_title('Gamma vs Reward')
        ax3.grid(True, alpha=0.3)
        
        # 4. Training time distribution
        ax4 = axes[1, 1]
        times = [t.training_time for t in self.trials]
        ax4.hist(times, bins=20, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Training Time (s)')
        ax4.set_ylabel('Count')
        ax4.set_title('Training Time Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / output_file, dpi=300)
        print(f"Plot saved to {self.output_dir / output_file}")
        plt.close()


# Convenience function for quick tuning
def tune_dqn_agent(
    env_factory: Callable,
    n_trials: int = 50,
    method: str = 'random'
) -> Dict[str, Any]:
    """
    Quick hyperparameter tuning for DQN agent
    
    Args:
        env_factory: Function that creates environment
        n_trials: Number of trials
        method: 'random', 'grid', or 'bayesian'
        
    Returns:
        Best hyperparameters
    """
    from src.agents.dqn_agent import DQNAgent
    
    def agent_factory(params):
        return DQNAgent(
            state_dim=env_factory().observation_space.shape[0],
            action_dim=env_factory().action_space.n,
            learning_rate=params['learning_rate'],
            gamma=params['gamma'],
            epsilon=params['epsilon_start'],
            epsilon_min=params['epsilon_end'],
            epsilon_decay=params['epsilon_decay'],
            hidden_dims=params['hidden_layers'],
            buffer_size=params['buffer_size'],
            batch_size=params['batch_size']
        )
    
    tuner = HyperparameterTuner(
        env_factory=env_factory,
        agent_factory=agent_factory,
        n_training_episodes=100,
        n_eval_episodes=10
    )
    
    if method == 'random':
        tuner.random_search(n_trials)
    elif method == 'grid':
        tuner.grid_search()
    elif method == 'bayesian':
        tuner.bayesian_optimization(n_trials)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return tuner.get_best_params()


if __name__ == '__main__':
    # Example usage with mock environment
    from src.environment import MockTrafficEnv
    
    def env_factory():
        return MockTrafficEnv()
    
    def simple_agent_factory(params):
        """Create simple agent for testing"""
        class SimpleAgent:
            def __init__(self, params):
                self.lr = params.get('learning_rate', 0.01)
                self.gamma = params.get('gamma', 0.99)
            
            def select_action(self, state, evaluate=False):
                return np.random.randint(2)
            
            def update(self, *args):
                pass
        
        return SimpleAgent(params)
    
    tuner = HyperparameterTuner(
        env_factory=env_factory,
        agent_factory=simple_agent_factory,
        n_training_episodes=10,
        n_eval_episodes=5
    )
    
    best = tuner.random_search(n_trials=5)
    print(f"\nBest params: {best.params}")
    print(f"Best reward: {best.mean_reward:.2f}")
