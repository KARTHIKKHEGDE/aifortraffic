#!/usr/bin/env python3
"""
Model Evaluation Script
Evaluates trained RL agents against baselines and generates comparison reports.

Features:
- Compare RL agents vs Fixed-Time vs Actuated controllers
- Evaluate on multiple traffic scenarios
- Generate performance reports and visualizations
- Calculate improvement percentages

Usage:
    python 05_evaluate_models.py --model models/dqn_final.pt
    python 05_evaluate_models.py --compare-all
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger
from src.evaluation.metrics import TrafficMetrics
from src.evaluation.baselines import FixedTimeController, ActuatedController

logger = setup_logger("evaluation")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate traffic control models"
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--algorithm', '-a',
        type=str,
        choices=['qlearning', 'dqn', 'ppo'],
        default='dqn',
        help='Algorithm type (default: dqn)'
    )
    
    parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=10,
        help='Number of evaluation episodes (default: 10)'
    )
    
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='Compare RL agent with all baselines'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['baseline', 'realistic_bangalore', 'all'],
        default='realistic_bangalore',
        help='Traffic mode for evaluation'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


class ModelEvaluator:
    """
    Comprehensive model evaluation framework.
    """
    
    def __init__(
        self,
        env,
        output_dir: str = "results",
        seed: int = 42
    ):
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        self.results: Dict[str, Any] = {}
    
    def evaluate_agent(
        self,
        agent,
        agent_name: str,
        num_episodes: int = 10,
        deterministic: bool = True
    ) -> TrafficMetrics:
        """
        Evaluate a single agent.
        
        Args:
            agent: Agent to evaluate
            agent_name: Name for logging
            num_episodes: Number of evaluation episodes
            deterministic: Use deterministic actions
        
        Returns:
            TrafficMetrics with evaluation results
        """
        logger.info(f"Evaluating: {agent_name} ({num_episodes} episodes)")
        
        metrics = TrafficMetrics()
        
        episode_rewards = []
        episode_queues = []
        episode_waiting = []
        emergency_clearances = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset(seed=self.seed + episode)
            
            episode_reward = 0
            episode_max_queue = 0
            episode_total_waiting = 0
            phase_switches = 0
            emergency_count = 0
            emergency_cleared = 0
            last_action = None
            done = False
            
            step = 0
            while not done:
                # Get action
                if hasattr(agent, 'get_action'):
                    action = agent.get_action(obs, training=not deterministic)
                elif hasattr(agent, 'predict'):
                    action, _ = agent.predict(obs, deterministic=deterministic)
                    action = int(action) if hasattr(action, '__int__') else action
                else:
                    action = agent(obs)  # Callable baseline
                
                # Track phase switches
                if last_action is not None and action != last_action:
                    phase_switches += 1
                last_action = action
                
                # Step
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Accumulate metrics
                episode_reward += reward
                
                if 'metrics' in info:
                    queue = info['metrics'].get('total_queue_length', 0)
                    episode_max_queue = max(episode_max_queue, queue)
                    episode_total_waiting += info['metrics'].get('total_waiting_time', 0)
                
                if info.get('emergency_active', False):
                    emergency_count += 1
                if info.get('emergency_cleared', False):
                    emergency_cleared += 1
                    if 'emergency_clearance_time' in info:
                        emergency_clearances.append(info['emergency_clearance_time'])
                
                # Add step metrics
                metrics.add_step_metrics(
                    queue_lengths=info.get('queue_lengths', {'avg': queue if 'metrics' in info else 0}),
                    waiting_times=info.get('waiting_times', {'avg': episode_total_waiting / max(step, 1)}),
                    speeds=info.get('speeds', {'avg': 10.0}),
                )
                
                obs = next_obs
                step += 1
            
            # Episode complete
            episode_rewards.append(episode_reward)
            episode_queues.append(episode_max_queue)
            episode_waiting.append(episode_total_waiting)
            
            metrics.add_episode_metrics(
                episode_length=step,
                episode_reward=episode_reward,
                phase_switches=phase_switches,
                emergency_count=emergency_count,
                emergency_cleared=emergency_cleared
            )
            
            logger.debug(
                f"  Episode {episode + 1}: "
                f"Reward={episode_reward:.1f}, "
                f"MaxQueue={episode_max_queue:.0f}"
            )
        
        # Store summary
        summary = metrics.get_summary()
        summary['agent_name'] = agent_name
        summary['num_episodes'] = num_episodes
        summary['avg_episode_reward'] = np.mean(episode_rewards)
        summary['std_episode_reward'] = np.std(episode_rewards)
        summary['avg_max_queue'] = np.mean(episode_queues)
        summary['avg_total_waiting'] = np.mean(episode_waiting)
        
        if emergency_clearances:
            summary['avg_emergency_clearance'] = np.mean(emergency_clearances)
        
        self.results[agent_name] = summary
        
        logger.info(
            f"  Results: Reward={summary['avg_episode_reward']:.1f}±{summary['std_episode_reward']:.1f}, "
            f"Queue={summary['avg_max_queue']:.1f}, "
            f"Waiting={summary['avg_total_waiting']:.1f}"
        )
        
        return metrics
    
    def compare_with_baselines(
        self,
        rl_agent,
        rl_name: str = "RL Agent",
        num_episodes: int = 10
    ) -> Dict[str, Any]:
        """
        Compare RL agent with baseline controllers.
        
        Args:
            rl_agent: Trained RL agent
            rl_name: Name for RL agent
            num_episodes: Episodes per evaluation
        
        Returns:
            Comparison results
        """
        logger.info("="*60)
        logger.info("BASELINE COMPARISON")
        logger.info("="*60)
        
        # Evaluate RL agent
        rl_metrics = self.evaluate_agent(rl_agent, rl_name, num_episodes)
        
        # Evaluate Fixed-Time controller
        fixed_controller = FixedTimeController(
            phase_durations=[30, 3, 30, 3],  # Typical fixed timing
            cycle_length=66
        )
        fixed_metrics = self.evaluate_agent(fixed_controller, "Fixed-Time", num_episodes)
        
        # Evaluate Actuated controller
        actuated_controller = ActuatedController(
            min_green=10,
            max_green=60,
            extension=3
        )
        actuated_metrics = self.evaluate_agent(actuated_controller, "Actuated", num_episodes)
        
        # Calculate improvements
        comparison = self._calculate_improvements(rl_name)
        
        return comparison
    
    def _calculate_improvements(self, rl_name: str) -> Dict[str, Any]:
        """Calculate improvement percentages vs baselines."""
        rl_results = self.results.get(rl_name, {})
        fixed_results = self.results.get("Fixed-Time", {})
        actuated_results = self.results.get("Actuated", {})
        
        improvements = {
            'vs_fixed_time': {},
            'vs_actuated': {},
        }
        
        # Metrics where lower is better
        lower_better = ['avg_max_queue', 'avg_total_waiting', 'avg_waiting_time']
        
        for metric in lower_better:
            rl_val = rl_results.get(metric, 0)
            fixed_val = fixed_results.get(metric, 1)
            actuated_val = actuated_results.get(metric, 1)
            
            if fixed_val > 0:
                improvements['vs_fixed_time'][metric] = (
                    (fixed_val - rl_val) / fixed_val * 100
                )
            
            if actuated_val > 0:
                improvements['vs_actuated'][metric] = (
                    (actuated_val - rl_val) / actuated_val * 100
                )
        
        # Metrics where higher is better
        higher_better = ['avg_episode_reward', 'throughput']
        
        for metric in higher_better:
            rl_val = rl_results.get(metric, 0)
            fixed_val = fixed_results.get(metric, 1)
            actuated_val = actuated_results.get(metric, 1)
            
            if abs(fixed_val) > 0:
                improvements['vs_fixed_time'][metric] = (
                    (rl_val - fixed_val) / abs(fixed_val) * 100
                )
            
            if abs(actuated_val) > 0:
                improvements['vs_actuated'][metric] = (
                    (rl_val - actuated_val) / abs(actuated_val) * 100
                )
        
        return improvements
    
    def generate_report(self) -> str:
        """Generate text report of results."""
        lines = [
            "="*70,
            "EVALUATION REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "="*70,
            ""
        ]
        
        # Individual results
        for agent_name, results in self.results.items():
            lines.append(f"Agent: {agent_name}")
            lines.append("-"*40)
            for key, value in results.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.2f}")
                else:
                    lines.append(f"  {key}: {value}")
            lines.append("")
        
        # Improvements
        if len(self.results) > 1:
            lines.append("="*70)
            lines.append("IMPROVEMENTS")
            lines.append("="*70)
            
            rl_names = [n for n in self.results.keys() if n not in ['Fixed-Time', 'Actuated']]
            for rl_name in rl_names:
                improvements = self._calculate_improvements(rl_name)
                
                lines.append(f"\n{rl_name} vs Fixed-Time:")
                for metric, pct in improvements.get('vs_fixed_time', {}).items():
                    symbol = "↑" if pct > 0 else "↓"
                    lines.append(f"  {metric}: {abs(pct):.1f}% {symbol}")
                
                lines.append(f"\n{rl_name} vs Actuated:")
                for metric, pct in improvements.get('vs_actuated', {}).items():
                    symbol = "↑" if pct > 0 else "↓"
                    lines.append(f"  {metric}: {abs(pct):.1f}% {symbol}")
        
        return "\n".join(lines)
    
    def save_results(self, filename: str = "evaluation_results") -> None:
        """Save results to JSON and text files."""
        # JSON
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            # Convert numpy values to Python types
            serializable = {}
            for name, results in self.results.items():
                serializable[name] = {
                    k: float(v) if isinstance(v, np.floating) else v
                    for k, v in results.items()
                }
            json.dump(serializable, f, indent=2)
        
        # Text report
        report_path = self.output_dir / f"{filename}.txt"
        with open(report_path, 'w') as f:
            f.write(self.generate_report())
        
        logger.info(f"Results saved to: {self.output_dir}")
    
    def plot_comparison(self, save_path: Optional[str] = None) -> None:
        """Generate comparison plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not available for plotting")
            return
        
        if len(self.results) < 2:
            logger.warning("Need at least 2 agents for comparison plot")
            return
        
        # Prepare data
        agents = list(self.results.keys())
        metrics = ['avg_max_queue', 'avg_total_waiting', 'avg_episode_reward']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics):
            values = [self.results[a].get(metric, 0) for a in agents]
            
            bars = axes[idx].bar(agents, values, color=['#2ecc71', '#3498db', '#e74c3c'][:len(agents)])
            axes[idx].set_title(metric.replace('_', ' ').title())
            axes[idx].set_ylabel('Value')
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=9
                )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        else:
            plt.savefig(self.output_dir / "comparison.png", dpi=150, bbox_inches='tight')
        
        plt.close()


def load_agent(model_path: str, algorithm: str, env):
    """Load a trained agent from checkpoint."""
    
    if algorithm == 'qlearning':
        from src.agents.qlearning import QLearningAgent
        import pickle
        
        agent = QLearningAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n
        )
        
        with open(model_path, 'rb') as f:
            agent.q_table = pickle.load(f)
        
        return agent
    
    elif algorithm == 'dqn':
        from src.agents.dqn_agent import DQNAgent
        
        agent = DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n
        )
        agent.load(model_path)
        
        return agent
    
    elif algorithm == 'ppo':
        from src.agents.ppo_agent import PPOMultiAgent
        
        agent = PPOMultiAgent(env=env)
        agent.load(model_path)
        
        return agent
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def main():
    """Main evaluation entry point."""
    args = parse_args()
    
    print("="*60)
    print("BANGALORE TRAFFIC RL EVALUATION")
    print("="*60)
    
    # Create environment
    logger.info("Creating environment...")
    try:
        from src.environment.multi_junction_env import MultiJunctionTrafficEnv, EnvConfig, FlattenedTrafficEnv
        
        config = EnvConfig(
            junction_ids=['silk_board', 'tin_factory', 'hebbal', 'marathahalli'],
            queue_mode=args.mode if args.mode != 'all' else 'realistic_bangalore',
            max_steps=3600,
        )
        env = FlattenedTrafficEnv(MultiJunctionTrafficEnv(config=config))
        
    except Exception as e:
        logger.warning(f"Could not create full environment: {e}")
        from src.environment.mock_env import MockTrafficEnv
        env = MockTrafficEnv()
    
    # Create evaluator
    evaluator = ModelEvaluator(
        env=env,
        output_dir=args.output,
        seed=args.seed
    )
    
    # Load and evaluate RL agent
    if args.model:
        logger.info(f"Loading model: {args.model}")
        agent = load_agent(args.model, args.algorithm, env)
        agent_name = f"{args.algorithm.upper()} Agent"
    else:
        # Use random policy for demo
        logger.info("No model specified, using random policy for demo")
        
        class RandomAgent:
            def __init__(self, action_space):
                self.action_space = action_space
            
            def get_action(self, obs, training=False):
                return self.action_space.sample()
        
        agent = RandomAgent(env.action_space)
        agent_name = "Random Agent"
    
    # Evaluate
    if args.compare_all:
        # Full comparison
        comparison = evaluator.compare_with_baselines(
            agent,
            rl_name=agent_name,
            num_episodes=args.episodes
        )
    else:
        # Just evaluate the agent
        evaluator.evaluate_agent(agent, agent_name, args.episodes)
    
    # Generate outputs
    evaluator.save_results()
    evaluator.plot_comparison()
    
    # Print report
    print("\n" + evaluator.generate_report())
    
    # Cleanup
    env.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
