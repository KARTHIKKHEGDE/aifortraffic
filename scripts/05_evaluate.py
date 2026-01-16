"""
Evaluation Script
Compare trained agents against baselines
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import ConfigManager
from src.utils.logger import setup_logger
from src.environment import BangaloreTrafficEnv
from src.agents import QLearningAgent, DQNAgent, PPOMultiAgent
from src.evaluation import (
    TrafficMetrics,
    compute_metrics,
    FixedTimeController,
    ActuatedController,
    run_baseline,
    ResultsAnalyzer
)

logger = setup_logger("evaluate")


def load_agent(
    agent_type: str,
    model_path: str,
    state_size: int,
    action_size: int,
    env=None,
    device: str = 'auto'
):
    """Load trained agent from checkpoint"""
    if agent_type == 'qlearning':
        agent = QLearningAgent(state_size=state_size, action_size=action_size)
        agent.load(model_path)
    
    elif agent_type == 'dqn':
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            device=device
        )
        agent.load(model_path)
    
    elif agent_type == 'ppo':
        if env is None:
            raise ValueError("Environment required for PPO agent")
        agent = PPOMultiAgent.from_pretrained(model_path, env)
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent


def evaluate_baselines(
    env,
    num_episodes: int = 10,
    max_steps: int = 3600
) -> Dict[str, Dict]:
    """Evaluate baseline controllers"""
    results = {}
    
    # Fixed-time controller
    logger.info("Evaluating Fixed-Time Controller...")
    fixed_controller = FixedTimeController()
    fixed_results = run_baseline(env, fixed_controller, num_episodes, max_steps)
    results['fixed_time'] = fixed_results
    
    # Actuated controller
    logger.info("Evaluating Actuated Controller...")
    actuated_controller = ActuatedController()
    actuated_results = run_baseline(env, actuated_controller, num_episodes, max_steps)
    results['actuated'] = actuated_results
    
    return results


def evaluate_agent(
    env,
    agent,
    num_episodes: int = 10,
    max_steps: int = 3600
) -> TrafficMetrics:
    """Evaluate trained agent"""
    logger.info(f"Evaluating agent for {num_episodes} episodes...")
    
    metrics = compute_metrics(
        env=env,
        agent=agent,
        num_episodes=num_episodes,
        max_steps=max_steps,
        deterministic=True
    )
    
    return metrics


def print_comparison_table(
    agent_metrics: TrafficMetrics,
    baseline_results: Dict[str, Dict]
):
    """Print formatted comparison table"""
    agent_summary = agent_metrics.get_summary()
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS COMPARISON")
    print("=" * 80)
    
    # Table header
    print(f"\n{'Metric':<30} {'RL Agent':>12} {'Fixed-Time':>12} {'Actuated':>12}")
    print("-" * 66)
    
    # Get baseline summaries
    fixed_summary = baseline_results.get('fixed_time', {}).get('summary', {})
    actuated_summary = baseline_results.get('actuated', {}).get('summary', {})
    
    # Metrics to display
    metrics_display = [
        ('avg_waiting_time', 'Avg Waiting Time (s)'),
        ('avg_queue_length', 'Avg Queue Length'),
        ('avg_speed', 'Avg Speed (m/s)'),
        ('avg_throughput_per_hour', 'Throughput (veh/hr)'),
        ('avg_emergency_clearance', 'Emergency Clearance (s)'),
        ('avg_episode_reward', 'Episode Reward'),
    ]
    
    for metric_key, metric_name in metrics_display:
        agent_val = agent_summary.get(metric_key, 0)
        fixed_val = fixed_summary.get(metric_key, 0)
        actuated_val = actuated_summary.get(metric_key, 0)
        
        print(f"{metric_name:<30} {agent_val:>12.1f} {fixed_val:>12.1f} {actuated_val:>12.1f}")
    
    print("-" * 66)
    
    # Improvement summary
    print("\nIMPROVEMENT vs FIXED-TIME:")
    
    if fixed_summary:
        comparison = agent_metrics.compare_to_baseline(
            baseline_results['fixed_time']['metrics']
        )
        
        for key, value in comparison.items():
            clean_name = key.replace('_improvement_%', '').replace('_', ' ').title()
            status = "↑" if value > 0 else "↓"
            print(f"  {clean_name}: {status} {abs(value):.1f}%")
    
    print("=" * 80)


def save_results(
    agent_metrics: TrafficMetrics,
    baseline_results: Dict[str, Dict],
    output_path: str
):
    """Save evaluation results to file"""
    results = {
        'agent': agent_metrics.to_dict(),
        'baselines': {
            name: data['summary'] for name, data in baseline_results.items()
        },
        'comparison': {}
    }
    
    for baseline_name, baseline_data in baseline_results.items():
        if 'metrics' in baseline_data:
            results['comparison'][baseline_name] = agent_metrics.compare_to_baseline(
                baseline_data['metrics']
            )
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained traffic control agent"
    )
    
    parser.add_argument(
        '--agent-type', '-a',
        type=str,
        required=True,
        choices=['qlearning', 'dqn', 'ppo'],
        help='Type of agent to evaluate'
    )
    
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        required=True,
        help='Path to trained model'
    )
    
    parser.add_argument(
        '--env-config', '-e',
        type=str,
        default='configs/env_config.yaml',
        help='Environment config path'
    )
    
    parser.add_argument(
        '--episodes', '-n',
        type=int,
        default=10,
        help='Number of evaluation episodes'
    )
    
    parser.add_argument(
        '--max-steps', '-s',
        type=int,
        default=3600,
        help='Maximum steps per episode'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='evaluation_results.json',
        help='Output file path'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='auto',
        help='Device (auto, cuda, cpu)'
    )
    
    parser.add_argument(
        '--report', '-r',
        action='store_true',
        help='Generate detailed report'
    )
    
    parser.add_argument(
        '--junction',
        type=str,
        default=None,
        help='Specific junction to evaluate'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_manager = ConfigManager()
    env_config = config_manager.load_yaml(args.env_config)
    
    # Override junction if specified
    if args.junction:
        env_config['junction_id'] = args.junction
    
    # Create environment
    logger.info("Creating environment...")
    env = BangaloreTrafficEnv(env_config)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Load agent
    logger.info(f"Loading {args.agent_type} agent from {args.model_path}")
    agent = load_agent(
        args.agent_type,
        args.model_path,
        state_size,
        action_size,
        env=env,
        device=args.device
    )
    
    try:
        # Evaluate baselines
        logger.info("Evaluating baseline controllers...")
        baseline_results = evaluate_baselines(env, args.episodes, args.max_steps)
        
        # Evaluate agent
        logger.info("Evaluating trained agent...")
        agent_metrics = evaluate_agent(env, agent, args.episodes, args.max_steps)
        
        # Print comparison
        print_comparison_table(agent_metrics, baseline_results)
        
        # Save results
        save_results(agent_metrics, baseline_results, args.output)
        
        # Generate report if requested
        if args.report:
            analyzer = ResultsAnalyzer()
            
            # Add baselines
            for name, data in baseline_results.items():
                analyzer.add_baseline_metrics(name, data['metrics'])
            
            # Generate report
            report_path = args.output.replace('.json', '_report.txt')
            report = analyzer.generate_report(
                {args.agent_type: agent_metrics},
                {name: data['metrics'] for name, data in baseline_results.items()},
                output_path=report_path
            )
            print(f"\nDetailed report saved to: {report_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        
        agent_summary = agent_metrics.get_summary()
        fixed_summary = baseline_results['fixed_time']['summary']
        
        wait_improvement = ((fixed_summary['avg_waiting_time'] - agent_summary['avg_waiting_time']) 
                           / fixed_summary['avg_waiting_time'] * 100)
        
        print(f"Waiting Time Improvement: {wait_improvement:+.1f}%")
        print(f"Target (33%): {'✓ MET' if wait_improvement >= 33 else '✗ NOT MET'}")
        print("=" * 80)
        
    finally:
        env.close()


if __name__ == "__main__":
    main()
