"""
Demo Script
Interactive demonstration of trained traffic control agent
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import ConfigManager
from src.utils.logger import setup_logger
from src.environment import BangaloreTrafficEnv
from src.agents import DQNAgent, QLearningAgent, PPOMultiAgent
from src.evaluation import FixedTimeController, ActuatedController

logger = setup_logger("demo")


def load_agent(agent_type: str, model_path: str, env, device: str = 'auto'):
    """Load trained agent"""
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    if agent_type == 'qlearning':
        agent = QLearningAgent(state_size=state_size, action_size=action_size)
        agent.load(model_path)
    elif agent_type == 'dqn':
        agent = DQNAgent(state_size=state_size, action_size=action_size, device=device)
        agent.load(model_path)
    elif agent_type == 'ppo':
        agent = PPOMultiAgent.from_pretrained(model_path, env)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent


def run_demo(
    env,
    controller,
    controller_name: str,
    num_steps: int = 1000,
    render: bool = True,
    delay: float = 0.1,
    verbose: bool = True
):
    """
    Run demonstration with given controller
    
    Args:
        env: Traffic environment
        controller: Controller (agent or baseline)
        controller_name: Name for display
        num_steps: Number of simulation steps
        render: Whether to render (SUMO-GUI)
        delay: Delay between steps for visualization
        verbose: Print step information
    """
    print(f"\n{'='*60}")
    print(f"Running Demo: {controller_name}")
    print(f"{'='*60}")
    
    obs, info = env.reset()
    
    if hasattr(controller, 'reset'):
        controller.reset()
    
    total_reward = 0
    total_waiting_time = 0
    emergency_events = 0
    emergency_cleared = 0
    
    for step in range(num_steps):
        # Get action
        if hasattr(controller, 'predict'):
            # PPO agent
            action, _ = controller.predict(obs, deterministic=True)
            action = int(action[0]) if hasattr(action, '__len__') else int(action)
        elif hasattr(controller, 'get_action'):
            # Q-learning, DQN, or baseline
            action = controller.get_action(obs, training=False)
        else:
            action = 0
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        total_waiting_time += info.get('avg_waiting_time', 0)
        
        if info.get('emergency_active', False):
            emergency_events += 1
        if info.get('emergency_cleared', False):
            emergency_cleared += 1
        
        if verbose and step % 100 == 0:
            avg_queue = info.get('avg_queue_length', 0)
            avg_wait = info.get('avg_waiting_time', 0)
            phase = info.get('current_phase', 0)
            
            print(f"Step {step:4d} | Action: {action} | Phase: {phase} | "
                  f"Queue: {avg_queue:.1f} | Wait: {avg_wait:.1f}s | "
                  f"Reward: {reward:.2f}")
        
        if render:
            env.render()
            time.sleep(delay)
        
        obs = next_obs
        
        if done:
            break
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Demo Complete: {controller_name}")
    print(f"{'='*60}")
    print(f"Steps: {step + 1}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Average Waiting Time: {total_waiting_time / (step + 1):.1f}s")
    print(f"Emergency Events: {emergency_events}")
    print(f"Emergencies Cleared: {emergency_cleared}")
    print(f"{'='*60}\n")
    
    return {
        'steps': step + 1,
        'total_reward': total_reward,
        'avg_waiting_time': total_waiting_time / (step + 1),
        'emergency_events': emergency_events,
        'emergency_cleared': emergency_cleared
    }


def run_comparison_demo(
    env_config: dict,
    agent_type: Optional[str] = None,
    model_path: Optional[str] = None,
    num_steps: int = 500,
    render: bool = False,
    device: str = 'auto'
):
    """
    Run comparison demo between RL agent and baselines
    
    Args:
        env_config: Environment configuration
        agent_type: Type of RL agent
        model_path: Path to trained model
        num_steps: Steps per demo
        render: Whether to render
        device: PyTorch device
    """
    results = {}
    
    # Create environment
    env = BangaloreTrafficEnv(env_config)
    
    try:
        # Fixed-time baseline
        print("\n" + "="*60)
        print("DEMO 1: Fixed-Time Controller (Baseline)")
        print("="*60)
        fixed_controller = FixedTimeController()
        results['fixed_time'] = run_demo(
            env, fixed_controller, "Fixed-Time Controller",
            num_steps=num_steps, render=render, verbose=True
        )
        
        # Actuated baseline
        print("\n" + "="*60)
        print("DEMO 2: Actuated Controller (Baseline)")
        print("="*60)
        actuated_controller = ActuatedController()
        results['actuated'] = run_demo(
            env, actuated_controller, "Actuated Controller",
            num_steps=num_steps, render=render, verbose=True
        )
        
        # RL Agent (if provided)
        if agent_type and model_path:
            print("\n" + "="*60)
            print(f"DEMO 3: RL Agent ({agent_type.upper()})")
            print("="*60)
            agent = load_agent(agent_type, model_path, env, device)
            results['rl_agent'] = run_demo(
                env, agent, f"RL Agent ({agent_type})",
                num_steps=num_steps, render=render, verbose=True
            )
        
        # Print comparison
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Controller':<25} {'Reward':>10} {'Wait Time':>12} {'Emergency':>10}")
        print("-"*60)
        
        for name, res in results.items():
            print(f"{name:<25} {res['total_reward']:>10.1f} "
                  f"{res['avg_waiting_time']:>10.1f}s "
                  f"{res['emergency_cleared']:>10}")
        
        print("="*60)
        
        # Calculate improvements
        if 'rl_agent' in results and 'fixed_time' in results:
            fixed_wait = results['fixed_time']['avg_waiting_time']
            rl_wait = results['rl_agent']['avg_waiting_time']
            improvement = (fixed_wait - rl_wait) / fixed_wait * 100
            
            print(f"\nRL Agent Improvement over Fixed-Time:")
            print(f"  Waiting Time: {improvement:+.1f}%")
            print(f"  Target (33%): {'✓ MET' if improvement >= 33 else '✗ NOT MET'}")
        
    finally:
        env.close()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Demo for Bangalore Traffic Signal Control"
    )
    
    parser.add_argument(
        '--agent-type', '-a',
        type=str,
        choices=['qlearning', 'dqn', 'ppo'],
        help='Type of RL agent'
    )
    
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        help='Path to trained model'
    )
    
    parser.add_argument(
        '--env-config', '-e',
        type=str,
        default='configs/env_config.yaml',
        help='Environment config path'
    )
    
    parser.add_argument(
        '--steps', '-s',
        type=int,
        default=500,
        help='Number of simulation steps'
    )
    
    parser.add_argument(
        '--render', '-r',
        action='store_true',
        help='Enable SUMO-GUI rendering'
    )
    
    parser.add_argument(
        '--baseline-only',
        action='store_true',
        help='Only run baseline controllers'
    )
    
    parser.add_argument(
        '--junction',
        type=str,
        default='silk_board',
        choices=['silk_board', 'tin_factory', 'hebbal', 'marathahalli'],
        help='Junction to simulate'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='auto',
        help='Device (auto, cuda, cpu)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_manager = ConfigManager()
    env_config = config_manager.load_yaml(args.env_config)
    
    # Set junction
    env_config['junction_id'] = args.junction
    
    # Set render mode
    if args.render:
        env_config['simulation'] = env_config.get('simulation', {})
        env_config['simulation']['gui'] = True
    
    print("\n" + "="*60)
    print("BANGALORE ADAPTIVE TRAFFIC SIGNAL CONTROL - DEMO")
    print("="*60)
    print(f"Junction: {args.junction}")
    print(f"Steps: {args.steps}")
    print(f"Render: {args.render}")
    
    if args.agent_type and args.model_path:
        print(f"Agent: {args.agent_type}")
        print(f"Model: {args.model_path}")
    
    print("="*60)
    
    # Run demo
    if args.baseline_only:
        # Only baselines
        run_comparison_demo(
            env_config=env_config,
            num_steps=args.steps,
            render=args.render
        )
    else:
        # Full comparison
        run_comparison_demo(
            env_config=env_config,
            agent_type=args.agent_type,
            model_path=args.model_path,
            num_steps=args.steps,
            render=args.render,
            device=args.device
        )


if __name__ == "__main__":
    main()
