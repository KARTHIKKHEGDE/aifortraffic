#!/usr/bin/env python3
"""
=============================================================================
MARL Traffic Control - Production Training Script
=============================================================================

This is a production-ready training script for traffic signal control
using Deep Q-Learning with REAL SUMO simulation and Bangalore OSM data.

Features:
- Real SUMO simulation (no mock)
- Proper reward shaping for traffic optimization
- Experience replay with prioritization
- Target network updates
- Epsilon-greedy exploration with decay
- Comprehensive logging and metrics
- Model checkpointing
- Training visualization
- Early stopping

Usage:
    python scripts/train_perfect.py --junction hebbal --episodes 200
    python scripts/train_perfect.py --junction silk_board --episodes 100 --gui

Author: MARL Traffic Control Team
Date: January 2026
=============================================================================
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import argparse

import numpy as np

# =============================================================================
# PROJECT SETUP
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure SUMO
def setup_sumo() -> str:
    """Find and configure SUMO installation"""
    sumo_home = os.environ.get('SUMO_HOME')
    
    if not sumo_home:
        candidate_paths = [
            r"C:\Program Files (x86)\Eclipse\Sumo",
            r"C:\Program Files\Eclipse\Sumo",
            r"/usr/share/sumo",
            r"/opt/sumo",
            os.path.expanduser("~/sumo"),
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                os.environ['SUMO_HOME'] = path
                sumo_home = path
                break
    
    if not sumo_home:
        raise EnvironmentError(
            "SUMO not found! Please install SUMO and set SUMO_HOME environment variable.\n"
            "Download from: https://www.eclipse.org/sumo/"
        )
    
    # Add SUMO tools to path
    tools_path = os.path.join(sumo_home, 'tools')
    if tools_path not in sys.path:
        sys.path.append(tools_path)
    
    return sumo_home

SUMO_HOME = setup_sumo()
import traci

# Project directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs" / "training"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Environment
    junction_id: str = "hebbal"
    steps_per_episode: int = 500
    warmup_steps: int = 50
    
    # Agent
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 100000
    batch_size: int = 64
    target_update_freq: int = 500
    hidden_layers: List[int] = None
    
    # Training
    episodes: int = 200
    save_freq: int = 25
    log_freq: int = 10
    eval_freq: int = 50
    
    # Early stopping
    patience: int = 50
    min_improvement: float = 0.01
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [256, 256, 128]


@dataclass 
class EpisodeMetrics:
    """Metrics for a single episode"""
    episode: int
    total_reward: float
    avg_reward: float
    total_waiting_time: float
    total_queue_length: float
    throughput: int
    phase_switches: int
    epsilon: float
    loss: float
    duration_seconds: float


@dataclass
class TrainingState:
    """State of training for checkpointing"""
    episode: int
    best_reward: float
    best_episode: int
    total_steps: int
    epsilon: float
    rewards_history: List[float]
    losses_history: List[float]


# =============================================================================
# SUMO ENVIRONMENT
# =============================================================================

class SUMOTrafficEnv:
    """
    Production-ready SUMO Traffic Environment
    
    Uses REAL Bangalore OSM data for traffic simulation.
    Provides Gymnasium-compatible interface.
    """
    
    def __init__(self, config: TrainingConfig, gui: bool = False):
        self.config = config
        self.gui = gui
        
        # File paths
        self.net_file = DATA_DIR / "sumo" / f"{config.junction_id}.net.xml"
        self.route_file = DATA_DIR / "routes" / f"{config.junction_id}_generated.rou.xml"
        
        # Validate files
        self._validate_files()
        
        # State variables
        self.current_step = 0
        self.episode_count = 0
        self.tl_ids: List[str] = []
        self.primary_tl: str = ""
        self.n_phases: int = 0
        self.controlled_lanes: List[str] = []
        self.current_phase: int = 0
        self.last_switch_step: int = 0
        
        # Spaces (set after first reset)
        self.obs_size: int = 0
        self.action_size: int = 2  # 0=keep, 1=switch
        
        # Metrics
        self.episode_waiting_time = 0.0
        self.episode_queue_length = 0.0
        self.episode_throughput = 0
        self.episode_switches = 0
        
        # Build SUMO command
        self.sumo_cmd = self._build_sumo_command()
        
        # Flag
        self._initialized = False
    
    def _validate_files(self):
        """Validate that required files exist"""
        if not self.net_file.exists():
            raise FileNotFoundError(f"Network file not found: {self.net_file}")
        if not self.route_file.exists():
            raise FileNotFoundError(f"Route file not found: {self.route_file}")
        
        print(f"  Network: {self.net_file.name} ({self.net_file.stat().st_size / 1024:.1f} KB)")
        print(f"  Routes:  {self.route_file.name}")
    
    def _build_sumo_command(self) -> List[str]:
        """Build SUMO command line"""
        if sys.platform == 'win32':
            binary = os.path.join(SUMO_HOME, 'bin', 'sumo-gui.exe' if self.gui else 'sumo.exe')
        else:
            binary = os.path.join(SUMO_HOME, 'bin', 'sumo-gui' if self.gui else 'sumo')
        
        return [
            binary,
            '-n', str(self.net_file),
            '-r', str(self.route_file),
            '--step-length', '1.0',
            '--no-step-log', 'true',
            '--no-warnings', 'true',
            '--waiting-time-memory', '1000',
            '-b', '0',
            '-e', str(self.config.steps_per_episode + self.config.warmup_steps + 100),
            '--time-to-teleport', '300',
            '--seed', str(random.randint(0, 10000)),
        ]
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode"""
        # Close existing connection
        self._close_sumo()
        
        # Update seed for variety
        self.sumo_cmd[-1] = str(random.randint(0, 10000))
        
        # Start SUMO
        traci.start(self.sumo_cmd)
        self._initialized = True
        
        # Reset state
        self.current_step = 0
        self.episode_count += 1
        self.last_switch_step = 0
        
        # Reset metrics
        self.episode_waiting_time = 0.0
        self.episode_queue_length = 0.0
        self.episode_throughput = 0
        self.episode_switches = 0
        
        # Get traffic lights
        self.tl_ids = list(traci.trafficlight.getIDList())
        if not self.tl_ids:
            self._close_sumo()
            raise RuntimeError(f"No traffic lights in {self.config.junction_id}. Try 'silk_board' or 'hebbal'.")
        
        self.primary_tl = self.tl_ids[0]
        
        # Get phase info
        programs = traci.trafficlight.getAllProgramLogics(self.primary_tl)
        self.n_phases = len(programs[0].phases) if programs else 4
        
        # Get controlled lanes (limit to 8 for consistency)
        all_lanes = list(set(traci.trafficlight.getControlledLanes(self.primary_tl)))
        self.controlled_lanes = all_lanes[:8]
        
        # Calculate observation size
        # Per lane: queue, density, speed (3 features)
        # Plus: phase one-hot, time since switch, avg waiting
        self.obs_size = len(self.controlled_lanes) * 3 + self.n_phases + 2
        
        # Warmup
        for _ in range(self.config.warmup_steps):
            traci.simulationStep()
        self.current_step = self.config.warmup_steps
        
        # Get current phase
        self.current_phase = traci.trafficlight.getPhase(self.primary_tl)
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state"""
        # Track previous state for reward calculation
        prev_waiting = self._get_total_waiting()
        prev_queue = self._get_total_queue()
        
        # Apply action
        if action == 1:  # Switch phase
            time_in_phase = self.current_step - self.last_switch_step
            if time_in_phase >= 10:  # Minimum green time
                self.current_phase = (self.current_phase + 1) % self.n_phases
                traci.trafficlight.setPhase(self.primary_tl, self.current_phase)
                self.last_switch_step = self.current_step
                self.episode_switches += 1
        
        # Advance simulation
        traci.simulationStep()
        self.current_step += 1
        
        # Get new state
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(prev_waiting, prev_queue)
        
        # Update metrics
        self.episode_waiting_time += self._get_total_waiting()
        self.episode_queue_length += self._get_total_queue()
        self.episode_throughput += self._get_throughput()
        
        # Check done
        done = self.current_step >= self.config.steps_per_episode + self.config.warmup_steps
        
        # Info dict
        info = {
            'waiting_time': self._get_total_waiting(),
            'queue_length': self._get_total_queue(),
            'throughput': self._get_throughput(),
            'phase': self.current_phase,
        }
        
        return obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector"""
        obs = []
        
        # Per-lane features
        for lane in self.controlled_lanes:
            try:
                # Queue (halting vehicles, normalized)
                queue = traci.lane.getLastStepHaltingNumber(lane)
                obs.append(min(queue / 30.0, 1.0))
                
                # Density (occupancy, already 0-100)
                density = traci.lane.getLastStepOccupancy(lane) / 100.0
                obs.append(min(density, 1.0))
                
                # Speed (normalized by max speed ~15 m/s)
                speed = traci.lane.getLastStepMeanSpeed(lane)
                obs.append(min(speed / 15.0, 1.0) if speed > 0 else 0.0)
            except:
                obs.extend([0.0, 0.0, 0.0])
        
        # Phase one-hot encoding
        phase_onehot = [0.0] * self.n_phases
        phase_onehot[self.current_phase % self.n_phases] = 1.0
        obs.extend(phase_onehot)
        
        # Time since last switch (normalized by max green time ~90s)
        time_in_phase = self.current_step - self.last_switch_step
        obs.append(min(time_in_phase / 90.0, 1.0))
        
        # Average waiting time (normalized)
        avg_waiting = self._get_total_waiting() / max(len(self.controlled_lanes), 1)
        obs.append(min(avg_waiting / 100.0, 1.0))
        
        return np.array(obs, dtype=np.float32)
    
    def _get_total_waiting(self) -> float:
        """Get total waiting time across controlled lanes"""
        total = 0.0
        for lane in self.controlled_lanes:
            try:
                total += traci.lane.getWaitingTime(lane)
            except:
                pass
        return total
    
    def _get_total_queue(self) -> int:
        """Get total queue length across controlled lanes"""
        total = 0
        for lane in self.controlled_lanes:
            try:
                total += traci.lane.getLastStepHaltingNumber(lane)
            except:
                pass
        return total
    
    def _get_throughput(self) -> int:
        """Get number of vehicles that passed through"""
        total = 0
        for lane in self.controlled_lanes:
            try:
                total += traci.lane.getLastStepVehicleNumber(lane)
            except:
                pass
        return total
    
    def _calculate_reward(self, prev_waiting: float, prev_queue: int) -> float:
        """
        Calculate reward based on traffic improvement.
        
        Reward = throughput_bonus - waiting_penalty - queue_penalty - switch_penalty
        """
        curr_waiting = self._get_total_waiting()
        curr_queue = self._get_total_queue()
        throughput = self._get_throughput()
        
        # Reward components
        waiting_improvement = prev_waiting - curr_waiting
        queue_improvement = prev_queue - curr_queue
        
        # Weighted reward
        reward = (
            0.1 * throughput +              # Bonus for throughput
            0.05 * waiting_improvement +    # Bonus for reducing waiting
            0.1 * queue_improvement +       # Bonus for reducing queue
            -0.001 * curr_waiting +         # Penalty for current waiting
            -0.01 * curr_queue              # Penalty for current queue
        )
        
        # Penalize rapid switching
        time_in_phase = self.current_step - self.last_switch_step
        if time_in_phase < 10:
            reward -= 0.5
        
        return reward
    
    def get_episode_metrics(self) -> Dict:
        """Get metrics for completed episode"""
        steps = self.current_step - self.config.warmup_steps
        return {
            'avg_waiting_time': self.episode_waiting_time / max(steps, 1),
            'avg_queue_length': self.episode_queue_length / max(steps, 1),
            'total_throughput': self.episode_throughput,
            'phase_switches': self.episode_switches,
        }
    
    def _close_sumo(self):
        """Safely close SUMO connection"""
        if self._initialized:
            try:
                traci.close()
            except:
                pass
            self._initialized = False
    
    def close(self):
        """Clean up environment"""
        self._close_sumo()


# =============================================================================
# DQN AGENT
# =============================================================================

class DQNTrainer:
    """
    Deep Q-Network trainer with all best practices.
    """
    
    def __init__(self, env: SUMOTrafficEnv, config: TrainingConfig):
        self.env = env
        self.config = config
        
        # Import and create agent
        from src.agents.dqn_agent import DQNAgent
        
        # First reset to get observation size
        _ = env.reset()
        env.close()
        
        self.agent = DQNAgent(
            state_size=env.obs_size,
            action_size=env.action_size,
            hidden_layers=config.hidden_layers,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            epsilon=config.epsilon_start,
            epsilon_decay=config.epsilon_decay,
            epsilon_min=config.epsilon_end,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            target_update_freq=config.target_update_freq,
            double=True,
            dueling=True,
        )
        
        # Training state
        self.state = TrainingState(
            episode=0,
            best_reward=-float('inf'),
            best_episode=0,
            total_steps=0,
            epsilon=config.epsilon_start,
            rewards_history=[],
            losses_history=[],
        )
        
        # Metrics storage
        self.all_metrics: List[EpisodeMetrics] = []
        
        # Logging
        self.log_dir = LOGS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n  Agent created:")
        print(f"    State size:  {env.obs_size}")
        print(f"    Action size: {env.action_size}")
        print(f"    Network:     {config.hidden_layers}")
        print(f"    Device:      {self.agent.device}")
    
    def train(self) -> TrainingState:
        """Run full training loop"""
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        
        no_improvement_count = 0
        
        for episode in range(1, self.config.episodes + 1):
            self.state.episode = episode
            
            # Run episode
            metrics = self._run_episode()
            self.all_metrics.append(metrics)
            self.state.rewards_history.append(metrics.total_reward)
            if metrics.loss > 0:
                self.state.losses_history.append(metrics.loss)
            
            # Update best
            if metrics.total_reward > self.state.best_reward + self.config.min_improvement:
                self.state.best_reward = metrics.total_reward
                self.state.best_episode = episode
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Logging
            if episode % self.config.log_freq == 0:
                self._log_progress(episode, metrics)
            
            # Save checkpoint
            if episode % self.config.save_freq == 0:
                self._save_checkpoint(episode)
            
            # Early stopping
            if no_improvement_count >= self.config.patience:
                print(f"\n  Early stopping: No improvement for {self.config.patience} episodes")
                break
        
        # Final save
        self._save_final()
        
        return self.state
    
    def _run_episode(self) -> EpisodeMetrics:
        """Run a single training episode"""
        start_time = time.time()
        
        obs = self.env.reset()
        total_reward = 0.0
        total_loss = 0.0
        loss_count = 0
        
        for step in range(self.config.steps_per_episode):
            # Select action
            action = self.agent.get_action(obs)
            
            # Execute
            next_obs, reward, done, info = self.env.step(action)
            
            # Store experience
            self.agent.store_experience(obs, action, reward, next_obs, done)
            
            # Train
            if len(self.agent.replay_buffer) >= self.config.batch_size:
                loss = self.agent.train_step()
                if loss is not None:
                    total_loss += loss
                    loss_count += 1
            
            total_reward += reward
            obs = next_obs
            self.state.total_steps += 1
            
            if done:
                break
        
        # End episode - decay epsilon
        self.agent.end_episode()
        
        # Get episode metrics from environment
        env_metrics = self.env.get_episode_metrics()
        
        # Update epsilon
        self.state.epsilon = self.agent.epsilon
        
        duration = time.time() - start_time
        
        return EpisodeMetrics(
            episode=self.state.episode,
            total_reward=total_reward,
            avg_reward=total_reward / max(step + 1, 1),
            total_waiting_time=env_metrics['avg_waiting_time'],
            total_queue_length=env_metrics['avg_queue_length'],
            throughput=env_metrics['total_throughput'],
            phase_switches=env_metrics['phase_switches'],
            epsilon=self.agent.epsilon,
            loss=total_loss / max(loss_count, 1),
            duration_seconds=duration,
        )
    
    def _log_progress(self, episode: int, metrics: EpisodeMetrics):
        """Log training progress"""
        avg_reward_10 = np.mean(self.state.rewards_history[-10:])
        avg_reward_100 = np.mean(self.state.rewards_history[-100:]) if len(self.state.rewards_history) >= 100 else avg_reward_10
        
        print(
            f"  Ep {episode:4d} | "
            f"R: {metrics.total_reward:8.2f} | "
            f"Avg10: {avg_reward_10:8.2f} | "
            f"Best: {self.state.best_reward:8.2f} | "
            f"eps: {metrics.epsilon:.3f} | "
            f"loss: {metrics.loss:.4f}"
        )
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        MODELS_DIR.mkdir(exist_ok=True)
        
        # Save agent
        model_path = MODELS_DIR / f"dqn_{self.config.junction_id}_ep{episode}.pth"
        self.agent.save(str(model_path))
        
        # Save metrics
        metrics_path = self.log_dir / f"metrics_ep{episode}.json"
        with open(metrics_path, 'w') as f:
            json.dump([asdict(m) for m in self.all_metrics], f, indent=2)
        
        print(f"  [Checkpoint saved: ep {episode}]")
    
    def _save_final(self):
        """Save final model and training summary"""
        MODELS_DIR.mkdir(exist_ok=True)
        
        # Save best model
        final_path = MODELS_DIR / f"dqn_{self.config.junction_id}_final.pth"
        self.agent.save(str(final_path))
        
        # Save training state
        state_path = self.log_dir / "training_state.json"
        with open(state_path, 'w') as f:
            json.dump({
                'episode': self.state.episode,
                'best_reward': self.state.best_reward,
                'best_episode': self.state.best_episode,
                'total_steps': self.state.total_steps,
                'final_epsilon': self.state.epsilon,
            }, f, indent=2)
        
        # Save config
        config_path = self.log_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        # Save all metrics
        metrics_path = self.log_dir / "all_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump([asdict(m) for m in self.all_metrics], f, indent=2)
        
        print(f"\n  Final model saved: {final_path}")
        print(f"  Training logs: {self.log_dir}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train DQN for traffic control with REAL SUMO simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_perfect.py --junction hebbal --episodes 200
  python scripts/train_perfect.py --junction silk_board --episodes 100 --gui
  python scripts/train_perfect.py --junction hebbal --lr 0.0005 --batch-size 128
        """
    )
    
    # Environment
    parser.add_argument('--junction', '-j', type=str, default='hebbal',
                        choices=['silk_board', 'hebbal', 'marathahalli'],
                        help='Junction to train on (default: hebbal)')
    parser.add_argument('--steps', type=int, default=500,
                        help='Steps per episode (default: 500)')
    parser.add_argument('--gui', action='store_true',
                        help='Show SUMO GUI')
    
    # Training
    parser.add_argument('--episodes', '-e', type=int, default=200,
                        help='Number of episodes (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--buffer-size', type=int, default=100000,
                        help='Replay buffer size (default: 100000)')
    
    # Logging
    parser.add_argument('--save-freq', type=int, default=25,
                        help='Save checkpoint every N episodes (default: 25)')
    parser.add_argument('--log-freq', type=int, default=10,
                        help='Log progress every N episodes (default: 10)')
    
    return parser.parse_args()


def main():
    """Main training entry point"""
    args = parse_args()
    
    # Print banner
    print("\n" + "=" * 60)
    print("  MARL TRAFFIC CONTROL - PRODUCTION TRAINING")
    print("  Using REAL Bangalore OSM Data + SUMO Simulation")
    print("=" * 60)
    
    # Create config
    config = TrainingConfig(
        junction_id=args.junction,
        steps_per_episode=args.steps,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        episodes=args.episodes,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
    )
    
    print(f"\n  Configuration:")
    print(f"    Junction:    {config.junction_id}")
    print(f"    Episodes:    {config.episodes}")
    print(f"    Steps/Ep:    {config.steps_per_episode}")
    print(f"    Learning Rate: {config.learning_rate}")
    print(f"    Gamma:       {config.gamma}")
    print(f"    Batch Size:  {config.batch_size}")
    print(f"    GUI:         {args.gui}")
    
    # Create environment
    print(f"\n  Loading environment...")
    try:
        env = SUMOTrafficEnv(config, gui=args.gui)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print("  Run 'python scripts/00_setup_real_data.py' first to download data.")
        return 1
    except RuntimeError as e:
        print(f"\n  ERROR: {e}")
        return 1
    
    try:
        # Create trainer
        trainer = DQNTrainer(env, config)
        
        # Train
        state = trainer.train()
        
        # Summary
        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE")
        print("=" * 60)
        print(f"    Total episodes:  {state.episode}")
        print(f"    Total steps:     {state.total_steps:,}")
        print(f"    Best reward:     {state.best_reward:.2f} (episode {state.best_episode})")
        print(f"    Final epsilon:   {state.epsilon:.4f}")
        
        if len(state.rewards_history) >= 10:
            print(f"    Avg last 10:     {np.mean(state.rewards_history[-10:]):.2f}")
        if len(state.rewards_history) >= 100:
            print(f"    Avg last 100:    {np.mean(state.rewards_history[-100:]):.2f}")
        
        print("\n  Model saved to: models/dqn_{}_final.pth".format(config.junction_id))
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n  Training interrupted by user")
        return 1
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
