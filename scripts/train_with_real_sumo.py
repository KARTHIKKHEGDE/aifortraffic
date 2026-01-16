#!/usr/bin/env python3
"""
Train Traffic Control Agent with REAL SUMO Simulation

This script trains a DQN agent using REAL Bangalore traffic data
from OpenStreetMap - NOT mock simulation.

Usage:
    python scripts/train_with_real_sumo.py --junction silk_board --episodes 100
    python scripts/train_with_real_sumo.py --junction hebbal --algorithm ppo --episodes 500
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup SUMO
def setup_sumo():
    """Find and configure SUMO"""
    sumo_home = os.environ.get('SUMO_HOME')
    if not sumo_home:
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
    
    if not sumo_home:
        print("ERROR: SUMO not found! Please install SUMO or set SUMO_HOME")
        sys.exit(1)
    
    sys.path.append(os.path.join(sumo_home, 'tools'))
    return sumo_home

SUMO_HOME = setup_sumo()
import traci

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train traffic agent with real SUMO")
    
    parser.add_argument(
        '--junction', '-j',
        type=str,
        choices=['silk_board', 'tin_factory', 'hebbal', 'marathahalli'],
        default='silk_board',
        help='Junction to train on'
    )
    
    parser.add_argument(
        '--algorithm', '-a',
        type=str,
        choices=['dqn', 'ppo'],
        default='dqn',
        help='Algorithm (dqn or ppo)'
    )
    
    parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=100,
        help='Number of episodes'
    )
    
    parser.add_argument(
        '--steps-per-episode',
        type=int,
        default=500,
        help='Steps per episode'
    )
    
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Show SUMO GUI'
    )
    
    parser.add_argument(
        '--save-freq',
        type=int,
        default=10,
        help='Save model every N episodes'
    )
    
    return parser.parse_args()


class RealSUMOTrafficEnv:
    """
    Gymnasium-compatible environment using REAL SUMO simulation
    with actual Bangalore OSM data.
    """
    
    def __init__(self, junction_id: str, steps_per_episode: int = 500, gui: bool = False):
        self.junction_id = junction_id
        self.steps_per_episode = steps_per_episode
        self.gui = gui
        
        # File paths
        self.net_file = DATA_DIR / "sumo" / f"{junction_id}.net.xml"
        self.route_file = DATA_DIR / "routes" / f"{junction_id}_generated.rou.xml"
        
        # Verify files exist
        if not self.net_file.exists():
            raise FileNotFoundError(f"Network file not found: {self.net_file}")
        if not self.route_file.exists():
            raise FileNotFoundError(f"Route file not found: {self.route_file}")
        
        print(f"[OK] Network: {self.net_file}")
        print(f"[OK] Routes: {self.route_file}")
        
        # State
        self.current_step = 0
        self.tl_ids = []
        self.primary_tl = None
        self.n_phases = 0
        self.controlled_lanes = []
        
        # Action/Observation spaces
        self.action_size = 2  # 0=keep, 1=switch
        self.obs_size = None  # Set after first reset
        
        # Build SUMO command
        sumo_binary = os.path.join(SUMO_HOME, 'bin', 'sumo-gui.exe' if gui else 'sumo.exe')
        if sys.platform != 'win32':
            sumo_binary = os.path.join(SUMO_HOME, 'bin', 'sumo-gui' if gui else 'sumo')
        
        self.sumo_cmd = [
            sumo_binary,
            '-n', str(self.net_file),
            '-r', str(self.route_file),
            '--step-length', '1.0',
            '--no-step-log', 'true',
            '--no-warnings', 'true',
            '-b', '0',
            '-e', str(steps_per_episode + 100),
            '--time-to-teleport', '300',
        ]
    
    @property
    def observation_space_shape(self):
        return (self.obs_size,) if self.obs_size else (20,)
    
    @property
    def action_space_n(self):
        return self.action_size
    
    def reset(self):
        """Reset environment for new episode"""
        # Close existing connection
        try:
            traci.close()
        except:
            pass
        
        # Start SUMO
        traci.start(self.sumo_cmd)
        self.current_step = 0
        
        # Get traffic lights
        self.tl_ids = list(traci.trafficlight.getIDList())
        if not self.tl_ids:
            traci.close()
            raise RuntimeError(f"No traffic lights found in {self.junction_id}")
        
        self.primary_tl = self.tl_ids[0]
        
        # Get phases and lanes
        programs = traci.trafficlight.getAllProgramLogics(self.primary_tl)
        self.n_phases = len(programs[0].phases) if programs else 4
        self.controlled_lanes = list(set(traci.trafficlight.getControlledLanes(self.primary_tl)))[:8]
        
        # Calculate observation size: queues + densities + speeds + phase_onehot + time
        self.obs_size = len(self.controlled_lanes) * 3 + self.n_phases + 1
        
        # Warmup
        for _ in range(50):
            traci.simulationStep()
        self.current_step = 50
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current observation from SUMO"""
        obs = []
        
        # Per-lane features
        for lane in self.controlled_lanes:
            try:
                # Queue (halting vehicles)
                queue = traci.lane.getLastStepHaltingNumber(lane)
                obs.append(min(queue / 50.0, 1.0))
                
                # Density (occupancy)
                density = traci.lane.getLastStepOccupancy(lane) / 100.0
                obs.append(min(density, 1.0))
                
                # Speed
                speed = traci.lane.getLastStepMeanSpeed(lane)
                obs.append(min(speed / 15.0, 1.0))
            except:
                obs.extend([0, 0, 0])
        
        # Pad if needed
        while len(obs) < len(self.controlled_lanes) * 3:
            obs.extend([0, 0, 0])
        
        # Current phase (one-hot)
        current_phase = traci.trafficlight.getPhase(self.primary_tl)
        phase_onehot = [0] * self.n_phases
        phase_onehot[current_phase % self.n_phases] = 1
        obs.extend(phase_onehot)
        
        # Time since last switch (normalized)
        time_in_phase = traci.trafficlight.getPhaseDuration(self.primary_tl)
        obs.append(min(time_in_phase / 90.0, 1.0))
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        """Execute action and return next state"""
        # Apply action
        if action == 1:  # Switch phase
            current_phase = traci.trafficlight.getPhase(self.primary_tl)
            next_phase = (current_phase + 1) % self.n_phases
            traci.trafficlight.setPhase(self.primary_tl, next_phase)
        
        # Simulation step
        traci.simulationStep()
        self.current_step += 1
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check done
        done = self.current_step >= self.steps_per_episode
        
        return obs, reward, done, {}
    
    def _calculate_reward(self):
        """Calculate reward based on traffic metrics"""
        total_waiting = 0
        total_queue = 0
        throughput = 0
        
        for lane in self.controlled_lanes:
            try:
                total_waiting += traci.lane.getWaitingTime(lane)
                total_queue += traci.lane.getLastStepHaltingNumber(lane)
                # Count vehicles that passed through
                throughput += traci.lane.getLastStepVehicleNumber(lane)
            except:
                pass
        
        # Negative reward for waiting time and queue, positive for throughput
        reward = -0.01 * total_waiting - 0.1 * total_queue + 0.05 * throughput
        
        # Round for display
        return round(reward, 4)
    
    def close(self):
        """Close SUMO connection"""
        try:
            traci.close()
        except:
            pass


def train_dqn(env: RealSUMOTrafficEnv, args):
    """Train DQN agent on real SUMO environment"""
    from src.agents.dqn_agent import DQNAgent
    
    # Initialize agent
    agent = DQNAgent(
        state_size=env.obs_size,
        action_size=env.action_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=50000,
        batch_size=64,
        target_update_freq=500
    )
    
    print(f"\nDQN Agent initialized:")
    print(f"  State size: {env.obs_size}")
    print(f"  Action size: {env.action_size}")
    
    # Training loop
    all_rewards = []
    
    for episode in range(args.episodes):
        obs = env.reset()
        total_reward = 0
        
        for step in range(args.steps_per_episode):
            # Select action
            action = agent.get_action(obs)
            
            # Execute
            next_obs, reward, done, info = env.step(action)
            
            # Store experience
            agent.store_experience(obs, action, reward, next_obs, done)
            
            # Train
            if len(agent.replay_buffer) >= 64:
                agent.train_step()
            
            total_reward += reward
            obs = next_obs
            
            if done:
                break
        
        all_rewards.append(total_reward)
        avg_reward = np.mean(all_rewards[-10:])
        
        print(f"Episode {episode+1}/{args.episodes} | "
              f"Reward: {total_reward:.2f} | "
              f"Avg(10): {avg_reward:.2f} | "
              f"Epsilon: {agent.epsilon:.3f}")
        
        # Save checkpoint
        if (episode + 1) % args.save_freq == 0:
            save_path = MODELS_DIR / f"dqn_{args.junction}_ep{episode+1}.pth"
            agent.save(str(save_path))
            print(f"  Saved: {save_path}")
    
    # Final save
    final_path = MODELS_DIR / f"dqn_{args.junction}_final.pth"
    agent.save(str(final_path))
    print(f"\nFinal model saved: {final_path}")
    
    return all_rewards


def main():
    args = parse_args()
    
    print("=" * 60)
    print("REAL SUMO TRAFFIC TRAINING")
    print("=" * 60)
    print(f"Junction: {args.junction}")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Episodes: {args.episodes}")
    print(f"Steps/Episode: {args.steps_per_episode}")
    print(f"SUMO GUI: {args.gui}")
    print("=" * 60)
    
    # Create models directory
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Create environment
    print("\nCreating REAL SUMO environment...")
    env = RealSUMOTrafficEnv(
        junction_id=args.junction,
        steps_per_episode=args.steps_per_episode,
        gui=args.gui
    )
    
    # First reset to get observation size
    print("\nInitializing simulation...")
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Traffic lights: {len(env.tl_ids)} (primary: {env.primary_tl})")
    print(f"Controlled lanes: {len(env.controlled_lanes)}")
    print(f"Phases: {env.n_phases}")
    env.close()
    
    try:
        # Train
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        
        if args.algorithm == 'dqn':
            rewards = train_dqn(env, args)
        else:
            print("PPO training not yet implemented for real SUMO")
            return 1
        
        # Summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total episodes: {len(rewards)}")
        print(f"Best reward: {max(rewards):.2f}")
        print(f"Final avg (10): {np.mean(rewards[-10:]):.2f}")
        
    finally:
        env.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
