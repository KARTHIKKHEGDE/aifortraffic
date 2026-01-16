#!/usr/bin/env python3
"""
Evaluate trained model on traffic control task.
"""

import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup SUMO
def setup_sumo():
    sumo_home = os.environ.get('SUMO_HOME')
    if not sumo_home:
        for path in [r"C:\Program Files (x86)\Eclipse\Sumo", r"C:\Program Files\Eclipse\Sumo"]:
            if os.path.exists(path):
                os.environ['SUMO_HOME'] = path
                sumo_home = path
                break
    sys.path.append(os.path.join(sumo_home, 'tools'))
    return sumo_home

SUMO_HOME = setup_sumo()
import traci

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def evaluate_model(junction_id: str, model_path: str, episodes: int = 5):
    """Evaluate a trained model"""
    from src.agents.dqn_agent import DQNAgent
    
    print(f"\n{'='*60}")
    print(f"EVALUATING TRAINED MODEL")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Junction: {junction_id}")
    print(f"Episodes: {episodes}")
    
    # Load network files
    net_file = DATA_DIR / "sumo" / f"{junction_id}.net.xml"
    route_file = DATA_DIR / "routes" / f"{junction_id}_generated.rou.xml"
    
    sumo_binary = os.path.join(SUMO_HOME, 'bin', 'sumo.exe')
    sumo_cmd = [
        sumo_binary,
        '-n', str(net_file),
        '-r', str(route_file),
        '--step-length', '1.0',
        '--no-step-log', 'true',
        '-b', '0', '-e', '600',
    ]
    
    all_rewards = []
    all_waiting = []
    
    agent = None
    
    for ep in range(episodes):
        traci.start(sumo_cmd)
        
        # Get TL info
        tl_ids = list(traci.trafficlight.getIDList())
        primary_tl = tl_ids[0]
        programs = traci.trafficlight.getAllProgramLogics(primary_tl)
        n_phases = len(programs[0].phases)
        controlled_lanes = list(set(traci.trafficlight.getControlledLanes(primary_tl)))[:8]
        
        obs_size = len(controlled_lanes) * 3 + n_phases + 2
        
        # Create agent on first episode
        if agent is None:
            agent = DQNAgent(
                state_size=obs_size,
                action_size=2,
                epsilon=0.0,  # No exploration during eval
                dueling=True,  # Match training architecture
                double=True,
                hidden_layers=[256, 256, 128],
            )
            agent.load(model_path)
            print(f"\nAgent loaded (state_size={obs_size})")
        
        # Warmup
        for _ in range(50):
            traci.simulationStep()
        
        current_phase = traci.trafficlight.getPhase(primary_tl)
        last_switch = 50
        total_reward = 0
        total_waiting = 0
        
        for step in range(50, 550):
            # Build observation
            obs = []
            for lane in controlled_lanes:
                try:
                    obs.append(min(traci.lane.getLastStepHaltingNumber(lane) / 30.0, 1.0))
                    obs.append(min(traci.lane.getLastStepOccupancy(lane) / 100.0, 1.0))
                    obs.append(min(traci.lane.getLastStepMeanSpeed(lane) / 15.0, 1.0))
                except:
                    obs.extend([0, 0, 0])
            
            phase_onehot = [0.0] * n_phases
            phase_onehot[current_phase % n_phases] = 1.0
            obs.extend(phase_onehot)
            obs.append(min((step - last_switch) / 90.0, 1.0))
            
            lane_waiting = sum(traci.lane.getWaitingTime(l) for l in controlled_lanes[:4])
            obs.append(min(lane_waiting / 100.0, 1.0))
            
            obs = np.array(obs, dtype=np.float32)
            
            # Get action
            action = agent.get_action(obs, training=False)
            
            # Apply action
            if action == 1 and (step - last_switch) >= 10:
                current_phase = (current_phase + 1) % n_phases
                traci.trafficlight.setPhase(primary_tl, current_phase)
                last_switch = step
            
            traci.simulationStep()
            
            # Calculate reward
            waiting = sum(traci.lane.getWaitingTime(l) for l in controlled_lanes[:4])
            queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in controlled_lanes[:4])
            reward = -0.001 * waiting - 0.01 * queue
            
            total_reward += reward
            total_waiting += waiting
        
        traci.close()
        
        avg_waiting = total_waiting / 500
        all_rewards.append(total_reward)
        all_waiting.append(avg_waiting)
        
        print(f"  Episode {ep+1}: Reward={total_reward:.2f}, Avg Waiting={avg_waiting:.2f}s")
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Average Reward: {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")
    print(f"  Average Waiting Time: {np.mean(all_waiting):.2f}s")
    print(f"  Best Episode Reward: {max(all_rewards):.2f}")
    
    return np.mean(all_rewards)


if __name__ == "__main__":
    model_path = MODELS_DIR / "dqn_hebbal_final.pth"
    
    if model_path.exists():
        evaluate_model("hebbal", str(model_path), episodes=3)
    else:
        print(f"Model not found: {model_path}")
        print("Run training first: python scripts/train_perfect.py --junction hebbal --episodes 100")
