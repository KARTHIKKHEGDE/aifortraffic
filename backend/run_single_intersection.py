"""
Run heuristic control on a single intersection
"""

import traci
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multi_agent_coordinator import MultiAgentCoordinator

def run_single_intersection(
    network_file: str,
    route_file: str,
    duration: float = 900.0,
    gui: bool = True
):
    """
    Run heuristic control on single intersection
    
    Args:
        network_file: Path to .net.xml file
        route_file: Path to .rou.xml file
        duration: Simulation duration in seconds
        gui: Show SUMO GUI
    """
    
    # Check files exist
    if not os.path.exists(network_file):
        print(f"❌ Network file not found: {network_file}")
        return
    
    if not os.path.exists(route_file):
        print(f"❌ Route file not found: {route_file}")
        return
    
    # SUMO command
    sumo_binary = 'sumo-gui' if gui else 'sumo'
    sumo_cmd = [
        sumo_binary,
        '-n', network_file,
        '-r', route_file,
        '--step-length', '0.1',
        '--no-warnings',
        '--quit-on-end',
        '--start',
        '--delay', '0'
    ]
    
    try:
        print("🚀 Starting SUMO...")
        print(f"   Network: {os.path.basename(network_file)}")
        print(f"   Routes: {os.path.basename(route_file)}")
        print(f"   Duration: {duration}s")
        
        traci.start(sumo_cmd)
        
        # Small delay for SUMO to fully start
        time.sleep(1)
        
        # Create coordinator
        coordinator = MultiAgentCoordinator(
            min_green_time=10.0,
            max_green_time=60.0
        )
        
        # Discover agents
        coordinator.discover_and_create_agents()
        
        print(f"\n▶️  Running simulation (Target Speed: 3x Real-Time)...\n")
        
        current_time = 0.0
        last_print = 0.0
        
        # Timing for speed control
        target_speed_ratio = 3.0
        start_wall_time = time.perf_counter()
        start_sim_time = 0.0
        
        while current_time < duration:
            # Step all agents
            coordinator.step(current_time)
            
            # Step SUMO
            traci.simulationStep()
            
            current_time += 0.1
            
            # --- Speed Control ---
            sim_elapsed = current_time - start_sim_time
            target_wall_elapsed = sim_elapsed / target_speed_ratio
            actual_wall_elapsed = time.perf_counter() - start_wall_time
            
            sleep_needed = target_wall_elapsed - actual_wall_elapsed
            if sleep_needed > 0:
                time.sleep(sleep_needed)
            # ---------------------
            
            # Print status every 10 seconds
            if current_time - last_print >= 10.0:
                coordinator.print_status(current_time)
                last_print = current_time
        
        # Final results
        print("\n" + "=" * 60)
        print("✅ SIMULATION COMPLETE")
        print("=" * 60)
        
        metrics = coordinator.get_metrics()
        print(f"\nFinal Results:")
        print(f"  Total Vehicles: {metrics['total_arrived']}")
        print(f"  Average Wait Time: {metrics['avg_wait_time']:.1f}s")
        
        agent_metrics = coordinator.get_all_agent_metrics()
        for tls_id, data in agent_metrics.items():
            print(f"\n  Agent {tls_id}:")
            print(f"    Phase switches: {data['total_switches']}")
            print(f"    Emergency interventions: {data['emergency_interventions']}")
        
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⏸️  Simulation interrupted")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            traci.close()
            print("\n🛑 SUMO closed")
        except:
            pass

if __name__ == '__main__':
    # Run with default simple intersection
    run_single_intersection(
        network_file='sumo/networks/simple_intersection.net.xml',
        route_file='sumo/routes/simple_intersection.rou.xml',
        duration=900.0,
        gui=True
    )
