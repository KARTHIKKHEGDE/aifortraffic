"""
Run heuristic control on multiple intersections (e.g., 12-junction grid)
"""

import traci
import time
from multi_agent_coordinator import MultiAgentCoordinator

def run_multi_intersection(
    network_file: str = 'sumo/networks/grid_3x4.net.xml',  # 3x4 = 12 intersections
    route_file: str = 'sumo/routes/grid_3x4.rou.xml',
    duration: float = 1800.0,  # 30 minutes
    gui: bool = True,
    enable_coordination: bool = True,
    print_interval: float = 30.0
):
    """
    Run heuristic control on multiple intersections
    
    Args:
        network_file: SUMO network file (grid or custom)
        route_file: SUMO route file
        duration: Simulation duration
        gui: Show GUI
        enable_coordination: Enable inter-agent coordination
        print_interval: Status print interval
    """
    
    sumo_binary = 'sumo-gui' if gui else 'sumo'
    sumo_cmd = [
        sumo_binary,
        '-n', network_file,
        '-r', route_file,
        '--step-length', '0.1',
        '--no-warnings',
        '--quit-on-end',
        '--start',
        '--collision.action', 'warn'
    ]
    
    try:
        print("🚀 Starting SUMO with multi-intersection network...")
        traci.start(sumo_cmd)
        
        # Create coordinator with coordination enabled
        coordinator = MultiAgentCoordinator(
            enable_coordination=enable_coordination,
            min_green_time=8.0,  # Shorter for urban grid
            max_green_time=45.0
        )
        
        # Auto-discover all intersections
        coordinator.discover_and_create_agents()
        
        print(f"\n▶️ Running {duration}s simulation with {len(coordinator.agents)} agents...\n")
        
        current_time = 0.0
        last_print_time = 0.0
        
        while current_time < duration:
            coordinator.step(current_time)
            traci.simulationStep()
            
            current_time += 0.1
            
            if current_time - last_print_time >= print_interval:
                coordinator.print_status(current_time)
                last_print_time = current_time
        
        # Final results
        print("\n" + "=" * 70)
        print("✅ MULTI-INTERSECTION SIMULATION COMPLETE")
        print("=" * 70)
        
        metrics = coordinator.get_network_metrics()
        print(f"\nNetwork Performance:")
        print(f"  Total Vehicles Completed: {metrics['total_arrived']}")
        print(f"  Average Wait Time: {metrics['avg_wait_time']:.1f}s")
        print(f"  Number of Intersections: {metrics['num_agents']}")
        
        print(f"\nPer-Agent Performance:")
        agent_metrics = coordinator.get_all_agent_metrics()
        for agent_id, agent_data in agent_metrics.items():
            print(f"  {agent_id}: {agent_data['total_phase_changes']} changes, "
                  f"{agent_data['emergency_interventions']} emergencies")
        
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n⏸️ Simulation interrupted")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        traci.close()
        print("\n🛑 SUMO closed")

if __name__ == '__main__':
    run_multi_intersection(
        network_file='sumo/networks/grid_3x4.net.xml',
        route_file='sumo/routes/grid_3x4.rou.xml',
        gui=True,
        enable_coordination=True
    )
