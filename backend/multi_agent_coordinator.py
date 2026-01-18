"""
Multi-Agent Coordinator
Manages multiple heuristic agents across a network
"""

import traci
from typing import Dict, List
from heuristic_agent import HeuristicAgent

class MultiAgentCoordinator:
    """
    Coordinates multiple heuristic agents
    Automatically discovers all traffic lights in any SUMO network
    """
    
    def __init__(self,
                 min_green_time: float = 10.0,
                 max_green_time: float = 60.0,
                 coordination_enabled: bool = False):
        """
        Initialize coordinator
        
        Args:
            min_green_time: Minimum green for all agents
            max_green_time: Maximum green for all agents
            coordination_enabled: Enable inter-agent coordination (future feature)
        """
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        self.coordination_enabled = coordination_enabled
        
        self.agents: Dict[str, HeuristicAgent] = {}
        
        # Network-wide metrics
        self.total_spawned = 0
        self.total_arrived = 0
        self.cumulative_wait = 0.0
        self.wait_times = {}
        
        print("🤖 Multi-Agent Coordinator initialized")
    
    def discover_and_create_agents(self):
        """
        Automatically discover all traffic lights and create agents
        Works with ANY SUMO network
        """
        try:
            tls_ids = traci.trafficlight.getIDList()
            
            print(f"\n🔍 Discovered {len(tls_ids)} traffic lights in network")
            
            for tls_id in tls_ids:
                agent = HeuristicAgent(
                    tls_id=tls_id,
                    min_green_time=self.min_green_time,
                    max_green_time=self.max_green_time
                )
                self.agents[tls_id] = agent
            
            print(f"\n✅ Created {len(self.agents)} heuristic agents\n")
            
        except Exception as e:
            print(f"❌ Error creating agents: {e}")
    
    def step(self, current_time: float):
        """Execute one step for all agents"""
        # Update network metrics
        self._update_metrics()
        
        # Execute each agent
        for agent in self.agents.values():
            try:
                agent.execute_action(current_time)
            except Exception as e:
                print(f"⚠️ Error in agent {agent.tls_id}: {e}")
    
    def _update_metrics(self):
        """Update network-wide performance metrics"""
        try:
            # Track new vehicles
            departed = traci.simulation.getDepartedNumber()
            self.total_spawned += departed
            
            # Track completed vehicles
            arrived_ids = traci.simulation.getArrivedIDList()
            for veh_id in arrived_ids:
                self.total_arrived += 1
                if veh_id in self.wait_times:
                    self.cumulative_wait += self.wait_times[veh_id]
                    del self.wait_times[veh_id]
            
            # Update waiting times
            for veh_id in traci.vehicle.getIDList():
                self.wait_times[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
        
        except Exception:
            pass
    
    def get_metrics(self) -> Dict:
        """Get network-wide metrics"""
        avg_wait = (
            self.cumulative_wait / self.total_arrived
            if self.total_arrived > 0 else 0.0
        )
        
        return {
            'total_spawned': self.total_spawned,
            'total_arrived': self.total_arrived,
            'active_vehicles': len(traci.vehicle.getIDList()),
            'avg_wait_time': round(avg_wait, 2),
            'num_agents': len(self.agents)
        }
    
    def get_all_agent_metrics(self) -> Dict:
        """Get metrics from all agents"""
        return {
            agent.tls_id: agent.get_metrics()
            for agent in self.agents.values()
        }
    
    def print_status(self, current_time: float):
        """Print current status"""
        metrics = self.get_metrics()
        
        print(f"\n⏱️  Time: {current_time:.1f}s")
        print(f"   Vehicles: {metrics['active_vehicles']} active, "
              f"{metrics['total_arrived']} completed")
        print(f"   Avg Wait: {metrics['avg_wait_time']:.1f}s")
        print(f"   Agents: {metrics['num_agents']}")
