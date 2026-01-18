import traci
import time
import sys
import os
import uuid
from typing import Dict, Tuple

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from heuristic_agent import HeuristicAgent
from controllers.fixed_timer_controller import FixedTimerController

def safe_close_all_traci():
    """Force close all existing TraCI connections safely"""
    try:
        # Get all connection label names
        # Note: getConnectionNames might fail if no connections, so wrap in try
        try:
            connections = traci.getConnectionLabelList()
        except AttributeError:
             # Fallback for older traci versions
             try:
                 connections = traci.getConnectionNames()
             except:
                 connections = []
        except:
             connections = []
             
        for conn in connections:
            try:
                traci.switch(conn)
                traci.close()
                print(f"🧹 Closed TraCI connection: {conn}")
            except Exception as e:
                # If switch/close fails, it might be already dead
                pass
    except Exception:
        pass

class DualSimulationManager:
    """
    Manages two parallel SUMO simulations for comparison
    CRITICAL: Both simulations use the SAME route file (identical traffic)
    """
    
    def __init__(self,
                 network_file: str,
                 route_file: str,
                 gui: bool = True):
        """
        Initialize dual simulation
        """
        self.network_file = network_file
        self.route_file = route_file
        self.gui = gui
        
        # Ensure files exist (same as before...)
        if not os.path.exists(network_file):
            if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), network_file)):
                 self.network_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), network_file)
        
        if not os.path.exists(route_file):
             if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), route_file)):
                 self.route_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), route_file)
        
        if not os.path.exists(self.network_file):
            raise FileNotFoundError(f"Network not found: {self.network_file}")
        if not os.path.exists(self.route_file):
            raise FileNotFoundError(f"Routes not found: {self.route_file}")
        
        # Simulation state
        self.is_running = False
        self.current_time = 0.0
        
        # Generate UNIQUE labels for this run
        run_id = uuid.uuid4().hex[:6]
        self.fixed_label = f"fixed_{run_id}"
        self.heuristic_label = f"heuristic_{run_id}"
        
        # Agents
        self.fixed_agents = {}
        self.heuristic_agents = {}
        
        # Metrics
        self.fixed_metrics = {
            'total_spawned': 0,
            'total_arrived': 0,
            'cumulative_wait': 0.0,
            'wait_times': {}
        }
        
        self.heuristic_metrics = {
            'total_spawned': 0,
            'total_arrived': 0,
            'cumulative_wait': 0.0,
            'wait_times': {}
        }
        
        print("🎮 Dual Simulation Manager initialized")
    
    def start(self):
        """Start both SUMO simulations"""
        
        # 1. CLEANUP PREVIOUS MESS (defensive)
        safe_close_all_traci()
        
        print("\n" + "="*60)
        print("🚀 STARTING DUAL SIMULATION")
        print("="*60)
        
        sumo_binary = 'sumo-gui' if self.gui else 'sumo'
        abs_net = os.path.abspath(self.network_file)
        abs_route = os.path.abspath(self.route_file)
        
        # SUMO command for FIXED TIMER
        sumo_cmd_fixed = [
            sumo_binary,
            '-n', abs_net,
            '-r', abs_route,
            '--step-length', '0.1',
            '--no-warnings',
            '--start',
            '--quit-on-end',
            '--delay', '50',
            '--window-size', '800,600',
            '--window-pos', '50,50',
            '--gui-settings-file', self._create_gui_settings('fixed')
        ]
        
        # SUMO command for HEURISTIC
        sumo_cmd_heuristic = [
            sumo_binary,
            '-n', abs_net,
            '-r', abs_route,
            '--step-length', '0.1',
            '--no-warnings',
            '--start',
            '--quit-on-end',
            '--delay', '50',
            '--window-size', '800,600',
            '--window-pos', '900,50',
            '--gui-settings-file', self._create_gui_settings('heuristic')
        ]
        
        try:
            # Start FIXED TIMER simulation
            print(f"\n1️⃣  Starting FIXED TIMER simulation ({self.fixed_label})...")
            # Using port 0 allows OS to assign free port, avoiding "port busy" errors entirely
            # But we must use different ports for side-by-side
            
            try:
                # Try finding a free port or just let traci handle it
                traci.start(sumo_cmd_fixed, label=self.fixed_label)
            except Exception as e:
                print(f"Retrying fixed start: {e}")
                time.sleep(1)
                traci.start(sumo_cmd_fixed, label=self.fixed_label)

            time.sleep(1)
            print("✅ Fixed simulation connected")
            
            # Start HEURISTIC simulation
            print(f"\n2️⃣  Starting HEURISTIC simulation ({self.heuristic_label})...")
            try:
                traci.start(sumo_cmd_heuristic, label=self.heuristic_label)
            except Exception as e:
                 print(f"Retrying heuristic start: {e}")
                 time.sleep(1)
                 traci.start(sumo_cmd_heuristic, label=self.heuristic_label)

            time.sleep(1)
            print("✅ Heuristic simulation connected")
            
            # Discover and create agents for both
            self._create_agents()
            
            self.is_running = True
            
            print("\n" + "="*60)
            print("✅ BOTH SIMULATIONS READY")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Failed to start simulations: {e}")
            self.close()
            raise
    
    def _create_gui_settings(self, label: str) -> str:
        """Create GUI settings file"""
        settings_file = f'sumo_gui_{label}.xml'
        settings_file = os.path.abspath(settings_file)
        
        settings_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<viewsettings>
    <scheme name="real world"/>
    <viewport zoom="150" x="0" y="0"/>
    <delay value="50"/>
</viewsettings>"""
        
        with open(settings_file, 'w') as f:
            f.write(settings_xml)
        
        return settings_file
    
    def _create_agents(self):
        """Create agents for both simulations"""
        
        # Switch to FIXED simulation
        traci.switch(self.fixed_label)
        tls_ids = traci.trafficlight.getIDList()
        
        print(f"\n🔍 Discovered {len(tls_ids)} traffic lights")
        
        for tls_id in tls_ids:
            self.fixed_agents[tls_id] = FixedTimerController(
                tls_id=tls_id,
                green_duration=30.0,
                yellow_duration=3.0
            )
        
        # Switch to HEURISTIC simulation
        traci.switch(self.heuristic_label)
        
        for tls_id in tls_ids:
            self.heuristic_agents[tls_id] = HeuristicAgent(
                tls_id=tls_id,
                min_green_time=10.0,
                max_green_time=60.0
            )
        
        print(f"\n✅ Created agents for both simulations")
    
    def step(self) -> Dict:
        """
        Execute one synchronized step in both simulations
        """
        if not self.is_running:
            raise Exception("Simulations not running")
        
        try:
            # Step FIXED simulation
            traci.switch(self.fixed_label)
            for agent in self.fixed_agents.values():
                try:
                    agent.step(self.current_time)
                except traci.exceptions.FatalTraCIError:
                    print("💥 Fixed simulation closed/crashed")
                    self.close()
                    raise Exception("Simulation closed by user")
            
            try:
                traci.simulationStep()
            except traci.exceptions.FatalTraCIError:
                 print("💥 Fixed simulation closed/crashed during step")
                 self.close()
                 raise Exception("Simulation closed by user")
                 
            self._update_metrics('fixed')
            
            # Step HEURISTIC simulation
            traci.switch(self.heuristic_label)
            for agent in self.heuristic_agents.values():
                try:
                    agent.execute_action(self.current_time)
                except traci.exceptions.FatalTraCIError:
                    print("💥 Heuristic simulation closed/crashed")
                    self.close()
                    raise Exception("Simulation closed by user")

            try:
                traci.simulationStep()
            except traci.exceptions.FatalTraCIError:
                print("💥 Heuristic simulation closed/crashed during step")
                self.close()
                raise Exception("Simulation closed by user")
                
            self._update_metrics('heuristic')
            
            # Increment time
            self.current_time += 0.1
            
            # Return combined state
            return self.get_comparison()
            
        except traci.exceptions.FatalTraCIError:
            print("💥 Fatal TraCI Error during check")
            self.close()
            raise Exception("Simulation crashed or closed")

    def _update_metrics(self, label: str):
        """Update metrics for one simulation"""
        metrics = self.fixed_metrics if label == 'fixed' else self.heuristic_metrics
        
        # Track new vehicles
        departed = traci.simulation.getDepartedNumber()
        metrics['total_spawned'] += departed
        
        # Track completed vehicles
        arrived_ids = traci.simulation.getArrivedIDList()
        for veh_id in arrived_ids:
            metrics['total_arrived'] += 1
            if veh_id in metrics['wait_times']:
                metrics['cumulative_wait'] += metrics['wait_times'][veh_id]
                del metrics['wait_times'][veh_id]
        
        # Update waiting times
        for veh_id in traci.vehicle.getIDList():
            metrics['wait_times'][veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
    
    def get_comparison(self) -> Dict:
        """Get comparison between fixed and heuristic"""
        
        fixed_avg_wait = (
            self.fixed_metrics['cumulative_wait'] / self.fixed_metrics['total_arrived']
            if self.fixed_metrics['total_arrived'] > 0 else 0.0
        )
        
        heuristic_avg_wait = (
            self.heuristic_metrics['cumulative_wait'] / self.heuristic_metrics['total_arrived']
            if self.heuristic_metrics['total_arrived'] > 0 else 0.0
        )
        
        improvement = (
            ((fixed_avg_wait - heuristic_avg_wait) / fixed_avg_wait * 100)
            if fixed_avg_wait > 0 else 0.0
        )
        
        return {
            'time': round(self.current_time, 1),
            'fixed': {
                'avg_wait_time': round(fixed_avg_wait, 2),
                'total_arrived': self.fixed_metrics['total_arrived'],
                'active_vehicles': len(self.fixed_metrics['wait_times'])
            },
            'heuristic': {
                'avg_wait_time': round(heuristic_avg_wait, 2),
                'total_arrived': self.heuristic_metrics['total_arrived'],
                'active_vehicles': len(self.heuristic_metrics['wait_times'])
            },
            'improvement_percentage': round(improvement, 2),
            'throughput_delta': self.heuristic_metrics['total_arrived'] - self.fixed_metrics['total_arrived']
        }
    
    def close(self):
        """Close both simulations"""
        print("\n🛑 Closing simulations...")
        
        try:
            traci.switch(self.fixed_label)
            traci.close()
        except:
            pass
        
        try:
            traci.switch(self.heuristic_label)
            traci.close()
        except:
            pass
            
        # Also try cleaning up any others
        safe_close_all_traci()
        
        self.is_running = False
        print("✅ Both simulations closed")
