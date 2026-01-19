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
                print(f"🧹 Force-closed TraCI connection: {conn}")
            except Exception:
                pass
    except Exception:
        pass

class DualSimulationManager:
    """
    Manages two parallel SUMO simulations for comparison
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
        
        # Ensure files exist
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
        
        # Initial labels (will be rotated on start)
        self.fixed_label = f"fixed_{uuid.uuid4().hex[:6]}"
        self.heuristic_label = f"heuristic_{uuid.uuid4().hex[:6]}"
        
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
        
        # --- FIXED TIMER SIMULATION ---
        print(f"\n1️⃣  Starting FIXED TIMER simulation...")
        
        # Generate Fresh Label
        self.fixed_label = f"fixed_{uuid.uuid4().hex[:6]}"
        
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
            '--gui-settings-file', self._create_gui_settings('fixed'),
            '--log', f'sumo_fixed.log',
            '--collision.action', 'none',
            '--collision.check-junctions', 'true',
            '--collision.mingap-factor', '0'
        ]
        
        try:
            traci.start(sumo_cmd_fixed, label=self.fixed_label)
        except Exception as e:
            print(f"⚠️ First attempt failed: {e}")
            # Retry with NEW label
            self.fixed_label = f"fixed_{uuid.uuid4().hex[:6]}"
            print(f"🔄 Retrying with new label: {self.fixed_label}")
            traci.start(sumo_cmd_fixed, label=self.fixed_label)

        # WARMUP STEPS (Critical for 12-intersection grid)
        traci.switch(self.fixed_label)
        for _ in range(5):
            traci.simulationStep()
            
        print(f"✅ Fixed simulation connected ({self.fixed_label})")
        
        
        # --- HEURISTIC SIMULATION ---
        print(f"\n2️⃣  Starting HEURISTIC simulation...")
        
        # Generate Fresh Label
        self.heuristic_label = f"heuristic_{uuid.uuid4().hex[:6]}"
        
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
            '--gui-settings-file', self._create_gui_settings('heuristic'),
            '--log', f'sumo_heuristic.log',
            '--collision.action', 'none',
            '--collision.check-junctions', 'true',
            '--collision.mingap-factor', '0'
        ]
        
        try:
            traci.start(sumo_cmd_heuristic, label=self.heuristic_label)
        except Exception as e:
            print(f"⚠️ First attempt failed: {e}")
            # Retry with NEW label
            self.heuristic_label = f"heuristic_{uuid.uuid4().hex[:6]}"
            print(f"🔄 Retrying with new label: {self.heuristic_label}")
            traci.start(sumo_cmd_heuristic, label=self.heuristic_label)
            
        # WARMUP STEPS
        traci.switch(self.heuristic_label)
        for _ in range(5):
            traci.simulationStep()
            
        print(f"✅ Heuristic simulation connected ({self.heuristic_label})")

        # Create Agents
        self._create_agents()
        self.is_running = True
        
        print("\n" + "="*60)
        print("✅ BOTH SIMULATIONS READY")
        print("="*60)

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
        print(f"\n🔍 Discovered {len(tls_ids)} traffic lights (Fixed)")
        for tls_id in tls_ids:
            self.fixed_agents[tls_id] = FixedTimerController(tls_id=tls_id)
        
        # Switch to HEURISTIC simulation
        traci.switch(self.heuristic_label)
        tls_ids = traci.trafficlight.getIDList()
        print(f"\n🔍 Discovered {len(tls_ids)} traffic lights (Heuristic)")
        for tls_id in tls_ids:
            self.heuristic_agents[tls_id] = HeuristicAgent(tls_id=tls_id)
        
        print(f"✅ Agents initialized")
    
    def step(self) -> Dict:
        """Execute one synchronized step"""
        if not self.is_running:
            raise Exception("Simulations not running")
        
        try:
            # FIXED
            traci.switch(self.fixed_label)
            for agent in self.fixed_agents.values():
                try:
                    agent.step(self.current_time)
                except traci.exceptions.FatalTraCIError:
                    self._handle_crash("Fixed")
            try:
                traci.simulationStep()
                self._update_metrics('fixed')
            except traci.exceptions.FatalTraCIError:
                 self._handle_crash("Fixed")

            # HEURISTIC
            traci.switch(self.heuristic_label)
            for agent in self.heuristic_agents.values():
                try:
                    agent.execute_action(self.current_time)
                except traci.exceptions.FatalTraCIError:
                    self._handle_crash("Heuristic")
            try:
                traci.simulationStep()
                self._update_metrics('heuristic')
            except traci.exceptions.FatalTraCIError:
                self._handle_crash("Heuristic")
                
            self.current_time += 0.1
            return self.get_comparison()
            
        except Exception as e:
            print(f"💥 Simulation Error: {e}")
            self.close()
            raise

    def _handle_crash(self, name: str):
        print(f"💥 {name} simulation closed/crashed")
        self.close()
        raise Exception("Simulation closed by user")

    def _update_metrics(self, label: str):
        """Update metrics for one simulation"""
        metrics = self.fixed_metrics if label == 'fixed' else self.heuristic_metrics
        
        try:
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
        except:
            pass
    
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
        
        # Calc active vehicles for real-time count
        fixed_active = len(self.fixed_metrics['wait_times'])
        try:
            traci.switch(self.heuristic_label) # Just to be safe querying traci
            heuristic_active = len(self.heuristic_metrics['wait_times'])
        except:
             heuristic_active = 0
        
        improvement = (
            ((fixed_avg_wait - heuristic_avg_wait) / fixed_avg_wait * 100)
            if fixed_avg_wait > 0 else 0.0
        )
        
        # Aggregate heuristic agent stats
        total_early_term = sum(agent.early_terminations for agent in self.heuristic_agents.values())
        total_extended = sum(agent.extended_phases for agent in self.heuristic_agents.values())
        total_emergency = sum(agent.emergency_interventions for agent in self.heuristic_agents.values())
        total_starvation = sum(agent.starvation_prevents for agent in self.heuristic_agents.values())
        total_congestion = sum(agent.congestion_responses for agent in self.heuristic_agents.values())
        total_heuristic_switches = sum(agent.total_switches for agent in self.heuristic_agents.values())
        
        # Fixed timer switches for comparison
        total_fixed_switches = sum(agent.total_switches for agent in self.fixed_agents.values())
        
        return {
            'time': round(self.current_time, 1),
            'fixed': {
                'avg_wait_time': round(fixed_avg_wait, 2),
                'total_arrived': self.fixed_metrics['total_arrived'],
                'active_vehicles': fixed_active,
                'total_switches': total_fixed_switches
            },
            'heuristic': {
                'avg_wait_time': round(heuristic_avg_wait, 2),
                'total_arrived': self.heuristic_metrics['total_arrived'],
                'active_vehicles': heuristic_active,
                'total_switches': total_heuristic_switches,
                'early_terminations': total_early_term,
                'extended_phases': total_extended,
                'emergency_interventions': total_emergency,
                'starvation_prevents': total_starvation,
                'congestion_responses': total_congestion
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
            
        # Last resort cleanup
        safe_close_all_traci()
        
        self.is_running = False
        print("✅ Both simulations closed")
