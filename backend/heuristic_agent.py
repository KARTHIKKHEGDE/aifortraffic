"""
Universal Heuristic Traffic Control Agent
Works with ANY SUMO network - single intersection, grids, or real city maps
"""

import traci
import numpy as np
from typing import Dict, List, Tuple, Optional

class HeuristicAgent:
    """
    Intelligent heuristic-based traffic controller
    
    Works with:
    - 3-way intersections (T-junctions)
    - 4-way intersections (standard cross)
    - Complex multi-phase intersections
    - Real city networks (Bangalore, etc.)
    
    Decision Rules:
    1. Emergency vehicle priority
    2. Minimum green time (prevent rapid switching)
    3. Maximum green time (prevent starvation)
    4. Queue imbalance (switch if one direction much longer)
    5. Waiting time threshold (fairness)
    """
    
    def __init__(self,
                 tls_id: str,
                 min_green_time: float = 10.0,
                 max_green_time: float = 60.0,
                 yellow_time: float = 3.0,
                 queue_threshold: int = 5,
                 wait_time_threshold: float = 45.0):
        """
        Initialize heuristic agent for a traffic light
        
        Args:
            tls_id: Traffic light system ID in SUMO
            min_green_time: Minimum green duration (seconds)
            max_green_time: Maximum green duration (seconds)
            yellow_time: Yellow phase duration (seconds)
            queue_threshold: Queue difference to trigger switch
            wait_time_threshold: Max wait before forcing switch
        """
        self.tls_id = tls_id
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        self.yellow_time = yellow_time
        self.queue_threshold = queue_threshold
        self.wait_time_threshold = wait_time_threshold
        
        # Discover junction structure
        self._discover_junction_structure()
        
        # State tracking
        self.current_phase = 0
        self.phase_start_time = 0.0
        self.in_yellow = False
        self.next_phase = None
        
        # Performance metrics
        self.total_switches = 0
        self.emergency_interventions = 0
        
        print(f"✅ Heuristic Agent initialized: {tls_id}")
        print(f"   Type: {self.junction_type}")
        print(f"   Phases: {self.num_green_phases} green phases")
        print(f"   Controlled lanes: {len(self.controlled_lanes)}")
    
    def _discover_junction_structure(self):
        """Automatically discover junction structure from SUMO"""
        try:
            # Get traffic light program
            programs = traci.trafficlight.getAllProgramLogics(self.tls_id)
            
            if not programs:
                raise Exception(f"No traffic light program found for {self.tls_id}")
            
            program = programs[0]
            self.phases = program.phases
            
            # Classify phases (green vs yellow/red)
            self.green_phases = []
            self.yellow_phases = []
            
            for i, phase in enumerate(self.phases):
                state = phase.state
                if 'G' in state or 'g' in state:
                    self.green_phases.append(i)
                elif 'y' in state or 'Y' in state:
                    self.yellow_phases.append(i)
            
            self.num_green_phases = len(self.green_phases)
            
            # Get controlled lanes
            self.controlled_lanes = list(set(traci.trafficlight.getControlledLanes(self.tls_id)))
            
            # Get controlled links (for understanding phase structure)
            self.controlled_links = traci.trafficlight.getControlledLinks(self.tls_id)
            
            # Classify junction type
            num_lanes = len(self.controlled_lanes)
            if num_lanes <= 4:
                self.junction_type = "simple"
            elif num_lanes <= 8:
                self.junction_type = "4-way"
            elif num_lanes <= 12:
                self.junction_type = "complex"
            else:
                self.junction_type = "major"
            
        except Exception as e:
            print(f"⚠️ Error discovering junction structure: {e}")
            self.green_phases = [0, 2]
            self.yellow_phases = [1, 3]
            self.num_green_phases = 2
            self.controlled_lanes = []
            self.junction_type = "unknown"
    
    def get_traffic_state(self, current_time: float) -> Dict:
        """
        Get current traffic state at this junction
        
        Returns dict with:
        - queue_lengths: per-lane queue counts (stopped vehicles)
        - total_waiting_times: per-lane sum of waiting times
        - max_waiting_times: per-lane max waiting time (single vehicle)
        - approaching_count: count of moving vehicles approaching green
        - emergency_present: list of emergency vehicle IDs
        - phase_duration: how long current phase has been active
        """
        queue_lengths = {}
        total_waiting_times = {}
        max_waiting_times = {}
        approaching_counts = {}
        emergency_vehicles = []
        
        for lane in self.controlled_lanes:
            try:
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                
                stopped_count = 0
                lane_total_wait = 0.0
                lane_max_wait = 0.0
                approaching = 0
                
                for veh_id in vehicle_ids:
                    speed = traci.vehicle.getSpeed(veh_id)
                    wait_time = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                    
                    if speed < 1.0:  # Vehicle is stopped
                        stopped_count += 1
                        lane_total_wait += wait_time
                        if wait_time > lane_max_wait:
                            lane_max_wait = wait_time
                    else:
                        # Moving vehicle
                        approaching += 1
                        
                    # Check if emergency
                    vtype = traci.vehicle.getTypeID(veh_id)
                    if vtype == 'emergency':
                        emergency_vehicles.append(veh_id)
                
                queue_lengths[lane] = stopped_count
                total_waiting_times[lane] = lane_total_wait
                max_waiting_times[lane] = lane_max_wait
                approaching_counts[lane] = approaching
                
            except Exception:
                queue_lengths[lane] = 0
                total_waiting_times[lane] = 0.0
                max_waiting_times[lane] = 0.0
                approaching_counts[lane] = 0
        
        phase_duration = current_time - self.phase_start_time
        
        return {
            'queue_lengths': queue_lengths,
            'total_waiting_times': total_waiting_times,
            'max_waiting_times': max_waiting_times,
            'approaching_counts': approaching_counts,
            'emergency_vehicles': emergency_vehicles,
            'phase_duration': phase_duration,
            'current_phase': self.current_phase
        }
    
    def decide_phase_change(self, state: Dict) -> bool:
        """
        Smart heuristic decision making
        Uses a weighted score combining Queue Length and Max Waiting Time
        """
        
        # Rule 0: Don't interrupt yellow
        if self.in_yellow:
            return False
        
        # Rule 1: Emergency vehicle priority (Immediate)
        if state['emergency_vehicles']:
            if state['phase_duration'] >= self.min_green_time * 0.5:
                print(f"🚑 Emergency detected at {self.tls_id}! Considering switch...")
                if self._emergency_on_red_lanes(state):
                    print(f"   Emergency on RED lane - switching!")
                    return True
        
        # Rule 2: Respect minimum green time
        if state['phase_duration'] < self.min_green_time:
            return False
        
        # Rule 3: Force switch at maximum green time
        if state['phase_duration'] >= self.max_green_time:
            print(f"⏰ Max green time reached at {self.tls_id}")
            return True

        # --- SCORING SYSTEM ---
        # Calculate urgency scores for current green phase vs next potential phase
        
        # Factors
        W_QUEUE = 1.0       # Weight for number of stopped cars
        W_WAIT = 0.5        # Weight for max waiting time (fairness)
        W_APPROACH = 0.2    # Weight for approaching cars (green extension)
        
        green_lanes = self._get_lanes_for_phase(self.current_phase)
        next_green_phase_idx = self._get_next_green_phase()
        red_lanes = self._get_lanes_for_phase(next_green_phase_idx)
        
        # Calculate GREEN Score (Keep Current)
        green_score = 0.0
        for lane in green_lanes:
            q = state['queue_lengths'].get(lane, 0)
            w = state['max_waiting_times'].get(lane, 0)
            a = state['approaching_counts'].get(lane, 0)
            # Logic: If I keep green, I serve these queues + approaching
            green_score += (q * W_QUEUE) + (a * W_APPROACH)
            
        # Calculate RED Score (Switch Now)
        red_score = 0.0
        max_red_wait_actual = 0.0
        for lane in red_lanes:
            q = state['queue_lengths'].get(lane, 0)
            w = state['max_waiting_times'].get(lane, 0)
            if w > max_red_wait_actual:
                max_red_wait_actual = w
            
            # Logic: If I switch, I relieve this pressure
            red_score += (q * W_QUEUE) + (w * W_WAIT) # Wait time matters more for red lanes (starvation)

        # Rule 4: Anti-Starvation / Fairness Override
        # If any car has waited too long on red, switch regardless of green flow
        if max_red_wait_actual > self.wait_time_threshold:
            print(f"⏳ Fairness trigger: Vehicle waited {max_red_wait_actual:.1f}s")
            return True
            
        # Rule 5: Gap Logic (Actuated Control)
        # If green is empty (no queue, no approaching), but red has ANYONE, switch
        if green_score < 0.1 and red_score > 0:
             print(f"🟢 Gap detected: Green empty, switching to serve red")
             return True

        # Rule 6: Weighted Pressure Comparison
        # Switch if Red urgency significantly outweighs keeping Green
        # We add a hysteresis factor (10-20%) to prevent rapid oscillation or changing for 1 car
        hysteresis = 1.2 # Red must be 20% more urgent
        
        if red_score > (green_score * hysteresis) + 2.0: # +2 buffer
            print(f"⚖️ Pressure Switch: Red Score ({red_score:.1f}) > Green Score ({green_score:.1f})")
            return True
            
        return False

    def _get_lanes_for_phase(self, phase_index: int) -> List[str]:
        """Get list of lanes that have green in the specified phase"""
        if phase_index >= len(self.phases):
            return []
            
        phase_state = self.phases[phase_index].state
        lanes = []
        for i, lane in enumerate(self.controlled_lanes):
            if i < len(phase_state) and phase_state[i] in ['G', 'g']:
                lanes.append(lane)
        return lanes

    def _emergency_on_red_lanes(self, state: Dict) -> bool:
        """Check if emergency vehicle is waiting on red lanes"""
        next_phase_idx = self._get_next_green_phase()
        next_lanes = self._get_lanes_for_phase(next_phase_idx)
        
        for lane in next_lanes:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            for veh_id in state['emergency_vehicles']:
                if veh_id in vehicle_ids:
                    return True
        return False
    
    # ... helpers ...
    def _get_next_green_phase(self) -> int:
        """Get index of next green phase"""
        current_idx = self.green_phases.index(self.current_phase) if self.current_phase in self.green_phases else 0
        next_idx = (current_idx + 1) % self.num_green_phases
        return self.green_phases[next_idx]
    
    def _find_yellow_phase(self, from_phase: int, to_phase: int) -> Optional[int]:
        """Find appropriate yellow phase"""
        for i in range(from_phase + 1, min(from_phase + 3, len(self.phases))):
            if i in self.yellow_phases:
                return i
        if self.yellow_phases:
            return self.yellow_phases[0]
        return None
    
    def execute_action(self, current_time: float):
        """Execute traffic light control action"""
        state = self.get_traffic_state(current_time)
        
        if self.in_yellow:
            if state['phase_duration'] >= self.yellow_time:
                self.current_phase = self.next_phase
                traci.trafficlight.setPhase(self.tls_id, self.current_phase)
                self.in_yellow = False
                self.next_phase = None
                self.phase_start_time = current_time
                self.total_switches += 1
            return
        
        if self.decide_phase_change(state):
            next_green = self._get_next_green_phase()
            yellow_phase = self._find_yellow_phase(self.current_phase, next_green)
            
            if yellow_phase is not None:
                traci.trafficlight.setPhase(self.tls_id, yellow_phase)
                self.in_yellow = True
                self.next_phase = next_green
                self.phase_start_time = current_time
            else:
                self.current_phase = next_green
                traci.trafficlight.setPhase(self.tls_id, self.current_phase)
                self.phase_start_time = current_time
                self.total_switches += 1
    
    def get_metrics(self) -> Dict:
        return {
            'tls_id': self.tls_id,
            'total_switches': self.total_switches,
            'emergency_interventions': self.emergency_interventions,
            'current_phase': self.current_phase,
            'junction_type': self.junction_type
        }
