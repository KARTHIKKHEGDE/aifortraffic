"""
Heuristic Traffic Agent
Smart traffic light control logic with robust error handling
"""

import traci
import time
from typing import Dict, List, Optional, Tuple

class HeuristicAgent:
    """
    Intelligent Traffic Agent logic
    Uses heuristics (rules) to decide traffic light phases
    """
    
    def __init__(self, tls_id: str):
        self.tls_id = tls_id
        
        # Configuration
        self.min_green_time = 10.0
        self.max_green_time = 45.0
        self.yellow_time = 3.0
        self.wait_time_threshold = 40.0
        
        # Discover junction structure
        self._discover_junction_structure()
        
        # State
        self.current_phase = self.green_phases[0] if self.green_phases else 0
        self.phase_start_time = 0.0
        self.in_yellow = False
        self.next_phase = None
        self.total_switches = 0
        self.emergency_interventions = 0
        
        # Initialize
        try:
            traci.trafficlight.setPhase(self.tls_id, self.current_phase)
        except Exception as e:
            print(f"⚠️ Initial phase setup failed for {self.tls_id}: {e}")
            
        print(f"✅ Heuristic Agent initialized: {tls_id}")
        
    def _discover_junction_structure(self):
        """Analyze junction to understand phases"""
        try:
            programs = traci.trafficlight.getAllProgramLogics(self.tls_id)
            if not programs:
                raise Exception(f"No traffic light program found for {self.tls_id}")
                
            logic = programs[0]
            self.phases = logic.phases
            self.green_phases = []
            self.yellow_phases = []
            
            for i, phase in enumerate(self.phases):
                if 'G' in phase.state or 'g' in phase.state:
                    self.green_phases.append(i)
                elif 'y' in phase.state or 'Y' in phase.state:
                    self.yellow_phases.append(i)
                    
            # Determine controlled lanes
            all_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
            self.controlled_lanes = list(set(all_lanes)) 
        except Exception as e:
            print(f"❌ Structural discovery failed for {self.tls_id}: {e}")
            self.phases = []
            self.green_phases = [0]
            self.yellow_phases = []
            self.controlled_lanes = []
        
    def get_traffic_state(self, current_time: float) -> Dict:
        """Collect sensor data from simulation with robust error handling"""
        state = {
            'queue_lengths': {},
            'waiting_times': {},
            'max_waiting_times': {},
            'approaching_counts': {},
            'emergency_vehicles': False,
            'phase_duration': current_time - self.phase_start_time
        }
        
        for lane in self.controlled_lanes:
            try:
                # Queue Length
                state['queue_lengths'][lane] = traci.lane.getLastStepHaltingNumber(lane)
                
                # Waiting Time
                state['waiting_times'][lane] = traci.lane.getWaitingTime(lane)
                
                # Max Waiting Time (Individual vehicle)
                max_wait = 0.0
                vehs = traci.lane.getLastStepVehicleIDs(lane)
                for v in vehs:
                    try:
                        w = traci.vehicle.getAccumulatedWaitingTime(v)
                        if w > max_wait:
                            max_wait = w
                        if traci.vehicle.getTypeID(v) == 'emergency':
                            state['emergency_vehicles'] = True
                    except:
                        continue
                state['max_waiting_times'][lane] = max_wait
                
                # Approaching count
                approaching = 0
                for v in vehs:
                    try:
                        if traci.vehicle.getSpeed(v) > 2.0:
                            approaching += 1
                    except:
                        continue
                state['approaching_counts'][lane] = approaching
            except:
                continue
                
        return state

    def _get_vehicle_movements(self) -> Dict[str, Dict[str, int]]:
        """Count vehicles by direction and turn intention with error handling."""
        movements = { d: {'L': 0, 'S': 0, 'R': 0} for d in ['N', 'S', 'E', 'W'] }

        for lane in self.controlled_lanes:
            if lane.startswith(':'): continue
                
            try:
                veh_ids = traci.lane.getLastStepVehicleIDs(lane)
                for veh_id in veh_ids:
                    try:
                        # OMNISCIENT SENSING - check route
                        route = traci.vehicle.getRoute(veh_id)
                        current_edge = traci.vehicle.getRoadID(veh_id)
                        
                        if len(route) < 2: continue
                        
                        try:
                            idx = route.index(current_edge)
                            if idx + 1 >= len(route): continue 
                            to_edge = route[idx+1]
                        except ValueError: continue

                        direction = self._classify_direction(current_edge)
                        turn = self._classify_turn(current_edge, to_edge)

                        if direction and turn:
                            movements[direction][turn] += 1
                    except:
                        continue
            except:
                continue

        return movements

    def _classify_direction(self, edge_id: str) -> Optional[str]:
        edge_id = edge_id.lower()
        if 'north' in edge_id or 'n_' in edge_id or '_n' in edge_id: return 'N'
        if 'south' in edge_id or 's_' in edge_id or '_s' in edge_id: return 'S'
        if 'east' in edge_id or 'e_' in edge_id or '_e' in edge_id: return 'E'
        if 'west' in edge_id or 'w_' in edge_id or '_w' in edge_id: return 'W'
        return None

    def _classify_turn(self, from_edge: str, to_edge: str) -> Optional[str]:
        # Fallback to straight if we can't tell
        return 'S'

    def _score_phases(self, movements: Dict) -> Dict[int, float]:
        """Score each SUMO green phase"""
        phase_scores = {}
        for phase in self.green_phases:
            lanes = self._get_lanes_for_phase(phase)
            score = 0.0
            for lane in lanes:
                edge_id = '_'.join(lane.split('_')[:-1]) 
                direction = self._classify_direction(edge_id)
                if not direction: continue
                m = movements.get(direction, {'L':0,'S':0,'R':0})
                score += (1.5 * m['L'] + 1.0 * m['S'] + 0.8 * m['R'])
            phase_scores[phase] = score
        return phase_scores
        
    def decide_phase_change(self, state: Dict) -> bool:
        """New Decision Logic using Omniscient Scores"""
        if self.in_yellow: return False

        movements = self._get_vehicle_movements()
        phase_scores = self._score_phases(movements)
        if not phase_scores: return False

        current_score = phase_scores.get(self.current_phase, 0)
        best_phase = max(phase_scores, key=phase_scores.get)
        best_score = phase_scores[best_phase]

        # Rule 1: Gap-Out
        if current_score <= 0.1 and best_score > 0.5:
            self.next_phase = best_phase
            return True

        # Rule 2: Dominant better phase
        if best_phase != self.current_phase and best_score >= current_score * 1.5 + 2.0:
            self.next_phase = best_phase
            return True
        
        # Rule 3: Max Green Time
        if state['phase_duration'] > self.max_green_time:
             if best_phase != self.current_phase:
                 self.next_phase = best_phase
                 return True
                 
        return False

    def _get_lanes_for_phase(self, phase_index: int) -> List[str]:
        if phase_index >= len(self.phases): return []
        state = self.phases[phase_index].state
        green_lanes = []
        try:
            all_controlled = traci.trafficlight.getControlledLanes(self.tls_id)
            for i, char in enumerate(state):
                if char in ['G', 'g'] and i < len(all_controlled):
                    green_lanes.append(all_controlled[i])
        except:
            pass
        return list(set(green_lanes))

    def _get_next_green_phase(self) -> int:
        if not self.green_phases: return 0
        try:
            current_idx = self.green_phases.index(self.current_phase)
            next_idx = (current_idx + 1) % len(self.green_phases)
            return self.green_phases[next_idx]
        except:
            return self.green_phases[0]

    def _find_yellow_phase(self, current_phase: int, next_phase: int) -> Optional[int]:
        if self.yellow_phases: return self.yellow_phases[0]
        return None
    
    def _trigger_phase_change(self, current_time: float):
        if self.next_phase is None:
            self.next_phase = self._get_next_green_phase()
            
        yellow_phase = self._find_yellow_phase(self.current_phase, self.next_phase)
        
        try:
            if yellow_phase is not None:
                traci.trafficlight.setPhase(self.tls_id, yellow_phase)
                self.in_yellow = True
                self.phase_start_time = current_time
            else:
                self.current_phase = self.next_phase
                traci.trafficlight.setPhase(self.tls_id, self.current_phase)
                self.phase_start_time = current_time
                self.total_switches += 1
        except Exception as e:
            print(f"⚠️ Phase switch failed: {e}")

    def execute_action(self, current_time: float):
        """Execute heuristic control step with safety wrappers"""
        try:
            state = self.get_traffic_state(current_time)
            
            if self.in_yellow:
                if current_time - self.phase_start_time >= self.yellow_time:
                    self.current_phase = self.next_phase if self.next_phase is not None else self.current_phase
                    traci.trafficlight.setPhase(self.tls_id, self.current_phase)
                    self.in_yellow = False
                    self.phase_start_time = current_time
                    self.total_switches += 1
            else:
                if self.decide_phase_change(state):
                    self._trigger_phase_change(current_time)
        except traci.exceptions.FatalTraCIError:
            raise # Let manager handle fatal error
        except Exception as e:
            print(f"⚠️ Agent iteration error: {e}")
    
    def get_metrics(self) -> Dict:
        return {
            'tls_id': self.tls_id,
            'type': 'heuristic',
            'total_switches': self.total_switches,
            'current_phase': self.current_phase
        }
