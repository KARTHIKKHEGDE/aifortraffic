"""
Fixed Timer Controller - Baseline comparison
Simple fixed-time traffic light control (30s green, 3s yellow)
"""

import traci
from typing import Dict

class FixedTimerController:
    """
    Simple fixed-time traffic light controller
    Used as baseline to compare against heuristic agent
    """
    
    def __init__(self, tls_id: str, green_duration: float = 30.0, yellow_duration: float = 3.0):
        """
        Initialize fixed timer controller
        
        Args:
            tls_id: Traffic light ID
            green_duration: How long each green phase lasts
            yellow_duration: Yellow phase duration
        """
        self.tls_id = tls_id
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        
        # Get phase structure
        programs = traci.trafficlight.getAllProgramLogics(self.tls_id)
        if programs:
            self.phases = programs[0].phases
            self.green_phases = []
            self.yellow_phases = []
            
            for i, phase in enumerate(self.phases):
                if 'G' in phase.state or 'g' in phase.state:
                    self.green_phases.append(i)
                elif 'y' in phase.state or 'Y' in phase.state:
                    self.yellow_phases.append(i)
        else:
            self.green_phases = [0, 2]
            self.yellow_phases = [1, 3]
        
        # State
        self.current_phase = self.green_phases[0] if self.green_phases else 0
        self.phase_start_time = 0.0
        self.in_yellow = False
        self.next_phase = None
        self.total_switches = 0
        
        traci.trafficlight.setPhase(self.tls_id, self.current_phase)
        
        print(f"✅ Fixed Timer Controller: {tls_id} (Green: {green_duration}s, Yellow: {yellow_duration}s)")
    
    def step(self, current_time: float):
        """Execute one step of fixed timer logic"""
        phase_duration = current_time - self.phase_start_time
        
        if self.in_yellow:
            # In yellow phase
            if phase_duration >= self.yellow_duration:
                # Switch to next green phase
                self.current_phase = self.next_phase
                traci.trafficlight.setPhase(self.tls_id, self.current_phase)
                self.in_yellow = False
                self.phase_start_time = current_time
                self.total_switches += 1
        else:
            # In green phase
            if phase_duration >= self.green_duration:
                # Start yellow transition
                current_green_idx = self.green_phases.index(self.current_phase)
                next_green_idx = (current_green_idx + 1) % len(self.green_phases)
                self.next_phase = self.green_phases[next_green_idx]
                
                # Find yellow phase
                yellow_phase = self._find_yellow_phase()
                if yellow_phase is not None:
                    traci.trafficlight.setPhase(self.tls_id, yellow_phase)
                    self.in_yellow = True
                    self.phase_start_time = current_time
                else:
                    # Direct switch (no yellow)
                    self.current_phase = self.next_phase
                    traci.trafficlight.setPhase(self.tls_id, self.current_phase)
                    self.phase_start_time = current_time
                    self.total_switches += 1
    
    def _find_yellow_phase(self):
        """Find appropriate yellow phase"""
        for i in range(self.current_phase + 1, min(self.current_phase + 3, len(self.phases))):
            if i in self.yellow_phases:
                return i
        return self.yellow_phases[0] if self.yellow_phases else None
    
    def get_metrics(self) -> Dict:
        """Get controller metrics"""
        return {
            'tls_id': self.tls_id,
            'type': 'fixed_timer',
            'total_switches': self.total_switches,
            'green_duration': self.green_duration,
            'current_phase': self.current_phase
        }
