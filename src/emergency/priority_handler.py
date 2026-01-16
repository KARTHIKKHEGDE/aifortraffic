"""
Emergency Vehicle Priority Handler
Manages detection and signal override for ambulances and emergency vehicles
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from ..utils.logger import setup_logger

logger = setup_logger("emergency_handler")


@dataclass
class EmergencyVehicle:
    """Data class for emergency vehicle tracking"""
    vehicle_id: str
    lane: str
    junction: str
    lane_index: int
    speed: float
    detection_time: float = 0.0
    clearance_time: Optional[float] = None
    is_waiting: bool = False


class EmergencyPriorityHandler:
    """
    Real-time Emergency Vehicle Priority System
    
    Features:
    - Ambulance detection via vehicle type
    - Immediate signal override (green corridor)
    - Green maintenance until vehicle clears
    - Resume RL control after clearance
    - Performance metrics tracking
    
    Priority Logic:
    1. Detect emergency vehicle (ambulance) via TraCI
    2. Identify approaching lane and junction
    3. Override RL decision with emergency action (action=3)
    4. Force green on ambulance lane
    5. Hold green until ambulance clears junction
    6. Resume RL control
    """
    
    # Lane direction mapping (typical 4-way intersection)
    LANE_TO_PHASE = {
        'north': 0,    # NS green
        'south': 0,
        'east': 2,     # EW green  
        'west': 2,
        'n': 0, 's': 0, 'e': 2, 'w': 2,
        0: 0, 1: 0, 2: 2, 3: 2,  # Lane indices
    }
    
    def __init__(
        self,
        max_override_duration: float = 120.0,
        detection_radius: float = 200.0
    ):
        """
        Initialize Emergency Priority Handler
        
        Args:
            max_override_duration: Maximum time to hold green for emergency
            detection_radius: Distance at which to detect emergency vehicles
        """
        self.max_override_duration = max_override_duration
        self.detection_radius = detection_radius
        
        # State tracking
        self.is_active = False
        self.current_emergency: Optional[EmergencyVehicle] = None
        self.override_start_time: Optional[float] = None
        
        # Flags for reward computation
        self.ambulance_waiting = False
        self.ambulance_just_cleared = False
        
        # Metrics
        self.total_emergencies = 0
        self.total_clearance_time = 0.0
        self.clearance_times: List[float] = []
        
        # History
        self.emergency_history: List[EmergencyVehicle] = []
    
    def reset(self) -> None:
        """Reset handler for new episode"""
        self.is_active = False
        self.current_emergency = None
        self.override_start_time = None
        self.ambulance_waiting = False
        self.ambulance_just_cleared = False
    
    def detect_emergency(self, emergency_info: Dict[str, Any]) -> bool:
        """
        Process detected emergency vehicle
        
        Args:
            emergency_info: Dictionary with emergency vehicle information
                - id: Vehicle ID
                - lane: Current lane
                - junction: Approaching junction
                - lane_index: Lane index at junction
                - speed: Current speed
                - is_waiting: Whether vehicle is stopped
        
        Returns:
            True if new emergency detected
        """
        if self.is_active:
            # Already handling an emergency
            if emergency_info['id'] == self.current_emergency.vehicle_id:
                # Update existing emergency
                self.current_emergency.speed = emergency_info['speed']
                self.current_emergency.is_waiting = emergency_info.get('is_waiting', False)
                self.ambulance_waiting = self.current_emergency.is_waiting
                return False
            else:
                # New emergency while handling another - queue or prioritize?
                # For now, focus on current one
                return False
        
        # New emergency detection
        self.is_active = True
        self.current_emergency = EmergencyVehicle(
            vehicle_id=emergency_info['id'],
            lane=emergency_info['lane'],
            junction=emergency_info.get('junction', ''),
            lane_index=emergency_info.get('lane_index', 0),
            speed=emergency_info['speed'],
            is_waiting=emergency_info.get('is_waiting', False),
        )
        
        self.ambulance_waiting = self.current_emergency.is_waiting
        self.ambulance_just_cleared = False
        self.total_emergencies += 1
        
        logger.warning(
            f"🚑 EMERGENCY DETECTED | "
            f"Vehicle: {self.current_emergency.vehicle_id} | "
            f"Junction: {self.current_emergency.junction} | "
            f"Lane: {self.current_emergency.lane}"
        )
        
        return True
    
    def check_clearance(self, current_vehicles: List[str], current_time: float) -> bool:
        """
        Check if emergency vehicle has cleared the junction
        
        Args:
            current_vehicles: List of current vehicle IDs in simulation
            current_time: Current simulation time
        
        Returns:
            True if emergency was just cleared
        """
        if not self.is_active or self.current_emergency is None:
            return False
        
        # Check if ambulance is still in simulation
        if self.current_emergency.vehicle_id not in current_vehicles:
            # Ambulance has left the simulation (cleared or arrived)
            self._handle_clearance(current_time)
            return True
        
        # Check for timeout
        if self.override_start_time:
            elapsed = current_time - self.override_start_time
            if elapsed > self.max_override_duration:
                logger.warning(
                    f"Emergency override timeout for {self.current_emergency.vehicle_id}"
                )
                self._handle_clearance(current_time)
                return True
        
        return False
    
    def _handle_clearance(self, current_time: float) -> None:
        """Handle emergency vehicle clearance"""
        if self.current_emergency is None:
            return
        
        self.current_emergency.clearance_time = current_time
        
        if self.override_start_time:
            clearance_duration = current_time - self.override_start_time
            self.clearance_times.append(clearance_duration)
            self.total_clearance_time += clearance_duration
            
            logger.info(
                f"🚑 EMERGENCY CLEARED | "
                f"Vehicle: {self.current_emergency.vehicle_id} | "
                f"Clearance Time: {clearance_duration:.1f}s"
            )
        
        # Save to history
        self.emergency_history.append(self.current_emergency)
        
        # Reset state
        self.ambulance_just_cleared = True
        self.is_active = False
        self.current_emergency = None
        self.override_start_time = None
        self.ambulance_waiting = False
    
    def override_actions(
        self,
        actions: Dict[str, int],
        emergency_info: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Override RL actions with emergency priority
        
        Args:
            actions: Original RL action decisions
            emergency_info: Emergency vehicle information
        
        Returns:
            Modified actions with emergency override
        """
        if not self.is_active or self.current_emergency is None:
            return actions
        
        # Get affected junction
        affected_junction = self.current_emergency.junction
        
        if affected_junction in actions:
            # Override with emergency action
            actions[affected_junction] = 3  # Emergency override action
            
            # Set override start time if not set
            if self.override_start_time is None:
                # Note: We don't have access to simulation time here
                # This will be set properly when action is applied
                pass
        
        return actions
    
    def get_emergency_phase(self, junction_id: str) -> int:
        """
        Determine which signal phase gives green to emergency vehicle
        
        Args:
            junction_id: Junction ID
        
        Returns:
            Phase index for green on emergency lane
        """
        if self.current_emergency is None:
            return 0
        
        if self.current_emergency.junction != junction_id:
            return 0
        
        # Determine direction from lane
        lane = self.current_emergency.lane
        lane_index = self.current_emergency.lane_index
        
        # Try to infer direction from lane name or index
        direction = self._infer_lane_direction(lane, lane_index)
        
        return self.LANE_TO_PHASE.get(direction, 0)
    
    def _infer_lane_direction(self, lane: str, lane_index: int) -> str:
        """Infer lane direction from lane ID or index"""
        lane_lower = lane.lower()
        
        # Check for direction keywords in lane name
        for direction in ['north', 'south', 'east', 'west', '_n_', '_s_', '_e_', '_w_']:
            if direction in lane_lower:
                return direction[0] if len(direction) > 1 else direction
        
        # Fall back to lane index (0-1: NS, 2-3: EW for typical 4-way)
        if lane_index in [0, 1]:
            return 'north'
        else:
            return 'east'
    
    def get_clearance_time(self) -> Optional[float]:
        """Get clearance time of most recent emergency"""
        if self.clearance_times:
            return self.clearance_times[-1]
        return None
    
    def get_average_clearance_time(self) -> float:
        """Get average clearance time across all emergencies"""
        if not self.clearance_times:
            return 0.0
        return sum(self.clearance_times) / len(self.clearance_times)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get emergency handling metrics"""
        return {
            'total_emergencies': self.total_emergencies,
            'total_clearance_time': self.total_clearance_time,
            'avg_clearance_time': self.get_average_clearance_time(),
            'clearance_times': self.clearance_times.copy(),
            'is_active': self.is_active,
        }
    
    def set_override_start_time(self, time: float) -> None:
        """Set the start time for current override"""
        if self.is_active and self.override_start_time is None:
            self.override_start_time = time


class EmergencyInjector:
    """
    Emergency Vehicle Injection System
    Generates emergency vehicle events for simulation
    """
    
    def __init__(
        self,
        probability: float = 0.01,
        min_interval: float = 300.0,  # 5 minutes minimum between emergencies
        seed: int = 42
    ):
        """
        Initialize emergency injector
        
        Args:
            probability: Probability of emergency per step
            min_interval: Minimum time between emergencies
            seed: Random seed
        """
        self.probability = probability
        self.min_interval = min_interval
        self.seed = seed
        
        import random
        self.rng = random.Random(seed)
        
        self.last_injection_time = -min_interval
        self.injection_count = 0
    
    def should_inject(self, current_time: float) -> bool:
        """
        Determine if an emergency should be injected
        
        Args:
            current_time: Current simulation time
        
        Returns:
            True if emergency should be injected
        """
        # Check minimum interval
        if current_time - self.last_injection_time < self.min_interval:
            return False
        
        # Probability check
        if self.rng.random() < self.probability:
            self.last_injection_time = current_time
            self.injection_count += 1
            return True
        
        return False
    
    def get_injection_route(
        self,
        available_edges: List[str],
        junction_edges: Dict[str, List[str]]
    ) -> tuple:
        """
        Generate origin-destination for emergency vehicle
        
        Args:
            available_edges: List of available edges
            junction_edges: Mapping of junction to approaching edges
        
        Returns:
            Tuple of (origin_edge, destination_edge)
        """
        if not available_edges:
            return None, None
        
        # Select random origin (simulating hospital/emergency service)
        origin = self.rng.choice(available_edges)
        
        # Select destination near one of the controlled junctions
        # This ensures the ambulance passes through our controlled area
        if junction_edges:
            junction = self.rng.choice(list(junction_edges.keys()))
            if junction_edges[junction]:
                destination = self.rng.choice(junction_edges[junction])
            else:
                destination = self.rng.choice(available_edges)
        else:
            destination = self.rng.choice(available_edges)
        
        # Avoid same edge
        while destination == origin and len(available_edges) > 1:
            destination = self.rng.choice(available_edges)
        
        return origin, destination


if __name__ == "__main__":
    # Test emergency handler
    print("Testing Emergency Priority Handler...")
    print("=" * 50)
    
    handler = EmergencyPriorityHandler()
    
    # Simulate emergency detection
    emergency_info = {
        'id': 'ambulance_001',
        'lane': 'silk_board_north_0',
        'junction': 'silk_board',
        'lane_index': 0,
        'speed': 5.0,
        'is_waiting': False,
    }
    
    print("\n1. Detecting emergency...")
    detected = handler.detect_emergency(emergency_info)
    print(f"   Emergency detected: {detected}")
    print(f"   Is active: {handler.is_active}")
    
    # Test override
    print("\n2. Testing action override...")
    original_actions = {'silk_board': 1, 'tin_factory': 2}
    modified_actions = handler.override_actions(original_actions, emergency_info)
    print(f"   Original actions: {original_actions}")
    print(f"   Modified actions: {modified_actions}")
    
    # Test emergency phase
    print("\n3. Getting emergency phase...")
    phase = handler.get_emergency_phase('silk_board')
    print(f"   Emergency phase for silk_board: {phase}")
    
    # Test clearance
    print("\n4. Simulating clearance...")
    handler.set_override_start_time(0.0)
    cleared = handler.check_clearance(['car_001', 'car_002'], 30.0)  # Ambulance not in list
    print(f"   Ambulance cleared: {cleared}")
    print(f"   Just cleared flag: {handler.ambulance_just_cleared}")
    
    # Get metrics
    print("\n5. Getting metrics...")
    metrics = handler.get_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 50)
    print("Emergency Handler test complete!")
