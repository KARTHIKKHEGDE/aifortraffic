"""
SUMO Connector Module
Handles all TraCI communication with SUMO simulator
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

from ..utils.sumo_utils import (
    get_sumo_binary,
    get_sumo_home,
    add_sumo_tools_to_path,
    import_traci,
    create_sumo_config_file
)
from ..utils.logger import setup_logger

# Import TraCI
add_sumo_tools_to_path()
try:
    import traci
    import traci.constants as tc
except ImportError:
    traci = None
    tc = None

logger = setup_logger("sumo_connector")


class SUMOConnector:
    """
    SUMO TraCI connector for traffic simulation
    
    Handles:
    - SUMO process management
    - Traffic light control
    - Vehicle/lane data retrieval
    - Simulation stepping
    """
    
    def __init__(
        self,
        net_file: str,
        route_file: str,
        gui: bool = False,
        step_length: float = 1.0,
        begin: int = 0,
        end: int = 3600,
        seed: int = 42,
        additional_files: List[str] = None,
        port: int = None
    ):
        """
        Initialize SUMO connector
        
        Args:
            net_file: Path to SUMO network file (.net.xml)
            route_file: Path to route/trip file (.rou.xml or .trips.xml)
            gui: Whether to use SUMO-GUI
            step_length: Simulation step length in seconds
            begin: Simulation start time
            end: Simulation end time
            seed: Random seed
            additional_files: Additional files to load
            port: TraCI port (auto-assigned if None)
        """
        self.net_file = net_file
        self.route_file = route_file
        self.gui = gui
        self.step_length = step_length
        self.begin = begin
        self.end = end
        self.seed = seed
        self.additional_files = additional_files or []
        self.port = port
        
        # State tracking
        self.is_running = False
        self.simulation_step = 0
        
        # Cached data
        self._controlled_tls: List[str] = []
        self._lanes_per_tls: Dict[str, List[str]] = {}
        
        # Build SUMO command
        self.sumo_cmd = self._build_command()
    
    def _build_command(self) -> List[str]:
        """Build SUMO command line arguments"""
        sumo_binary = get_sumo_binary(gui=self.gui)
        
        cmd = [
            sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--step-length", str(self.step_length),
            "-b", str(self.begin),
            "-e", str(self.end),
            "--seed", str(self.seed),
            "--time-to-teleport", "-1",  # Disable teleporting
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--waiting-time-memory", "1000",
        ]
        
        # Add additional files
        if self.additional_files:
            cmd.extend(["--additional-files", ",".join(self.additional_files)])
        
        # GUI-specific options
        if self.gui:
            cmd.extend([
                "--start", "true",  # Auto-start simulation
                "--quit-on-end", "true",
            ])
        
        return cmd
    
    def start(self) -> None:
        """Start SUMO simulation"""
        if traci is None:
            raise ImportError("TraCI not available. Check SUMO installation.")
        
        if self.is_running:
            logger.warning("SUMO already running, closing existing instance")
            self.close()
        
        logger.info(f"Starting SUMO: {' '.join(self.sumo_cmd)}")
        
        try:
            if self.port:
                traci.start(self.sumo_cmd, port=self.port)
            else:
                traci.start(self.sumo_cmd)
            
            self.is_running = True
            self.simulation_step = 0
            
            # Cache traffic light and lane information
            self._cache_network_info()
            
            logger.info("SUMO started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start SUMO: {e}")
            raise
    
    def close(self) -> None:
        """Close SUMO simulation"""
        if self.is_running:
            try:
                traci.close()
            except Exception:
                pass
            self.is_running = False
            self.simulation_step = 0
            logger.info("SUMO closed")
    
    def step(self) -> None:
        """Advance simulation by one step"""
        if not self.is_running:
            raise RuntimeError("SUMO not running")
        
        traci.simulationStep()
        self.simulation_step += 1
    
    def get_simulation_time(self) -> float:
        """Get current simulation time in seconds"""
        return traci.simulation.getTime()
    
    def is_simulation_ended(self) -> bool:
        """Check if simulation has ended"""
        return traci.simulation.getMinExpectedNumber() <= 0
    
    def _cache_network_info(self) -> None:
        """Cache network topology information"""
        # Get all traffic lights
        self._controlled_tls = list(traci.trafficlight.getIDList())
        logger.info(f"Found {len(self._controlled_tls)} traffic lights")
        
        # Get lanes controlled by each TLS
        for tls_id in self._controlled_tls:
            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
            # Remove duplicates while preserving order
            unique_lanes = list(dict.fromkeys(controlled_lanes))
            self._lanes_per_tls[tls_id] = unique_lanes
    
    # =========================================================================
    # TRAFFIC LIGHT CONTROL
    # =========================================================================
    
    def get_tls_ids(self) -> List[str]:
        """Get list of all traffic light IDs"""
        return self._controlled_tls.copy()
    
    def get_tls_lanes(self, tls_id: str) -> List[str]:
        """Get lanes controlled by a traffic light"""
        return self._lanes_per_tls.get(tls_id, [])
    
    def get_tls_phase(self, tls_id: str) -> int:
        """Get current phase of a traffic light"""
        return traci.trafficlight.getPhase(tls_id)
    
    def get_tls_phase_duration(self, tls_id: str) -> float:
        """Get duration of current phase"""
        return traci.trafficlight.getPhaseDuration(tls_id)
    
    def get_tls_state(self, tls_id: str) -> str:
        """Get current state string of traffic light"""
        return traci.trafficlight.getRedYellowGreenState(tls_id)
    
    def get_tls_time_since_switch(self, tls_id: str) -> float:
        """Get time since last phase switch"""
        return traci.trafficlight.getNextSwitch(tls_id) - self.get_simulation_time()
    
    def set_tls_phase(self, tls_id: str, phase: int) -> None:
        """Set traffic light to a specific phase"""
        traci.trafficlight.setPhase(tls_id, phase)
    
    def set_tls_phase_duration(self, tls_id: str, duration: float) -> None:
        """Set duration of current phase"""
        traci.trafficlight.setPhaseDuration(tls_id, duration)
    
    def set_tls_state(self, tls_id: str, state: str) -> None:
        """Set traffic light state directly"""
        traci.trafficlight.setRedYellowGreenState(tls_id, state)
    
    def get_num_phases(self, tls_id: str) -> int:
        """Get number of phases for a traffic light"""
        program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        return len(program.phases)
    
    # =========================================================================
    # LANE DATA RETRIEVAL
    # =========================================================================
    
    def get_lane_queue_length(self, lane_id: str) -> int:
        """Get number of halted vehicles in a lane"""
        return traci.lane.getLastStepHaltingNumber(lane_id)
    
    def get_lane_vehicle_count(self, lane_id: str) -> int:
        """Get number of vehicles in a lane"""
        return traci.lane.getLastStepVehicleNumber(lane_id)
    
    def get_lane_density(self, lane_id: str) -> float:
        """Get vehicle density in a lane (vehicles per meter)"""
        length = traci.lane.getLength(lane_id)
        count = traci.lane.getLastStepVehicleNumber(lane_id)
        return count / length if length > 0 else 0.0
    
    def get_lane_mean_speed(self, lane_id: str) -> float:
        """Get mean speed of vehicles in a lane"""
        return traci.lane.getLastStepMeanSpeed(lane_id)
    
    def get_lane_waiting_time(self, lane_id: str) -> float:
        """Get total waiting time in a lane"""
        return traci.lane.getWaitingTime(lane_id)
    
    def get_lane_length(self, lane_id: str) -> float:
        """Get length of a lane"""
        return traci.lane.getLength(lane_id)
    
    def get_lane_vehicles(self, lane_id: str) -> List[str]:
        """Get list of vehicle IDs in a lane"""
        return list(traci.lane.getLastStepVehicleIDs(lane_id))
    
    # =========================================================================
    # VEHICLE DATA RETRIEVAL
    # =========================================================================
    
    def get_vehicle_ids(self) -> List[str]:
        """Get all vehicle IDs in simulation"""
        return list(traci.vehicle.getIDList())
    
    def get_vehicle_count(self) -> int:
        """Get total number of vehicles"""
        return traci.vehicle.getIDCount()
    
    def get_vehicle_type(self, veh_id: str) -> str:
        """Get type of a vehicle"""
        return traci.vehicle.getTypeID(veh_id)
    
    def get_vehicle_speed(self, veh_id: str) -> float:
        """Get speed of a vehicle"""
        return traci.vehicle.getSpeed(veh_id)
    
    def get_vehicle_lane(self, veh_id: str) -> str:
        """Get lane of a vehicle"""
        return traci.vehicle.getLaneID(veh_id)
    
    def get_vehicle_waiting_time(self, veh_id: str) -> float:
        """Get waiting time of a vehicle"""
        return traci.vehicle.getWaitingTime(veh_id)
    
    def get_vehicle_accumulated_waiting_time(self, veh_id: str) -> float:
        """Get accumulated waiting time of a vehicle"""
        return traci.vehicle.getAccumulatedWaitingTime(veh_id)
    
    def get_vehicle_co2_emission(self, veh_id: str) -> float:
        """Get CO2 emission of a vehicle (mg/s)"""
        return traci.vehicle.getCO2Emission(veh_id)
    
    def is_vehicle_emergency(self, veh_id: str) -> bool:
        """Check if vehicle is an emergency vehicle"""
        veh_type = self.get_vehicle_type(veh_id)
        return veh_type == 'ambulance' or 'emergency' in veh_type.lower()
    
    def set_vehicle_speed(self, veh_id: str, speed: float) -> None:
        """Set vehicle speed"""
        traci.vehicle.setSpeed(veh_id, speed)
    
    def set_vehicle_max_speed(self, veh_id: str, max_speed: float) -> None:
        """Set vehicle maximum speed"""
        traci.vehicle.setMaxSpeed(veh_id, max_speed)
    
    # =========================================================================
    # EMERGENCY VEHICLE DETECTION
    # =========================================================================
    
    def get_emergency_vehicles(self) -> List[Dict[str, Any]]:
        """
        Get all emergency vehicles and their locations
        
        Returns:
            List of dicts with vehicle info
        """
        emergency_vehicles = []
        
        for veh_id in self.get_vehicle_ids():
            if self.is_vehicle_emergency(veh_id):
                lane = self.get_vehicle_lane(veh_id)
                speed = self.get_vehicle_speed(veh_id)
                
                emergency_vehicles.append({
                    'id': veh_id,
                    'lane': lane,
                    'speed': speed,
                    'is_waiting': speed < 0.5
                })
        
        return emergency_vehicles
    
    def get_emergency_at_junction(self, tls_id: str) -> Optional[Dict[str, Any]]:
        """
        Check if emergency vehicle is approaching a junction
        
        Args:
            tls_id: Traffic light ID
        
        Returns:
            Emergency vehicle info or None
        """
        controlled_lanes = self.get_tls_lanes(tls_id)
        emergency_vehicles = self.get_emergency_vehicles()
        
        for ev in emergency_vehicles:
            # Check if emergency vehicle is in controlled lanes
            if ev['lane'] in controlled_lanes:
                ev['junction'] = tls_id
                ev['lane_index'] = controlled_lanes.index(ev['lane'])
                return ev
        
        return None
    
    # =========================================================================
    # SIMULATION METRICS
    # =========================================================================
    
    def get_departed_count(self) -> int:
        """Get number of departed vehicles this step"""
        return traci.simulation.getDepartedNumber()
    
    def get_arrived_count(self) -> int:
        """Get number of arrived vehicles this step"""
        return traci.simulation.getArrivedNumber()
    
    def get_total_waiting_time(self) -> float:
        """Get total waiting time of all vehicles"""
        total = 0.0
        for veh_id in self.get_vehicle_ids():
            total += self.get_vehicle_waiting_time(veh_id)
        return total
    
    def get_average_speed(self) -> float:
        """Get average speed of all vehicles"""
        speeds = [self.get_vehicle_speed(v) for v in self.get_vehicle_ids()]
        return sum(speeds) / len(speeds) if speeds else 0.0
    
    def get_total_co2_emission(self) -> float:
        """Get total CO2 emission from all vehicles"""
        total = 0.0
        for veh_id in self.get_vehicle_ids():
            total += self.get_vehicle_co2_emission(veh_id)
        return total
    
    def get_junction_metrics(self, tls_id: str) -> Dict[str, Any]:
        """
        Get comprehensive metrics for a junction
        
        Args:
            tls_id: Traffic light ID
        
        Returns:
            Dictionary of metrics
        """
        lanes = self.get_tls_lanes(tls_id)
        
        total_queue = 0
        total_waiting = 0.0
        total_density = 0.0
        total_speed = 0.0
        vehicle_count = 0
        
        for lane in lanes:
            total_queue += self.get_lane_queue_length(lane)
            total_waiting += self.get_lane_waiting_time(lane)
            total_density += self.get_lane_density(lane)
            total_speed += self.get_lane_mean_speed(lane)
            vehicle_count += self.get_lane_vehicle_count(lane)
        
        num_lanes = len(lanes)
        
        return {
            'queue_length': total_queue,
            'total_waiting_time': total_waiting,
            'avg_density': total_density / num_lanes if num_lanes > 0 else 0,
            'avg_speed': total_speed / num_lanes if num_lanes > 0 else 0,
            'vehicle_count': vehicle_count,
            'current_phase': self.get_tls_phase(tls_id),
        }
    
    # =========================================================================
    # WEATHER EFFECTS (Applied to vehicles)
    # =========================================================================
    
    def apply_rain_effects(self, speed_factor: float = 0.75) -> None:
        """
        Apply rain effects to all vehicles
        
        Args:
            speed_factor: Speed reduction factor (0.75 = 25% slower)
        """
        for veh_id in self.get_vehicle_ids():
            # Reduce max speed
            current_max = traci.vehicle.getMaxSpeed(veh_id)
            traci.vehicle.setMaxSpeed(veh_id, current_max * speed_factor)
            
            # Increase headway (time gap)
            traci.vehicle.setTau(veh_id, 2.5)  # Default is ~1.0
            
            # Reduce acceleration
            traci.vehicle.setAccel(veh_id, 1.5)
    
    def remove_rain_effects(self) -> None:
        """Remove rain effects from all vehicles"""
        for veh_id in self.get_vehicle_ids():
            veh_type = self.get_vehicle_type(veh_id)
            
            # Reset to default values (approximate)
            traci.vehicle.setTau(veh_id, 1.0)
            traci.vehicle.setAccel(veh_id, 2.0)


if __name__ == "__main__":
    # Test SUMO connector (requires actual network and route files)
    print("SUMO Connector module loaded successfully")
    print("Run with actual SUMO files to test functionality")
