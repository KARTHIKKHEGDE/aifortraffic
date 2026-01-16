"""
Traffic Route Generation Script
Generates realistic traffic demand for Bangalore intersections
"""

import os
import sys
import subprocess
import random
from pathlib import Path
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, get_project_root
from src.utils.logger import setup_logger
from src.utils.sumo_utils import get_sumo_home, get_random_trips_script

logger = setup_logger("route_generator")


class TrafficRouteGenerator:
    """
    Generate realistic traffic demand for SUMO simulation
    Supports multiple queue configurations and traffic scenarios
    """
    
    def __init__(self, maps_dir: str = None, routes_dir: str = None):
        """
        Initialize route generator
        
        Args:
            maps_dir: Directory containing SUMO network files
            routes_dir: Directory for output route files
        """
        project_root = get_project_root()
        
        self.maps_dir = Path(maps_dir) if maps_dir else project_root / "maps"
        self.routes_dir = Path(routes_dir) if routes_dir else project_root / "routes"
        
        # Create route subdirectories
        (self.routes_dir / "baseline").mkdir(parents=True, exist_ok=True)
        (self.routes_dir / "realistic").mkdir(parents=True, exist_ok=True)
        (self.routes_dir / "emergency").mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        try:
            self.env_config = load_config('env_config')
            self.queue_config = self.env_config.get('queue', {})
        except FileNotFoundError:
            logger.warning("Config not found, using defaults")
            self.queue_config = self._get_default_queue_config()
        
        # Vehicle type probabilities
        self.vehicle_types = self._get_vehicle_types()
    
    def _get_default_queue_config(self) -> dict:
        """Default queue configuration"""
        return {
            'mode': 'realistic_bangalore',
            'modes': {
                'baseline': {
                    'vehicle_period': 3.0,
                    'peak_factor': 1.0
                },
                'realistic_bangalore': {
                    'vehicle_period': 1.2,
                    'peak_factor': 2.5
                }
            }
        }
    
    def _get_vehicle_types(self) -> dict:
        """Get vehicle type definitions"""
        return {
            'car': {
                'probability': 0.60,
                'vClass': 'passenger',
                'accel': 2.0,
                'decel': 4.5,
                'sigma': 0.5,
                'length': 4.5,
                'maxSpeed': 16.67,
                'speedFactor': 0.9,
                'color': '1,1,0'
            },
            'bus': {
                'probability': 0.10,
                'vClass': 'bus',
                'accel': 1.2,
                'decel': 3.5,
                'sigma': 0.3,
                'length': 12.0,
                'maxSpeed': 13.89,
                'speedFactor': 0.7,
                'color': '0,1,0'
            },
            'bike': {
                'probability': 0.20,
                'vClass': 'motorcycle',
                'accel': 2.5,
                'decel': 5.0,
                'sigma': 0.8,
                'length': 2.0,
                'maxSpeed': 11.11,
                'speedFactor': 1.1,
                'width': 0.8,
                'color': '0,0,1'
            },
            'auto_rickshaw': {
                'probability': 0.10,
                'vClass': 'passenger',
                'accel': 1.8,
                'decel': 4.0,
                'sigma': 0.6,
                'length': 3.0,
                'maxSpeed': 13.89,
                'speedFactor': 0.85,
                'width': 1.5,
                'color': '1,1,0'
            },
            'ambulance': {
                'probability': 0.0,  # Injected separately
                'vClass': 'emergency',
                'accel': 2.8,
                'decel': 4.5,
                'sigma': 0.2,
                'length': 6.0,
                'maxSpeed': 25.0,
                'speedFactor': 1.3,
                'color': '1,0,0',
                'emergencyDecel': 5.0,
                'lcStrategic': 100,
                'lcSpeedGain': 100
            }
        }
    
    def create_vehicle_types_file(self, output_file: str) -> str:
        """
        Create vehicle types definition file
        
        Args:
            output_file: Path for output file
        
        Returns:
            Path to created file
        """
        root = ET.Element('additional')
        root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        
        for vtype_id, vtype_props in self.vehicle_types.items():
            vtype_elem = ET.SubElement(root, 'vType')
            vtype_elem.set('id', vtype_id)
            
            for prop, value in vtype_props.items():
                if prop != 'probability':
                    vtype_elem.set(prop, str(value))
        
        # Write file with pretty formatting
        self._write_xml_pretty(root, output_file)
        logger.info(f"Created vehicle types file: {output_file}")
        
        return output_file
    
    def _write_xml_pretty(self, root: ET.Element, output_file: str):
        """Write XML with pretty formatting"""
        rough_string = ET.tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(reparsed.toprettyxml(indent="    "))
    
    def generate_random_trips(
        self,
        net_file: str,
        output_file: str,
        end_time: int = 3600,
        period: float = 2.0,
        fringe_factor: float = 5.0,
        seed: int = 42
    ) -> bool:
        """
        Generate random trips using SUMO's randomTrips.py
        
        Args:
            net_file: Path to SUMO network file
            output_file: Path for output trips file
            end_time: Simulation end time in seconds
            period: Average time between vehicle insertions
            fringe_factor: Factor for fringe network generation
            seed: Random seed for reproducibility
        
        Returns:
            True if successful
        """
        try:
            random_trips_script = get_random_trips_script()
        except Exception as e:
            logger.warning(f"randomTrips.py not found: {e}")
            return self._generate_trips_manually(net_file, output_file, end_time, period, seed)
        
        cmd = [
            sys.executable,
            random_trips_script,
            "-n", net_file,
            "-o", output_file,
            "-e", str(end_time),
            "--period", str(period),
            "--fringe-factor", str(fringe_factor),
            "--min-distance", "500",
            "--validate",
            "--seed", str(seed),
            "--trip-attributes", 'departLane="best" departSpeed="max"',
        ]
        
        logger.info(f"Generating random trips: period={period}s, duration={end_time}s")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if os.path.exists(output_file):
                # Count generated trips
                tree = ET.parse(output_file)
                num_trips = len(tree.findall('.//trip'))
                logger.info(f"Generated {num_trips} trips to {output_file}")
                return True
            
            return False
            
        except subprocess.CalledProcessError as e:
            logger.error(f"randomTrips.py failed: {e.stderr}")
            return self._generate_trips_manually(net_file, output_file, end_time, period, seed)
    
    def _generate_trips_manually(
        self,
        net_file: str,
        output_file: str,
        end_time: int,
        period: float,
        seed: int
    ) -> bool:
        """
        Generate trips manually when randomTrips.py is not available
        """
        random.seed(seed)
        
        try:
            # Parse network to get edges
            tree = ET.parse(net_file)
            root = tree.getroot()
            
            # Get all valid edges (non-internal)
            edges = []
            for edge in root.findall('.//edge'):
                edge_id = edge.get('id', '')
                if not edge_id.startswith(':'):  # Skip internal edges
                    function = edge.get('function', '')
                    if function != 'internal':
                        edges.append(edge_id)
            
            if len(edges) < 2:
                logger.error("Not enough edges in network")
                return False
            
            logger.info(f"Found {len(edges)} edges in network")
            
            # Create routes element
            routes_root = ET.Element('routes')
            routes_root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
            
            # Add vehicle types
            for vtype_id, vtype_props in self.vehicle_types.items():
                if vtype_props['probability'] > 0:
                    vtype_elem = ET.SubElement(routes_root, 'vType')
                    vtype_elem.set('id', vtype_id)
                    for prop, value in vtype_props.items():
                        if prop != 'probability':
                            vtype_elem.set(prop, str(value))
            
            # Generate trips
            trip_id = 0
            current_time = 0.0
            
            while current_time < end_time:
                # Select vehicle type based on probability
                vtype = self._select_vehicle_type()
                
                # Select random origin and destination
                from_edge = random.choice(edges)
                to_edge = random.choice(edges)
                
                # Avoid same edge
                while to_edge == from_edge and len(edges) > 1:
                    to_edge = random.choice(edges)
                
                # Create trip element
                trip_elem = ET.SubElement(routes_root, 'trip')
                trip_elem.set('id', f"trip_{trip_id}")
                trip_elem.set('depart', f"{current_time:.2f}")
                trip_elem.set('from', from_edge)
                trip_elem.set('to', to_edge)
                trip_elem.set('type', vtype)
                trip_elem.set('departLane', 'best')
                trip_elem.set('departSpeed', 'max')
                
                trip_id += 1
                current_time += random.expovariate(1.0 / period)
            
            # Write file
            self._write_xml_pretty(routes_root, output_file)
            logger.info(f"Manually generated {trip_id} trips to {output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Manual trip generation failed: {e}")
            return False
    
    def _select_vehicle_type(self) -> str:
        """Select vehicle type based on probability distribution"""
        r = random.random()
        cumulative = 0.0
        
        for vtype_id, vtype_props in self.vehicle_types.items():
            cumulative += vtype_props.get('probability', 0)
            if r <= cumulative:
                return vtype_id
        
        return 'car'  # Default
    
    def generate_baseline_routes(self, net_file: str) -> dict:
        """
        Generate baseline queue routes (SUMO defaults)
        
        Args:
            net_file: Path to SUMO network file
        
        Returns:
            Dictionary with generated file paths
        """
        baseline_config = self.queue_config.get('modes', {}).get('baseline', {})
        period = baseline_config.get('vehicle_period', 3.0)
        
        output_dir = self.routes_dir / "baseline"
        results = {}
        
        # Low demand
        low_file = output_dir / "low_demand.trips.xml"
        if self.generate_random_trips(net_file, str(low_file), period=period * 2):
            results['low'] = str(low_file)
        
        # Medium demand (baseline)
        medium_file = output_dir / "medium_demand.trips.xml"
        if self.generate_random_trips(net_file, str(medium_file), period=period):
            results['medium'] = str(medium_file)
        
        # High demand
        high_file = output_dir / "high_demand.trips.xml"
        if self.generate_random_trips(net_file, str(high_file), period=period * 0.5):
            results['high'] = str(high_file)
        
        return results
    
    def generate_realistic_routes(self, net_file: str) -> dict:
        """
        Generate realistic Bangalore traffic patterns
        
        Args:
            net_file: Path to SUMO network file
        
        Returns:
            Dictionary with generated file paths
        """
        realistic_config = self.queue_config.get('modes', {}).get('realistic_bangalore', {})
        base_period = realistic_config.get('vehicle_period', 1.2)
        peak_factor = realistic_config.get('peak_factor', 2.5)
        
        output_dir = self.routes_dir / "realistic"
        results = {}
        
        # Morning peak (8-10 AM simulation)
        # Higher demand, adjusted for 3600s simulation
        morning_file = output_dir / "morning_peak.trips.xml"
        if self.generate_random_trips(
            net_file, str(morning_file),
            period=base_period / peak_factor,
            fringe_factor=10.0
        ):
            results['morning_peak'] = str(morning_file)
        
        # Evening peak (6-8 PM simulation)
        evening_file = output_dir / "evening_peak.trips.xml"
        if self.generate_random_trips(
            net_file, str(evening_file),
            period=base_period / (peak_factor * 1.1),  # Evening is busier
            fringe_factor=10.0
        ):
            results['evening_peak'] = str(evening_file)
        
        # Off-peak
        offpeak_file = output_dir / "off_peak.trips.xml"
        if self.generate_random_trips(
            net_file, str(offpeak_file),
            period=base_period * 2.0,
            fringe_factor=5.0
        ):
            results['off_peak'] = str(offpeak_file)
        
        return results
    
    def generate_emergency_scenarios(self, net_file: str) -> dict:
        """
        Generate routes with emergency vehicles
        
        Args:
            net_file: Path to SUMO network file
        
        Returns:
            Dictionary with generated file paths
        """
        output_dir = self.routes_dir / "emergency"
        results = {}
        
        # Parse network to get edges
        try:
            tree = ET.parse(net_file)
            root = tree.getroot()
            
            edges = []
            for edge in root.findall('.//edge'):
                edge_id = edge.get('id', '')
                if not edge_id.startswith(':'):
                    edges.append(edge_id)
            
            if len(edges) < 2:
                logger.error("Not enough edges for emergency scenarios")
                return results
            
        except Exception as e:
            logger.error(f"Failed to parse network: {e}")
            return results
        
        # Generate scenario with ambulance
        scenario_file = output_dir / "ambulance_scenario.rou.xml"
        
        routes_root = ET.Element('routes')
        
        # Add ambulance vehicle type
        ambulance_type = self.vehicle_types['ambulance']
        vtype_elem = ET.SubElement(routes_root, 'vType')
        vtype_elem.set('id', 'ambulance')
        for prop, value in ambulance_type.items():
            if prop != 'probability':
                vtype_elem.set(prop, str(value))
        
        # Add normal vehicle type
        car_type = self.vehicle_types['car']
        car_elem = ET.SubElement(routes_root, 'vType')
        car_elem.set('id', 'car')
        for prop, value in car_type.items():
            if prop != 'probability':
                car_elem.set(prop, str(value))
        
        # Generate background traffic
        trip_id = 0
        current_time = 0.0
        
        while current_time < 3600:
            from_edge = random.choice(edges)
            to_edge = random.choice(edges)
            while to_edge == from_edge:
                to_edge = random.choice(edges)
            
            trip_elem = ET.SubElement(routes_root, 'trip')
            trip_elem.set('id', f"car_{trip_id}")
            trip_elem.set('depart', f"{current_time:.2f}")
            trip_elem.set('from', from_edge)
            trip_elem.set('to', to_edge)
            trip_elem.set('type', 'car')
            
            trip_id += 1
            current_time += random.expovariate(0.5)  # 2 vehicles/second average
        
        # Add ambulance events at strategic times
        ambulance_times = [300, 900, 1500, 2100, 2700, 3300]  # Every 10 minutes
        
        for i, depart_time in enumerate(ambulance_times):
            from_edge = random.choice(edges)
            to_edge = random.choice(edges)
            while to_edge == from_edge:
                to_edge = random.choice(edges)
            
            amb_elem = ET.SubElement(routes_root, 'trip')
            amb_elem.set('id', f"ambulance_{i}")
            amb_elem.set('depart', str(depart_time))
            amb_elem.set('from', from_edge)
            amb_elem.set('to', to_edge)
            amb_elem.set('type', 'ambulance')
        
        self._write_xml_pretty(routes_root, str(scenario_file))
        logger.info(f"Created emergency scenario with {len(ambulance_times)} ambulances")
        results['ambulance'] = str(scenario_file)
        
        return results
    
    def create_vehicle_types_additional(self) -> str:
        """Create additional file with vehicle types"""
        output_file = self.routes_dir / "vehicle_types.add.xml"
        return self.create_vehicle_types_file(str(output_file))


def main():
    """Main function to generate all route files"""
    logger.info("=" * 60)
    logger.info("BANGALORE TRAFFIC RL - ROUTE GENERATION")
    logger.info("=" * 60)
    
    generator = TrafficRouteGenerator()
    
    # Find network files
    maps_dir = generator.maps_dir
    net_files = list(maps_dir.glob("*.net.xml"))
    
    if not net_files:
        logger.error(f"No network files found in {maps_dir}")
        logger.info("Please run 01_download_osm.py and 02_convert_to_sumo.py first")
        return
    
    # Use first available network file
    # Prefer full network if available
    net_file = None
    for nf in net_files:
        if 'full' in nf.stem:
            net_file = nf
            break
    
    if net_file is None:
        net_file = net_files[0]
    
    logger.info(f"Using network file: {net_file}")
    
    # Create vehicle types file
    logger.info("\nStep 1: Creating vehicle types...")
    generator.create_vehicle_types_additional()
    
    # Generate baseline routes
    logger.info("\nStep 2: Generating baseline routes...")
    baseline_results = generator.generate_baseline_routes(str(net_file))
    for demand_level, file_path in baseline_results.items():
        logger.info(f"  ✓ {demand_level}: {file_path}")
    
    # Generate realistic routes
    logger.info("\nStep 3: Generating realistic Bangalore routes...")
    realistic_results = generator.generate_realistic_routes(str(net_file))
    for scenario, file_path in realistic_results.items():
        logger.info(f"  ✓ {scenario}: {file_path}")
    
    # Generate emergency scenarios
    logger.info("\nStep 4: Generating emergency scenarios...")
    emergency_results = generator.generate_emergency_scenarios(str(net_file))
    for scenario, file_path in emergency_results.items():
        logger.info(f"  ✓ {scenario}: {file_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("ROUTE GENERATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
