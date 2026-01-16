"""
Dynamic Route Generator for Bangalore Traffic Simulation

Generates realistic traffic routes based on:
- Actual commuter behavior patterns
- Time-of-day variations
- Zone-based origin-destination flows
- Vehicle type distributions
- Special events and incidents
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class VehicleType:
    """Vehicle type definition for SUMO"""
    id: str
    accel: float
    decel: float
    sigma: float
    length: float
    min_gap: float
    max_speed: float
    gui_shape: str
    color: str
    
    def to_xml(self) -> str:
        return (
            f'<vType id="{self.id}" accel="{self.accel}" decel="{self.decel}" '
            f'sigma="{self.sigma}" length="{self.length}" minGap="{self.min_gap}" '
            f'maxSpeed="{self.max_speed}" guiShape="{self.gui_shape}" color="{self.color}"/>'
        )


@dataclass
class TimeOfDayPattern:
    """Traffic pattern for a specific time period"""
    name: str
    time_range: Tuple[int, int]  # (start_seconds, end_seconds)
    flows: Dict[Tuple[str, str], float]  # (origin_zone, dest_zone) -> proportion
    base_rate: int = 1000  # vehicles/hour


# Default vehicle types for Bangalore traffic
BANGALORE_VEHICLE_TYPES = [
    VehicleType("car_aggressive", 3.0, 5.0, 0.7, 5.0, 2.0, 16.0, "passenger", "1,0,0"),
    VehicleType("car_normal", 2.6, 4.5, 0.5, 5.0, 2.5, 15.0, "passenger", "1,1,0"),
    VehicleType("car_cautious", 2.0, 4.0, 0.3, 5.0, 3.0, 13.0, "passenger", "0,1,0"),
    VehicleType("suv", 2.4, 4.2, 0.5, 5.5, 2.5, 14.0, "passenger/hatchback", "0.5,0.5,0"),
    VehicleType("bus", 1.2, 3.5, 0.3, 12.0, 3.0, 12.0, "bus", "0,0,1"),
    VehicleType("bike", 3.5, 5.5, 0.8, 2.2, 1.0, 18.0, "motorcycle", "1,0.5,0"),
    VehicleType("auto", 2.8, 4.8, 0.6, 3.5, 1.5, 13.0, "delivery", "0,1,1"),
    VehicleType("truck", 1.0, 3.0, 0.4, 15.0, 4.0, 11.0, "truck", "0.5,0.5,0.5"),
    VehicleType("ambulance", 3.5, 6.0, 0.3, 6.0, 2.0, 20.0, "emergency", "1,1,1"),
]


# Default time-of-day patterns for Bangalore
DEFAULT_TOD_PATTERNS = [
    TimeOfDayPattern(
        name="morning_peak",
        time_range=(7*3600, 10*3600),  # 7 AM - 10 AM
        flows={
            ('residential', 'it_hub'): 0.40,
            ('residential', 'commercial'): 0.25,
            ('residential', 'industrial'): 0.15,
            ('it_hub', 'commercial'): 0.10,
            ('it_hub', 'it_hub'): 0.10
        },
        base_rate=1500
    ),
    TimeOfDayPattern(
        name="midday",
        time_range=(10*3600, 13*3600),  # 10 AM - 1 PM
        flows={
            ('residential', 'commercial'): 0.30,
            ('commercial', 'commercial'): 0.25,
            ('it_hub', 'commercial'): 0.20,
            ('residential', 'residential'): 0.15,
            ('industrial', 'commercial'): 0.10
        },
        base_rate=800
    ),
    TimeOfDayPattern(
        name="afternoon",
        time_range=(13*3600, 17*3600),  # 1 PM - 5 PM
        flows={
            ('commercial', 'commercial'): 0.25,
            ('residential', 'commercial'): 0.25,
            ('it_hub', 'commercial'): 0.20,
            ('commercial', 'residential'): 0.15,
            ('it_hub', 'it_hub'): 0.15
        },
        base_rate=900
    ),
    TimeOfDayPattern(
        name="evening_peak",
        time_range=(17*3600, 21*3600),  # 5 PM - 9 PM
        flows={
            ('it_hub', 'residential'): 0.45,
            ('commercial', 'residential'): 0.30,
            ('industrial', 'residential'): 0.15,
            ('commercial', 'it_hub'): 0.05,
            ('it_hub', 'it_hub'): 0.05
        },
        base_rate=1600
    ),
    TimeOfDayPattern(
        name="night",
        time_range=(21*3600, 24*3600),  # 9 PM - Midnight
        flows={
            ('commercial', 'residential'): 0.40,
            ('it_hub', 'residential'): 0.30,
            ('residential', 'residential'): 0.20,
            ('commercial', 'commercial'): 0.10
        },
        base_rate=400
    ),
    TimeOfDayPattern(
        name="early_morning",
        time_range=(0, 7*3600),  # Midnight - 7 AM
        flows={
            ('residential', 'residential'): 0.30,
            ('industrial', 'industrial'): 0.25,
            ('residential', 'industrial'): 0.25,
            ('residential', 'commercial'): 0.20
        },
        base_rate=200
    ),
]


class BangaloreRouteGenerator:
    """
    Generates realistic traffic routes based on actual urban patterns
    
    Features:
    - Zone-based origin-destination modeling
    - Time-of-day traffic variations
    - Realistic vehicle type distributions
    - Special event handling
    - Network-aware routing
    """
    
    def __init__(self, network_analyzer=None):
        """
        Initialize route generator
        
        Args:
            network_analyzer: Optional NetworkAnalyzer instance for topology info
        """
        self.analyzer = network_analyzer
        
        # Define zones (simplified Bangalore areas)
        self.zones = {
            'residential': [
                'koramangala', 'indiranagar', 'jayanagar', 'malleshwaram',
                'btm', 'hsr', 'jp_nagar', 'banashankari'
            ],
            'commercial': [
                'mg_road', 'brigade_road', 'commercial_street', 'ub_city',
                'forum', 'mantri_mall'
            ],
            'it_hub': [
                'whitefield', 'electronic_city', 'outer_ring_road', 'manyata',
                'bagmane', 'ecospace'
            ],
            'industrial': [
                'peenya', 'bommasandra', 'jigani', 'doddaballapur'
            ]
        }
        
        # Junction to zone mapping cache
        self._zone_cache = {}
        
        # Vehicle types
        self.vehicle_types = BANGALORE_VEHICLE_TYPES
        
        # Time-of-day patterns
        self.tod_patterns = DEFAULT_TOD_PATTERNS
    
    def map_junction_to_zone(self, junction_id: str) -> str:
        """
        Map junction to urban zone type
        
        Uses geographic heuristics and junction naming patterns
        """
        if junction_id in self._zone_cache:
            return self._zone_cache[junction_id]
        
        junction_lower = junction_id.lower()
        
        # Check for specific area names
        for zone, areas in self.zones.items():
            for area in areas:
                if area in junction_lower:
                    self._zone_cache[junction_id] = zone
                    return zone
        
        # IT hub indicators
        it_keywords = ['silk', 'electronic', 'whitefield', 'ring_road', 'orr', 'tech']
        if any(kw in junction_lower for kw in it_keywords):
            self._zone_cache[junction_id] = 'it_hub'
            return 'it_hub'
        
        # Commercial indicators
        commercial_keywords = ['mg', 'brigade', 'commercial', 'mall', 'forum']
        if any(kw in junction_lower for kw in commercial_keywords):
            self._zone_cache[junction_id] = 'commercial'
            return 'commercial'
        
        # Industrial indicators
        industrial_keywords = ['peenya', 'industrial', 'factory']
        if any(kw in junction_lower for kw in industrial_keywords):
            self._zone_cache[junction_id] = 'industrial'
            return 'industrial'
        
        # Default to residential
        self._zone_cache[junction_id] = 'residential'
        return 'residential'
    
    def get_zone_junctions(self, zone_type: str) -> List[str]:
        """
        Get all junctions belonging to a zone
        """
        if self.analyzer is None:
            return []
        
        junctions = []
        for jid in self.analyzer.junctions.keys():
            if self.map_junction_to_zone(jid) == zone_type:
                junctions.append(jid)
        
        # If no junctions found, return first few junctions
        if not junctions:
            return list(self.analyzer.junctions.keys())[:5]
        
        return junctions
    
    def get_pattern_for_time(self, time_seconds: int) -> TimeOfDayPattern:
        """
        Get traffic pattern for given time
        """
        hour = (time_seconds / 3600) % 24
        
        for pattern in self.tod_patterns:
            start_hour = pattern.time_range[0] / 3600
            end_hour = pattern.time_range[1] / 3600
            
            if start_hour <= hour < end_hour:
                return pattern
        
        # Default to early morning
        return self.tod_patterns[-1]
    
    def get_vtype_distribution(
        self, 
        time_seconds: int, 
        origin_zone: str
    ) -> Dict[str, float]:
        """
        Get vehicle type distribution based on time and zone
        
        Returns:
            Dict mapping vehicle_type_id -> probability
        """
        hour = (time_seconds / 3600) % 24
        
        # Peak hours: more cars and bikes
        if 7 <= hour < 10 or 17 <= hour < 21:
            dist = {
                'car_normal': 0.40,
                'car_aggressive': 0.10,
                'car_cautious': 0.05,
                'suv': 0.10,
                'bike': 0.20,
                'bus': 0.05,
                'auto': 0.08,
                'truck': 0.02
            }
        # Midday: more varied
        elif 10 <= hour < 17:
            dist = {
                'car_normal': 0.35,
                'car_cautious': 0.10,
                'suv': 0.08,
                'bike': 0.20,
                'bus': 0.07,
                'auto': 0.10,
                'truck': 0.08,
                'ambulance': 0.02
            }
        # Night: fewer vehicles
        else:
            dist = {
                'car_normal': 0.45,
                'car_cautious': 0.15,
                'suv': 0.10,
                'bike': 0.10,
                'auto': 0.10,
                'truck': 0.08,
                'ambulance': 0.02
            }
        
        # Zone adjustments
        if origin_zone == 'it_hub':
            dist['car_normal'] += 0.10
            dist['bike'] -= 0.05
            dist['auto'] -= 0.05
        elif origin_zone == 'industrial':
            dist['truck'] += 0.15
            dist['car_normal'] -= 0.10
            dist['bike'] -= 0.05
        
        # Normalize
        total = sum(dist.values())
        return {k: v/total for k, v in dist.items()}
    
    def sample_vehicle_type(
        self, 
        time_seconds: int, 
        origin_zone: str
    ) -> str:
        """Sample a random vehicle type based on distribution"""
        dist = self.get_vtype_distribution(time_seconds, origin_zone)
        types = list(dist.keys())
        probs = list(dist.values())
        return np.random.choice(types, p=probs)
    
    def path_to_edges(self, junction_path: List[str]) -> List[str]:
        """
        Convert junction path to edge IDs
        """
        if self.analyzer is None:
            return []
        
        edges = []
        for i in range(len(junction_path) - 1):
            from_j = junction_path[i]
            to_j = junction_path[i + 1]
            
            # Find edge connecting these junctions
            for eid, edata in self.analyzer.edges.items():
                if edata.from_junction == from_j and edata.to_junction == to_j:
                    edges.append(eid)
                    break
        
        return edges
    
    def generate_vehicle_types_xml(self) -> str:
        """Generate XML for all vehicle types"""
        lines = ['<!-- Vehicle Type Definitions for Bangalore Traffic -->']
        for vtype in self.vehicle_types:
            lines.append(vtype.to_xml())
        return '\n'.join(lines)
    
    def generate_time_varying_routes(
        self,
        start_time: int,
        end_time: int,
        output_file: str,
        scale_factor: float = 1.0
    ) -> int:
        """
        Generate complete route file with time-varying demand
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            output_file: Path to output .rou.xml file
            scale_factor: Scale traffic volume (1.0 = normal)
            
        Returns:
            Number of flows generated
        """
        if self.analyzer is None:
            raise RuntimeError("NetworkAnalyzer required for route generation")
        
        # Ensure OD matrix is computed
        if self.analyzer.od_paths is None:
            self.analyzer.compute_od_matrix()
        
        routes_xml = ['<?xml version="1.0" encoding="UTF-8"?>']
        routes_xml.append('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                         'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">')
        
        # Add vehicle types
        routes_xml.append('\n<!-- Vehicle Types -->')
        for vtype in self.vehicle_types:
            routes_xml.append(f'    {vtype.to_xml()}')
        
        routes_xml.append('\n<!-- Routes and Flows -->')
        
        current_time = start_time
        flow_id = 0
        route_id = 0
        
        while current_time < end_time:
            # Get pattern for this time
            pattern = self.get_pattern_for_time(current_time)
            
            # Period end (1-hour blocks)
            period_end = min(current_time + 3600, end_time)
            
            # Generate flows for each OD pair
            for (origin_zone, dest_zone), flow_fraction in pattern.flows.items():
                # Get junctions in these zones
                origin_junctions = self.get_zone_junctions(origin_zone)
                dest_junctions = self.get_zone_junctions(dest_zone)
                
                if not origin_junctions or not dest_junctions:
                    continue
                
                # Calculate flow rate
                flow_rate = int(pattern.base_rate * flow_fraction * scale_factor)
                
                # Distribute among junction pairs
                pairs_to_generate = min(3, len(origin_junctions) * len(dest_junctions))
                
                for _ in range(pairs_to_generate):
                    origin_j = random.choice(origin_junctions)
                    dest_j = random.choice(dest_junctions)
                    
                    if origin_j == dest_j:
                        continue
                    
                    # Get route
                    path = self.analyzer.od_paths.get((origin_j, dest_j))
                    if not path or len(path) < 2:
                        continue
                    
                    # Convert to edges
                    route_edges = self.path_to_edges(path)
                    if not route_edges:
                        continue
                    
                    # Create route
                    route_name = f"route_{route_id}"
                    routes_xml.append(f'    <route id="{route_name}" edges="{" ".join(route_edges)}"/>')
                    route_id += 1
                    
                    # Sample vehicle type
                    vtype = self.sample_vehicle_type(current_time, origin_zone)
                    
                    # Create flow
                    per_pair_rate = max(1, flow_rate // pairs_to_generate)
                    routes_xml.append(
                        f'    <flow id="flow_{flow_id}" route="{route_name}" '
                        f'begin="{current_time}" end="{period_end}" '
                        f'vehsPerHour="{per_pair_rate}" '
                        f'type="{vtype}" departLane="best" departSpeed="max"/>'
                    )
                    flow_id += 1
            
            current_time = period_end
        
        routes_xml.append('</routes>')
        
        # Write to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(routes_xml))
        
        print(f"Generated {flow_id} flows, {route_id} routes in {output_file}")
        return flow_id
    
    def generate_individual_trips(
        self,
        start_time: int,
        end_time: int,
        output_file: str,
        n_vehicles: int = 1000
    ) -> int:
        """
        Generate individual vehicle trips instead of flows
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            output_file: Path to output .trips.xml file
            n_vehicles: Total number of vehicles to generate
            
        Returns:
            Number of trips generated
        """
        if self.analyzer is None:
            raise RuntimeError("NetworkAnalyzer required for trip generation")
        
        if self.analyzer.od_paths is None:
            self.analyzer.compute_od_matrix()
        
        trips_xml = ['<?xml version="1.0" encoding="UTF-8"?>']
        trips_xml.append('<trips>')
        
        # Add vehicle types
        for vtype in self.vehicle_types:
            trips_xml.append(f'    {vtype.to_xml()}')
        
        duration = end_time - start_time
        trip_count = 0
        
        for i in range(n_vehicles):
            # Random departure time
            depart = start_time + random.random() * duration
            
            # Get pattern for this time
            pattern = self.get_pattern_for_time(int(depart))
            
            # Sample OD pair based on pattern
            od_pairs = list(pattern.flows.keys())
            od_probs = list(pattern.flows.values())
            origin_zone, dest_zone = random.choices(od_pairs, weights=od_probs)[0]
            
            # Get random junctions
            origin_junctions = self.get_zone_junctions(origin_zone)
            dest_junctions = self.get_zone_junctions(dest_zone)
            
            if not origin_junctions or not dest_junctions:
                continue
            
            origin_j = random.choice(origin_junctions)
            dest_j = random.choice(dest_junctions)
            
            if origin_j == dest_j:
                continue
            
            # Get route
            path = self.analyzer.od_paths.get((origin_j, dest_j))
            if not path or len(path) < 2:
                continue
            
            route_edges = self.path_to_edges(path)
            if not route_edges:
                continue
            
            # Sample vehicle type
            vtype = self.sample_vehicle_type(int(depart), origin_zone)
            
            # Create trip
            trips_xml.append(
                f'    <trip id="veh_{i}" depart="{depart:.2f}" '
                f'from="{route_edges[0]}" to="{route_edges[-1]}" '
                f'type="{vtype}"/>'
            )
            trip_count += 1
        
        trips_xml.append('</trips>')
        
        # Write to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(trips_xml))
        
        print(f"Generated {trip_count} trips in {output_file}")
        return trip_count
    
    def add_emergency_vehicles(
        self,
        routes_xml: List[str],
        n_emergencies: int = 5,
        time_range: Tuple[int, int] = (0, 3600)
    ) -> List[str]:
        """
        Add emergency vehicle trips
        
        Args:
            routes_xml: Existing routes XML lines
            n_emergencies: Number of emergency vehicles
            time_range: Time range for emergencies
            
        Returns:
            Updated routes XML lines
        """
        if self.analyzer is None or not self.analyzer.junctions:
            return routes_xml
        
        junctions = list(self.analyzer.junctions.keys())
        
        for i in range(n_emergencies):
            depart = random.uniform(time_range[0], time_range[1])
            
            # Random origin and destination
            origin_j = random.choice(junctions)
            dest_j = random.choice(junctions)
            
            while dest_j == origin_j:
                dest_j = random.choice(junctions)
            
            path = self.analyzer.od_paths.get((origin_j, dest_j))
            if not path:
                continue
            
            route_edges = self.path_to_edges(path)
            if not route_edges:
                continue
            
            # Insert before closing tag
            routes_xml.insert(-1, 
                f'    <trip id="emergency_{i}" depart="{depart:.2f}" '
                f'from="{route_edges[0]}" to="{route_edges[-1]}" '
                f'type="ambulance"/>'
            )
        
        return routes_xml
    
    def add_special_event(
        self,
        output_file: str,
        event_junction: str,
        event_time: int,
        event_type: str = 'concert',
        duration: int = 7200
    ):
        """
        Generate additional traffic for special events
        
        Args:
            output_file: Path to output file
            event_junction: Junction near event
            event_time: Event start time (seconds)
            event_type: Type of event ('concert', 'match', 'festival')
            duration: Event duration (seconds)
        """
        if self.analyzer is None:
            return
        
        # Get traffic multiplier based on event type
        multipliers = {
            'concert': 3.0,
            'match': 4.0,
            'festival': 2.5
        }
        multiplier = multipliers.get(event_type, 2.0)
        
        # Get junctions that route through event junction
        event_flows = []
        
        for (origin, dest), path in self.analyzer.od_paths.items():
            if event_junction in path:
                event_flows.append((origin, dest, path))
        
        # Generate additional flows
        flows_xml = ['<?xml version="1.0" encoding="UTF-8"?>']
        flows_xml.append(f'<!-- Special Event: {event_type} at {event_junction} -->')
        flows_xml.append('<routes>')
        
        for i, (origin, dest, path) in enumerate(event_flows[:10]):  # Limit flows
            route_edges = self.path_to_edges(path)
            if not route_edges:
                continue
            
            flows_xml.append(f'    <route id="event_route_{i}" edges="{" ".join(route_edges)}"/>')
            flows_xml.append(
                f'    <flow id="event_flow_{i}" route="event_route_{i}" '
                f'begin="{event_time - 1800}" end="{event_time + duration}" '
                f'vehsPerHour="{int(200 * multiplier)}" type="car_cautious"/>'
            )
        
        flows_xml.append('</routes>')
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(flows_xml))
        
        print(f"Generated event traffic file: {output_file}")


# Command-line interface
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate realistic Bangalore traffic routes')
    parser.add_argument('--net-file', required=True, help='Path to .net.xml file')
    parser.add_argument('--output', default='bangalore_routes.rou.xml', help='Output route file')
    parser.add_argument('--start', type=int, default=0, help='Start time (seconds)')
    parser.add_argument('--end', type=int, default=3600, help='End time (seconds)')
    parser.add_argument('--scale', type=float, default=1.0, help='Traffic scale factor')
    parser.add_argument('--trips', action='store_true', help='Generate individual trips instead of flows')
    parser.add_argument('--n-vehicles', type=int, default=1000, help='Number of vehicles for trips mode')
    
    args = parser.parse_args()
    
    # Import and use NetworkAnalyzer
    from network_analyzer import NetworkAnalyzer
    
    print(f"Loading network: {args.net_file}")
    analyzer = NetworkAnalyzer(args.net_file)
    
    print("Computing OD matrix...")
    analyzer.compute_od_matrix()
    
    generator = BangaloreRouteGenerator(analyzer)
    
    if args.trips:
        generator.generate_individual_trips(
            args.start, args.end, args.output, args.n_vehicles
        )
    else:
        generator.generate_time_varying_routes(
            args.start, args.end, args.output, args.scale
        )
