#!/usr/bin/env python3
"""
Traffic Demand Generation Script
Generates realistic Bangalore traffic patterns with multiple modes.

Traffic Modes:
1. Baseline - Light traffic for debugging (10-20 vehicles/lane)
2. Realistic Bangalore - Peak hour congestion (80-150 vehicles at Silk Board)
3. Calibrated - Future: Match Google Maps delay data

Features:
- Time-varying demand (morning peak, evening peak, off-peak)
- Direction-specific flows (inbound/outbound asymmetry)
- Multiple vehicle types (cars, buses, bikes, autos)
- Randomized OD patterns
"""

import os
import sys
import random
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TrafficMode(Enum):
    BASELINE = "baseline"
    REALISTIC_BANGALORE = "realistic_bangalore"
    CALIBRATED = "calibrated"


@dataclass
class TimeInterval:
    """Time interval for demand variation"""
    start: int  # seconds from simulation start
    end: int
    name: str
    multiplier: float = 1.0


@dataclass
class VehicleType:
    """SUMO vehicle type definition"""
    id: str
    accel: float  # m/s²
    decel: float  # m/s²
    sigma: float  # driver imperfection (0-1)
    length: float  # meters
    min_gap: float  # meters
    max_speed: float  # m/s
    color: str  # RGB (0-1 range)
    gui_shape: str
    probability: float  # Distribution probability
    
    # Optional lane change parameters
    lc_strategic: Optional[float] = None
    lc_speed_gain: Optional[float] = None


# ============================================================================
# VEHICLE TYPE DEFINITIONS - Realistic Indian Traffic Mix
# ============================================================================

VEHICLE_TYPES = {
    'car': VehicleType(
        id='car',
        accel=2.6,
        decel=4.5,
        sigma=0.5,
        length=5.0,
        min_gap=2.5,
        max_speed=15.0,  # 54 km/h
        color='1,1,0',  # Yellow
        gui_shape='passenger',
        probability=0.55,  # 55% cars
    ),
    
    'suv': VehicleType(
        id='suv',
        accel=2.3,
        decel=4.2,
        sigma=0.45,
        length=5.5,
        min_gap=2.8,
        max_speed=14.0,
        color='0.8,0.8,0.8',  # Gray
        gui_shape='passenger/sedan',
        probability=0.10,  # 10% SUVs
    ),
    
    'bus': VehicleType(
        id='bus',
        accel=1.2,  # Slow acceleration - causes shockwaves
        decel=3.5,
        sigma=0.3,  # Professional drivers
        length=12.0,  # Long vehicle
        min_gap=3.0,
        max_speed=12.0,  # 43 km/h
        color='0,0,1',  # Blue
        gui_shape='bus',
        probability=0.05,  # 5% buses
    ),
    
    'bike': VehicleType(
        id='bike',
        accel=3.5,  # Fast acceleration
        decel=5.5,
        sigma=0.8,  # High imperfection (aggressive)
        length=2.2,
        min_gap=1.0,  # Can squeeze through
        max_speed=18.0,  # 65 km/h - speedy
        color='1,0,0',  # Red
        gui_shape='motorcycle',
        probability=0.20,  # 20% two-wheelers
        lc_strategic=2.0,  # Aggressive lane changing
        lc_speed_gain=2.0,
    ),
    
    'auto': VehicleType(
        id='auto',
        accel=2.8,
        decel=4.8,
        sigma=0.6,
        length=3.5,
        min_gap=1.5,
        max_speed=13.0,  # 47 km/h
        color='0,1,0',  # Green
        gui_shape='delivery',
        probability=0.08,  # 8% auto-rickshaws
    ),
    
    'ambulance': VehicleType(
        id='ambulance',
        accel=3.0,
        decel=5.0,
        sigma=0.2,  # Very attentive drivers
        length=6.0,
        min_gap=2.0,
        max_speed=20.0,  # 72 km/h - fast when needed
        color='1,0.2,0.2',  # Light red
        gui_shape='emergency',
        probability=0.002,  # Rare
    ),
}


# ============================================================================
# BANGALORE JUNCTION FLOW PATTERNS
# ============================================================================

@dataclass
class JunctionFlowConfig:
    """Traffic flow configuration for a junction"""
    junction_id: str
    name: str
    
    # Base flow rates (vehicles per hour) during off-peak
    base_flows: Dict[str, int]
    
    # Direction classification (for peak asymmetry)
    inbound_directions: List[str]  # Towards city center in morning
    outbound_directions: List[str]  # Away from city center in morning
    
    # Special characteristics
    is_it_corridor: bool = False
    has_bus_stops: bool = False
    
    def get_peak_multiplier(self, direction: str, is_morning: bool) -> float:
        """Get flow multiplier for peak hours"""
        if is_morning:
            if direction in self.inbound_directions:
                return 2.5  # Heavy inbound
            else:
                return 0.7  # Light outbound
        else:  # Evening
            if direction in self.inbound_directions:
                return 0.6  # Light inbound
            else:
                return 2.8  # Heavy outbound


JUNCTION_FLOWS = {
    'silk_board': JunctionFlowConfig(
        junction_id='silk_board',
        name='Silk Board Junction',
        base_flows={
            'north': 800,   # Outer Ring Road
            'south': 900,   # Hosur Road (main artery)
            'east': 700,    # BTM Layout
            'west': 750,    # Bannerghatta
        },
        inbound_directions=['south', 'east'],  # Coming from Hosur/Electronics City
        outbound_directions=['north', 'west'],
    ),
    
    'tin_factory': JunctionFlowConfig(
        junction_id='tin_factory',
        name='Tin Factory Junction',
        base_flows={
            'north': 600,
            'south': 650,
            'east': 500,   # Whitefield direction
            'west': 550,
        },
        inbound_directions=['east', 'north'],  # From Whitefield
        outbound_directions=['west', 'south'],
        is_it_corridor=True,
    ),
    
    'hebbal': JunctionFlowConfig(
        junction_id='hebbal',
        name='Hebbal Junction',
        base_flows={
            'north': 1000,  # NH-44 (heavy)
            'south': 950,   # City center
            'east': 600,    # Manyata Tech Park
            'west': 650,    # Yeshwanthpur
        },
        inbound_directions=['north', 'east'],  # From airport/tech park
        outbound_directions=['south', 'west'],
        has_bus_stops=True,
    ),
    
    'marathahalli': JunctionFlowConfig(
        junction_id='marathahalli',
        name='Marathahalli Junction',
        base_flows={
            'north': 700,
            'south': 700,
            'east': 800,   # ITPL/Whitefield
            'west': 850,   # Koramangala
        },
        inbound_directions=['east', 'north'],  # From IT parks
        outbound_directions=['west', 'south'],
        is_it_corridor=True,
    ),
}


# ============================================================================
# TIME INTERVALS FOR DEMAND VARIATION
# ============================================================================

def get_time_intervals(simulation_hours: int = 24) -> List[TimeInterval]:
    """Get time intervals for a full day simulation."""
    intervals = []
    
    if simulation_hours >= 24:
        # Full day simulation
        intervals = [
            TimeInterval(0, 6*3600, "night", 0.3),
            TimeInterval(6*3600, 8*3600, "early_morning", 0.7),
            TimeInterval(8*3600, 10*3600, "morning_peak", 1.0),  # Peak!
            TimeInterval(10*3600, 12*3600, "late_morning", 0.6),
            TimeInterval(12*3600, 14*3600, "lunch", 0.5),
            TimeInterval(14*3600, 16*3600, "afternoon", 0.6),
            TimeInterval(16*3600, 18*3600, "pre_evening", 0.8),
            TimeInterval(18*3600, 21*3600, "evening_peak", 1.0),  # Peak!
            TimeInterval(21*3600, 24*3600, "night", 0.4),
        ]
    else:
        # Shorter simulation - single peak hour
        intervals = [
            TimeInterval(0, simulation_hours*3600, "peak", 1.0),
        ]
    
    return intervals


# ============================================================================
# ROUTE GENERATION
# ============================================================================

class TrafficDemandGenerator:
    """
    Generates SUMO route files with realistic traffic patterns.
    """
    
    def __init__(
        self,
        mode: TrafficMode = TrafficMode.REALISTIC_BANGALORE,
        seed: int = 42
    ):
        self.mode = mode
        self.seed = seed
        random.seed(seed)
        
        # Mode-specific multipliers
        self.mode_multipliers = {
            TrafficMode.BASELINE: 0.3,  # 30% of normal
            TrafficMode.REALISTIC_BANGALORE: 1.0,  # Full traffic
            TrafficMode.CALIBRATED: 1.2,  # Extra heavy (to be calibrated)
        }
        
    def generate_vehicle_types_xml(self) -> str:
        """Generate vehicle type definitions."""
        lines = []
        
        for vtype in VEHICLE_TYPES.values():
            line = f'''    <vType id="{vtype.id}" 
           accel="{vtype.accel}" 
           decel="{vtype.decel}" 
           sigma="{vtype.sigma}" 
           length="{vtype.length}" 
           minGap="{vtype.min_gap}" 
           maxSpeed="{vtype.max_speed}"
           color="{vtype.color}"
           guiShape="{vtype.gui_shape}"'''
            
            if vtype.lc_strategic:
                line += f'\n           lcStrategic="{vtype.lc_strategic}"'
            if vtype.lc_speed_gain:
                line += f'\n           lcSpeedGain="{vtype.lc_speed_gain}"'
            
            line += '/>'
            lines.append(line)
        
        return '\n'.join(lines)
    
    def generate_flows_xml(
        self,
        junction_flows: Dict[str, JunctionFlowConfig],
        intervals: List[TimeInterval],
        edge_mapping: Dict[str, Dict[str, str]]  # junction -> direction -> edge_id
    ) -> str:
        """
        Generate flow definitions for all junctions and time intervals.
        
        Args:
            junction_flows: Flow configurations per junction
            intervals: Time intervals for demand variation
            edge_mapping: Maps junction/direction to SUMO edge IDs
        """
        lines = []
        flow_id = 0
        
        mode_mult = self.mode_multipliers[self.mode]
        
        for junction_id, config in junction_flows.items():
            for direction, base_rate in config.base_flows.items():
                for interval in intervals:
                    # Calculate flow rate for this interval
                    rate = base_rate * mode_mult * interval.multiplier
                    
                    # Apply peak direction multiplier
                    is_morning = 'morning' in interval.name
                    is_evening = 'evening' in interval.name
                    
                    if is_morning or is_evening:
                        rate *= config.get_peak_multiplier(direction, is_morning)
                    
                    rate = int(rate)
                    if rate <= 0:
                        continue
                    
                    # Get edge ID (or use placeholder)
                    from_edge = edge_mapping.get(junction_id, {}).get(
                        direction, f"{junction_id}_{direction}_in"
                    )
                    to_edge = edge_mapping.get(junction_id, {}).get(
                        f"{direction}_out", f"{junction_id}_{direction}_out"
                    )
                    
                    # Create flow
                    flow_xml = f'''    <flow id="flow_{flow_id}" 
          from="{from_edge}" 
          to="{to_edge}"
          begin="{interval.start}" 
          end="{interval.end}" 
          vehsPerHour="{rate}"
          departLane="best" 
          departSpeed="max">'''
                    
                    # Add vehicle type distribution
                    type_dist = self._get_type_distribution(interval.name, config)
                    flow_xml += f'\n        <param key="typeDistribution" value="{type_dist}"/>'
                    flow_xml += '\n    </flow>'
                    
                    lines.append(flow_xml)
                    flow_id += 1
        
        return '\n'.join(lines)
    
    def _get_type_distribution(
        self,
        interval_name: str,
        config: JunctionFlowConfig
    ) -> str:
        """Get vehicle type distribution based on time and junction."""
        # Base distribution
        dist = {
            'car': 0.55,
            'suv': 0.10,
            'bus': 0.05,
            'bike': 0.20,
            'auto': 0.08,
            'ambulance': 0.002,
        }
        
        # Adjust for peak hours - more buses
        if 'peak' in interval_name:
            dist['bus'] = 0.08
            dist['car'] = 0.52
        
        # Adjust for IT corridors - more cars, fewer autos
        if config.is_it_corridor:
            dist['car'] = 0.60
            dist['auto'] = 0.05
            dist['bike'] = 0.15
        
        # Format as string
        parts = [f"{vtype}:{prob:.3f}" for vtype, prob in dist.items()]
        return ','.join(parts)
    
    def generate_random_trips(
        self,
        duration: int,
        num_vehicles: int,
        edge_list: List[str]
    ) -> str:
        """
        Generate random trips for vehicles.
        Used when we don't have specific OD data.
        """
        lines = []
        
        for i in range(num_vehicles):
            depart = random.uniform(0, duration)
            from_edge = random.choice(edge_list)
            to_edge = random.choice([e for e in edge_list if e != from_edge])
            
            vtype = random.choices(
                list(VEHICLE_TYPES.keys()),
                weights=[v.probability for v in VEHICLE_TYPES.values()]
            )[0]
            
            trip_xml = f'''    <trip id="veh_{i}" 
          depart="{depart:.1f}" 
          from="{from_edge}" 
          to="{to_edge}"
          type="{vtype}"/>'''
            
            lines.append(trip_xml)
        
        # Sort by departure time
        lines.sort(key=lambda x: float(x.split('depart="')[1].split('"')[0]))
        
        return '\n'.join(lines)
    
    def generate_emergency_vehicles(
        self,
        duration: int,
        num_emergencies: int = 5
    ) -> str:
        """Generate scheduled emergency vehicle trips."""
        lines = []
        
        # Distribute emergencies throughout simulation
        for i in range(num_emergencies):
            depart = random.uniform(0.1 * duration, 0.9 * duration)
            
            # Pick random junction
            junction = random.choice(list(JUNCTION_FLOWS.keys()))
            direction = random.choice(['north', 'south', 'east', 'west'])
            
            trip_xml = f'''    <trip id="ambulance_{i}" 
          depart="{depart:.1f}" 
          from="{junction}_{direction}_in" 
          to="{junction}_{direction}_out"
          type="ambulance"/>'''
            
            lines.append(trip_xml)
        
        return '\n'.join(lines)
    
    def generate_full_routes_file(
        self,
        output_file: Path,
        simulation_hours: int = 1,
        include_emergencies: bool = True
    ) -> None:
        """
        Generate complete routes file.
        
        Args:
            output_file: Path for output .rou.xml file
            simulation_hours: Duration in hours
            include_emergencies: Whether to include ambulance trips
        """
        duration = simulation_hours * 3600
        intervals = get_time_intervals(simulation_hours)
        
        # Generate edge mapping (placeholder - would be from real network)
        edge_mapping = self._generate_placeholder_edge_mapping()
        
        # Build XML
        xml_parts = ['<routes>']
        
        # Vehicle types
        xml_parts.append('\n    <!-- Vehicle Type Definitions -->')
        xml_parts.append(self.generate_vehicle_types_xml())
        
        # Flows
        xml_parts.append('\n    <!-- Traffic Flows -->')
        xml_parts.append(self.generate_flows_xml(
            JUNCTION_FLOWS,
            intervals,
            edge_mapping
        ))
        
        # Emergency vehicles
        if include_emergencies:
            xml_parts.append('\n    <!-- Emergency Vehicles -->')
            num_emergencies = max(1, simulation_hours)  # 1 per hour
            xml_parts.append(self.generate_emergency_vehicles(
                duration,
                num_emergencies
            ))
        
        xml_parts.append('\n</routes>')
        
        # Write file
        with open(output_file, 'w') as f:
            f.write('\n'.join(xml_parts))
        
        print(f"Routes file generated: {output_file}")
        print(f"  Mode: {self.mode.value}")
        print(f"  Duration: {simulation_hours} hour(s)")
        print(f"  Time intervals: {len(intervals)}")
    
    def _generate_placeholder_edge_mapping(self) -> Dict[str, Dict[str, str]]:
        """Generate placeholder edge mappings."""
        mapping = {}
        for junction_id in JUNCTION_FLOWS.keys():
            mapping[junction_id] = {
                'north': f"{junction_id}_N_in",
                'south': f"{junction_id}_S_in",
                'east': f"{junction_id}_E_in",
                'west': f"{junction_id}_W_in",
                'north_out': f"{junction_id}_N_out",
                'south_out': f"{junction_id}_S_out",
                'east_out': f"{junction_id}_E_out",
                'west_out': f"{junction_id}_W_out",
            }
        return mapping


def main():
    """Main traffic demand generation workflow."""
    print("="*70)
    print("TRAFFIC DEMAND GENERATION")
    print("="*70)
    
    # Output directory
    routes_dir = PROJECT_ROOT / "data" / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)
    
    # Mode selection
    print("\nTraffic Mode Options:")
    print("1. baseline - Light traffic for debugging")
    print("2. realistic_bangalore - Peak hour congestion")
    print("3. calibrated - Heavy traffic (for calibration)")
    
    mode_input = input("\nSelect mode [1/2/3]: ").strip() or "2"
    mode_map = {
        '1': TrafficMode.BASELINE,
        '2': TrafficMode.REALISTIC_BANGALORE,
        '3': TrafficMode.CALIBRATED,
    }
    mode = mode_map.get(mode_input, TrafficMode.REALISTIC_BANGALORE)
    
    # Duration
    duration_input = input("Simulation duration in hours [1]: ").strip() or "1"
    duration_hours = int(duration_input)
    
    # Generate
    generator = TrafficDemandGenerator(mode=mode, seed=42)
    
    output_file = routes_dir / f"routes_{mode.value}.rou.xml"
    generator.generate_full_routes_file(
        output_file,
        simulation_hours=duration_hours,
        include_emergencies=True
    )
    
    # Also generate a basic routes file for quick testing
    basic_output = routes_dir / "routes.rou.xml"
    generator.generate_full_routes_file(
        basic_output,
        simulation_hours=1,
        include_emergencies=True
    )
    
    print("\n" + "="*70)
    print("GENERATED FILES")
    print("="*70)
    print(f"  Main routes: {output_file}")
    print(f"  Quick test: {basic_output}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. If using real SUMO network, update edge mappings in this script")
    print("2. Run training: python scripts/04_train_agents.py")


if __name__ == "__main__":
    main()
