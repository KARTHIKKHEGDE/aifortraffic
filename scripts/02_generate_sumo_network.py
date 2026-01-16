#!/usr/bin/env python3
"""
SUMO Network Generation Script
Converts OSM data to SUMO network format with proper traffic light inference.

This script:
1. Reads OSM files from data/osm/
2. Converts to SUMO network format
3. Identifies and catalogs traffic light junctions
4. Creates network configuration files
"""

import os
import sys
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class JunctionInfo:
    """Information about a detected junction"""
    id: str
    x: float
    y: float
    junction_type: str
    incoming_lanes: int
    outgoing_lanes: int
    tls_id: Optional[str] = None


# ============================================================================
# NETCONVERT OPTIONS EXPLAINED
# ============================================================================

NETCONVERT_OPTIONS = {
    # Geometry simplification
    '--geometry.remove': 'Remove unnecessary geometry nodes (simplifies network)',
    '--geometry.max-segment-length': 'Maximum segment length (100m default)',
    
    # Ramp detection
    '--ramps.guess': 'Automatically detect highway ramps',
    '--ramps.set': 'Manually specify ramp edges',
    
    # Junction handling
    '--junctions.join': 'Merge close junctions into one (important for complex intersections)',
    '--junctions.join-dist': 'Distance for junction merging (default 20m)',
    '--junctions.corner-detail': 'Detail level for junction corners (5 recommended)',
    '--junctions.limit-turn-speed': 'Limit turning speed at junctions',
    
    # Traffic light inference
    '--tls.guess': 'Guess which junctions should have traffic lights',
    '--tls.guess-signals': 'Detect traffic signals from OSM data',
    '--tls.discard-simple': 'Remove TLS from simple 2-way junctions',
    '--tls.join': 'Join nearby traffic lights into coordinated systems',
    '--tls.default-type': 'Default TLS type (static, actuated)',
    '--tls.layout': 'TLS layout algorithm (opposites, incoming, alternateOneWay)',
    '--tls.min-dur': 'Minimum green phase duration',
    '--tls.max-dur': 'Maximum green phase duration',
    '--tls.yellow.time': 'Yellow phase duration',
    '--tls.all-red.time': 'All-red clearance time',
    
    # Speed and lane handling
    '--default.speed': 'Default speed limit if not specified',
    '--default.lanewidth': 'Default lane width',
    '--lanes.from-osm': 'Use lane count from OSM data',
    
    # Output options
    '--output.street-names': 'Keep street names from OSM',
    '--output.original-names': 'Keep original OSM IDs',
    '--output-file': 'Output network file path',
    
    # Additional processing
    '--no-internal-links': 'Skip internal junction links (faster, less accurate)',
    '--no-turnarounds': 'Disable U-turn connections',
    '--remove-edges.isolated': 'Remove disconnected edges',
}


def check_sumo_installation() -> Tuple[bool, str]:
    """
    Check if SUMO is properly installed and accessible.
    
    Returns:
        Tuple of (is_installed, sumo_home_path)
    """
    sumo_home = os.environ.get('SUMO_HOME')
    
    if sumo_home and Path(sumo_home).exists():
        print(f"✓ SUMO_HOME found: {sumo_home}")
        
        # Check for netconvert
        netconvert = Path(sumo_home) / 'bin' / 'netconvert'
        if sys.platform == 'win32':
            netconvert = netconvert.with_suffix('.exe')
        
        if netconvert.exists():
            print(f"✓ netconvert found: {netconvert}")
            return True, sumo_home
        else:
            print(f"✗ netconvert not found at {netconvert}")
    else:
        print("✗ SUMO_HOME environment variable not set")
    
    # Try to find netconvert in PATH
    try:
        result = subprocess.run(
            ['netconvert', '--version'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ netconvert found in PATH")
            return True, ""
    except FileNotFoundError:
        pass
    
    return False, ""


def convert_osm_to_net(
    osm_file: Path,
    output_net: Path,
    options: Optional[Dict[str, str]] = None
) -> bool:
    """
    Convert OSM file to SUMO network using netconvert.
    
    Args:
        osm_file: Path to input .osm file
        output_net: Path for output .net.xml file
        options: Additional netconvert options
    
    Returns:
        True if conversion successful
    """
    print(f"\nConverting: {osm_file.name}")
    print(f"Output: {output_net.name}")
    
    # Build command
    cmd = [
        'netconvert',
        '--osm-files', str(osm_file),
        '--output-file', str(output_net),
        
        # Geometry
        '--geometry.remove', 'true',
        '--geometry.max-segment-length', '100',
        
        # Junctions
        '--junctions.join', 'true',
        '--junctions.join-dist', '20',
        '--junctions.corner-detail', '5',
        '--junctions.limit-turn-speed', '5.5',
        
        # Traffic lights
        '--tls.guess-signals', 'true',
        '--tls.discard-simple', 'true',
        '--tls.join', 'true',
        '--tls.default-type', 'static',
        '--tls.layout', 'opposites',
        '--tls.min-dur', '10',
        '--tls.max-dur', '90',
        '--tls.yellow.time', '3',
        '--tls.all-red.time', '2',
        
        # Speed/lanes
        '--default.speed', '13.89',  # 50 km/h in m/s
        '--default.lanewidth', '3.2',
        
        # Output
        '--output.street-names', 'true',
        '--output.original-names', 'true',
        
        # Cleanup
        '--remove-edges.isolated', 'true',
        '--no-turnarounds', 'true',
        
        # Warnings
        '--no-warnings', 'false',
    ]
    
    # Add custom options
    if options:
        for key, value in options.items():
            if value is True:
                cmd.extend([key, 'true'])
            elif value is False:
                cmd.extend([key, 'false'])
            elif value is not None:
                cmd.extend([key, str(value)])
    
    print(f"Command: {' '.join(cmd[:6])}...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"✓ Conversion successful!")
            if result.stderr:
                # Print warnings (non-fatal)
                warnings = [l for l in result.stderr.split('\n') if 'Warning' in l]
                if warnings:
                    print(f"  Warnings: {len(warnings)}")
            return True
        else:
            print(f"✗ Conversion failed!")
            print(f"  Error: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Conversion timed out (>5 minutes)")
        return False
    except FileNotFoundError:
        print("✗ netconvert not found. Please install SUMO.")
        return False


def analyze_network(net_file: Path) -> Dict[str, any]:
    """
    Analyze generated network and extract junction information.
    
    Args:
        net_file: Path to .net.xml file
    
    Returns:
        Dictionary with network statistics
    """
    print(f"\nAnalyzing network: {net_file.name}")
    
    try:
        tree = ET.parse(net_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"✗ Failed to parse: {e}")
        return {}
    
    stats = {
        'edges': [],
        'junctions': [],
        'traffic_lights': [],
        'lanes_total': 0,
        'length_total': 0.0,
    }
    
    # Parse edges
    for edge in root.findall('.//edge'):
        if edge.get('function') != 'internal':
            edge_info = {
                'id': edge.get('id'),
                'from': edge.get('from'),
                'to': edge.get('to'),
                'lanes': len(edge.findall('lane')),
            }
            stats['edges'].append(edge_info)
            stats['lanes_total'] += edge_info['lanes']
            
            # Sum lane lengths
            for lane in edge.findall('lane'):
                length = float(lane.get('length', 0))
                stats['length_total'] += length
    
    # Parse junctions
    for junction in root.findall('.//junction'):
        jtype = junction.get('type')
        if jtype != 'internal':
            junction_info = JunctionInfo(
                id=junction.get('id'),
                x=float(junction.get('x', 0)),
                y=float(junction.get('y', 0)),
                junction_type=jtype,
                incoming_lanes=len(junction.get('incLanes', '').split()),
                outgoing_lanes=len(junction.get('intLanes', '').split()),
            )
            stats['junctions'].append(junction_info)
    
    # Parse traffic lights
    for tls in root.findall('.//tlLogic'):
        tls_info = {
            'id': tls.get('id'),
            'type': tls.get('type'),
            'program_id': tls.get('programID'),
            'offset': float(tls.get('offset', 0)),
            'phases': len(tls.findall('phase')),
        }
        stats['traffic_lights'].append(tls_info)
        
        # Link TLS to junction
        for j in stats['junctions']:
            if j.id == tls_info['id'] or f"cluster_{j.id}" == tls_info['id']:
                j.tls_id = tls_info['id']
    
    # Print summary
    print(f"  Edges: {len(stats['edges'])}")
    print(f"  Total lanes: {stats['lanes_total']}")
    print(f"  Total length: {stats['length_total']/1000:.2f} km")
    print(f"  Junctions: {len(stats['junctions'])}")
    print(f"  Traffic lights: {len(stats['traffic_lights'])}")
    
    # List traffic light junctions
    tls_junctions = [j for j in stats['junctions'] if j.junction_type == 'traffic_light']
    print(f"\n  Traffic Light Junctions ({len(tls_junctions)}):")
    for j in tls_junctions[:10]:  # Show first 10
        print(f"    - {j.id}: {j.incoming_lanes} incoming lanes at ({j.x:.1f}, {j.y:.1f})")
    
    if len(tls_junctions) > 10:
        print(f"    ... and {len(tls_junctions) - 10} more")
    
    return stats


def find_target_junctions(stats: Dict, target_coords: Dict) -> Dict[str, JunctionInfo]:
    """
    Find junctions closest to our target coordinates.
    
    Args:
        stats: Network statistics from analyze_network
        target_coords: Dict mapping junction name to (lat, lon) tuple
    
    Returns:
        Dict mapping junction name to closest JunctionInfo
    """
    from math import sqrt
    
    print("\nFinding target junctions...")
    
    # Note: SUMO uses Cartesian coordinates, not lat/lon
    # We'll find junctions by relative position
    
    results = {}
    tls_junctions = [j for j in stats['junctions'] if j.junction_type == 'traffic_light']
    
    if not tls_junctions:
        print("  No traffic light junctions found!")
        return results
    
    # Find centroid
    cx = sum(j.x for j in tls_junctions) / len(tls_junctions)
    cy = sum(j.y for j in tls_junctions) / len(tls_junctions)
    
    # Sort by distance from centroid
    for j in tls_junctions:
        j.dist_from_center = sqrt((j.x - cx)**2 + (j.y - cy)**2)
    
    # For now, just take the 4 largest junctions (by incoming lanes)
    sorted_junctions = sorted(tls_junctions, key=lambda j: -j.incoming_lanes)
    
    junction_names = ['silk_board', 'tin_factory', 'hebbal', 'marathahalli']
    for i, name in enumerate(junction_names):
        if i < len(sorted_junctions):
            results[name] = sorted_junctions[i]
            print(f"  {name}: {sorted_junctions[i].id} ({sorted_junctions[i].incoming_lanes} lanes)")
    
    return results


def create_junction_mapping_file(
    results: Dict[str, JunctionInfo],
    output_file: Path
) -> None:
    """
    Create a YAML file mapping friendly names to SUMO junction IDs.
    """
    import yaml
    
    mapping = {}
    for name, junction in results.items():
        mapping[name] = {
            'sumo_id': junction.id,
            'tls_id': junction.tls_id or junction.id,
            'x': junction.x,
            'y': junction.y,
            'incoming_lanes': junction.incoming_lanes,
        }
    
    with open(output_file, 'w') as f:
        yaml.dump(mapping, f, default_flow_style=False)
    
    print(f"\nJunction mapping saved to: {output_file}")


def create_sumo_config(
    net_file: Path,
    output_dir: Path,
    simulation_time: int = 3600
) -> Path:
    """
    Create a basic SUMO configuration file.
    """
    config_file = output_dir / "bangalore.sumocfg"
    
    config_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="{net_file.name}"/>
        <route-files value="routes.rou.xml"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="{simulation_time}"/>
        <step-length value="1"/>
    </time>

    <processing>
        <time-to-teleport value="-1"/>
        <collision.action value="warn"/>
        <collision.check-junctions value="true"/>
    </processing>

    <output>
        <tripinfo-output value="tripinfo.xml"/>
        <queue-output value="queue.xml"/>
        <statistic-output value="statistics.xml"/>
    </output>

    <report>
        <verbose value="true"/>
        <no-warnings value="false"/>
        <duration-log.statistics value="true"/>
    </report>

    <gui_only>
        <gui-settings-file value="gui-settings.xml"/>
    </gui_only>

</configuration>
"""
    
    with open(config_file, 'w') as f:
        f.write(config_xml)
    
    print(f"SUMO config saved to: {config_file}")
    
    # Create GUI settings
    gui_file = output_dir / "gui-settings.xml"
    gui_xml = """<?xml version="1.0" encoding="UTF-8"?>
<viewsettings>
    <viewport y="0" x="0" zoom="100"/>
    <delay value="100"/>
    <scheme name="real world"/>
    <showGrid value="false"/>
</viewsettings>
"""
    with open(gui_file, 'w') as f:
        f.write(gui_xml)
    
    return config_file


def main():
    """Main network generation workflow."""
    print("="*70)
    print("SUMO NETWORK GENERATION")
    print("="*70)
    
    # Check SUMO installation
    sumo_ok, sumo_home = check_sumo_installation()
    if not sumo_ok:
        print("\n⚠️  SUMO is not properly installed.")
        print("Please install SUMO from: https://eclipse.dev/sumo/")
        print("And set the SUMO_HOME environment variable.")
        
        # Continue anyway for demo/testing without actual conversion
        print("\nContinuing with mock network for development...")
    
    # Directories
    osm_dir = PROJECT_ROOT / "data" / "osm"
    net_dir = PROJECT_ROOT / "data" / "networks"
    net_dir.mkdir(parents=True, exist_ok=True)
    
    # Find OSM files
    osm_files = list(osm_dir.glob("*.osm"))
    
    if not osm_files:
        print(f"\n⚠️  No OSM files found in {osm_dir}")
        print("Run 01_download_osm_data.py first to download map data.")
        
        # Create mock network for development
        print("\nCreating mock network for development...")
        create_mock_network(net_dir)
        return
    
    print(f"\nFound {len(osm_files)} OSM file(s):")
    for f in osm_files:
        print(f"  - {f.name}")
    
    # Convert each file
    all_stats = {}
    for osm_file in osm_files:
        net_file = net_dir / osm_file.with_suffix('.net.xml').name
        
        if sumo_ok:
            success = convert_osm_to_net(osm_file, net_file)
            if success:
                all_stats[osm_file.stem] = analyze_network(net_file)
        else:
            print(f"  Skipping {osm_file.name} (SUMO not available)")
    
    # Create combined config if we have the combined network
    combined_net = net_dir / "bangalore_combined.net.xml"
    if combined_net.exists():
        create_sumo_config(combined_net, net_dir)
        
        # Find and map target junctions
        if 'bangalore_combined' in all_stats:
            target_junctions = find_target_junctions(
                all_stats['bangalore_combined'],
                {  # Target coordinates
                    'silk_board': (12.9173, 77.6228),
                    'tin_factory': (12.9988, 77.6515),
                    'hebbal': (13.0358, 77.5970),
                    'marathahalli': (12.9591, 77.7011),
                }
            )
            
            if target_junctions:
                mapping_file = net_dir / "junction_mapping.yaml"
                create_junction_mapping_file(target_junctions, mapping_file)
    
    # Next steps
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review generated network in SUMO-GUI:")
    print(f"   sumo-gui -n {net_dir}/bangalore_combined.net.xml")
    print("")
    print("2. Generate traffic demand:")
    print("   python scripts/03_generate_traffic_demand.py")


def create_mock_network(net_dir: Path):
    """Create a simple mock network for development without SUMO."""
    mock_net = net_dir / "mock_bangalore.net.xml"
    
    # Simple 4-junction grid network
    mock_xml = """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.16" junctionCornerDetail="5" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
     xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">

    <!-- Mock network for development - replace with real Bangalore network -->
    
    <!-- Junction definitions -->
    <junction id="silk_board" type="traffic_light" x="0" y="0" 
              incLanes="E_N_0 E_N_1 E_S_0 E_S_1 E_E_0 E_E_1 E_W_0 E_W_1"
              intLanes=""/>
    <junction id="tin_factory" type="traffic_light" x="1000" y="500"
              incLanes="TF_N_0 TF_N_1 TF_S_0 TF_S_1 TF_E_0 TF_E_1 TF_W_0 TF_W_1"
              intLanes=""/>
    <junction id="hebbal" type="traffic_light" x="500" y="1500"
              incLanes="HB_N_0 HB_N_1 HB_S_0 HB_S_1 HB_E_0 HB_E_1 HB_W_0 HB_W_1"
              intLanes=""/>
    <junction id="marathahalli" type="traffic_light" x="1500" y="0"
              incLanes="MH_N_0 MH_N_1 MH_S_0 MH_S_1 MH_E_0 MH_E_1 MH_W_0 MH_W_1"
              intLanes=""/>

    <!-- Traffic light programs -->
    <tlLogic id="silk_board" type="static" programID="0" offset="0">
        <phase duration="30" state="GGGGrrrrGGGGrrrr"/>
        <phase duration="3" state="yyyyrrrryyyyrrrr"/>
        <phase duration="30" state="rrrrGGGGrrrrGGGG"/>
        <phase duration="3" state="rrrryyyyrrrryyyy"/>
    </tlLogic>
    
    <tlLogic id="tin_factory" type="static" programID="0" offset="0">
        <phase duration="30" state="GGGGrrrrGGGGrrrr"/>
        <phase duration="3" state="yyyyrrrryyyyrrrr"/>
        <phase duration="30" state="rrrrGGGGrrrrGGGG"/>
        <phase duration="3" state="rrrryyyyrrrryyyy"/>
    </tlLogic>
    
    <tlLogic id="hebbal" type="static" programID="0" offset="0">
        <phase duration="30" state="GGGGrrrrGGGGrrrr"/>
        <phase duration="3" state="yyyyrrrryyyyrrrr"/>
        <phase duration="30" state="rrrrGGGGrrrrGGGG"/>
        <phase duration="3" state="rrrryyyyrrrryyyy"/>
    </tlLogic>
    
    <tlLogic id="marathahalli" type="static" programID="0" offset="0">
        <phase duration="30" state="GGGGrrrrGGGGrrrr"/>
        <phase duration="3" state="yyyyrrrryyyyrrrr"/>
        <phase duration="30" state="rrrrGGGGrrrrGGGG"/>
        <phase duration="3" state="rrrryyyyrrrryyyy"/>
    </tlLogic>

</net>
"""
    
    with open(mock_net, 'w') as f:
        f.write(mock_xml)
    
    print(f"Mock network created: {mock_net}")
    print("⚠️  This is a placeholder. Use real OSM data and SUMO for production.")


if __name__ == "__main__":
    main()
