"""
SUMO Network Conversion Script
Converts OSM data to SUMO network files with proper traffic light configurations
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List
import xml.etree.ElementTree as ET

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, get_project_root
from src.utils.logger import setup_logger
from src.utils.sumo_utils import get_netconvert_binary, check_sumo_installation

logger = setup_logger("sumo_convert")


class SUMONetworkConverter:
    """
    Convert OSM files to SUMO network format with traffic light configurations
    """
    
    def __init__(self, input_dir: str = None, output_dir: str = None):
        """
        Initialize network converter
        
        Args:
            input_dir: Directory containing OSM files
            output_dir: Directory for SUMO network files
        """
        project_root = get_project_root()
        
        self.input_dir = Path(input_dir) if input_dir else project_root / "data" / "osm"
        self.output_dir = Path(output_dir) if output_dir else project_root / "maps"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify SUMO installation
        success, message = check_sumo_installation()
        if not success:
            raise EnvironmentError(message)
        
        self.netconvert = get_netconvert_binary()
        logger.info(f"Using netconvert: {self.netconvert}")
    
    def convert_osm_to_net(
        self,
        osm_file: str,
        output_file: str,
        lefthand: bool = True,
        guess_signals: bool = True,
        join_tls: bool = True
    ) -> bool:
        """
        Convert single OSM file to SUMO network
        
        Args:
            osm_file: Path to OSM file
            output_file: Path for output .net.xml file
            lefthand: Enable left-hand traffic (India)
            guess_signals: Guess traffic signal locations
            join_tls: Join nearby traffic lights
        
        Returns:
            True if successful
        """
        cmd = [
            self.netconvert,
            "--osm-files", osm_file,
            "--output-file", output_file,
            
            # Geometry processing
            "--geometry.remove", "true",
            "--roundabouts.guess", "true",
            "--ramps.guess", "true",
            "--junctions.join", "true",
            "--junctions.corner-detail", "5",
            "--edges.join", "true",
            
            # Traffic light settings
            "--tls.guess-signals", str(guess_signals).lower(),
            "--tls.join", str(join_tls).lower(),
            "--tls.default-type", "actuated",
            "--tls.guess", "true",
            
            # India-specific settings
            "--lefthand", str(lefthand).lower(),
            
            # Coordinate projection (UTM zone 43N for Bangalore)
            "--proj", "+proj=utm +zone=43 +datum=WGS84",
            
            # Processing options
            "--no-turnarounds", "true",
            "--offset.disable-normalization", "true",
            
            # Output options
            "--output.street-names", "true",
            "--output.original-names", "true",
        ]
        
        logger.info(f"Converting {osm_file} to {output_file}...")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.stderr:
                # netconvert outputs warnings to stderr
                for line in result.stderr.split('\n'):
                    if 'Warning' in line:
                        logger.warning(line.strip())
                    elif 'Error' in line:
                        logger.error(line.strip())
            
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / 1024
                logger.info(f"Successfully created {output_file} ({file_size:.1f} KB)")
                return True
            else:
                logger.error(f"Output file not created: {output_file}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"netconvert failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return False
    
    def convert_all_junctions(self) -> dict:
        """
        Convert all OSM files in input directory to SUMO networks
        
        Returns:
            Dictionary mapping junction_id to network file path
        """
        results = {}
        
        for osm_file in self.input_dir.glob("*.osm"):
            junction_id = osm_file.stem
            output_file = self.output_dir / f"{junction_id}.net.xml"
            
            if self.convert_osm_to_net(str(osm_file), str(output_file)):
                results[junction_id] = str(output_file)
            else:
                results[junction_id] = None
        
        return results
    
    def add_traffic_light_programs(
        self,
        net_file: str,
        junction_id: str
    ) -> bool:
        """
        Add custom traffic light programs to network file
        
        Args:
            net_file: Path to network file
            junction_id: Junction identifier for TLS configuration
        
        Returns:
            True if successful
        """
        try:
            tree = ET.parse(net_file)
            root = tree.getroot()
            
            # Find all traffic light logics
            tls_elements = root.findall('.//tlLogic')
            
            if not tls_elements:
                logger.warning(f"No traffic lights found in {net_file}")
                return False
            
            # Load junction-specific configuration
            junctions_config = load_config('junctions')
            junction_config = junctions_config.get(junction_id, {})
            phases_config = junction_config.get('phases', [])
            
            # Update each traffic light
            for tls in tls_elements:
                tls_id = tls.get('id')
                logger.info(f"Configuring TLS: {tls_id}")
                
                # Clear existing phases
                for phase in list(tls.findall('phase')):
                    tls.remove(phase)
                
                # Add configured phases or default phases
                if phases_config:
                    for phase_info in phases_config:
                        phase_elem = ET.SubElement(tls, 'phase')
                        phase_elem.set('duration', str(phase_info.get('duration', 30)))
                        
                        # Generate state string based on phase type
                        state = self._generate_phase_state(
                            phase_info.get('name', 'default'),
                            len(tls.get('state', 'GGGG'))
                        )
                        phase_elem.set('state', state)
                else:
                    # Default 4-phase cycle
                    self._add_default_phases(tls)
            
            # Write updated network
            tree.write(net_file, encoding='utf-8', xml_declaration=True)
            logger.info(f"Updated traffic lights in {net_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update traffic lights: {e}")
            return False
    
    def _generate_phase_state(self, phase_name: str, num_lanes: int) -> str:
        """Generate signal state string for a phase"""
        # Standard 4-approach intersection state patterns
        patterns = {
            'NS_Through': 'GGGGrrrrGGGGrrrr',
            'NS_Yellow': 'yyyyrrrryyyyrrrr',
            'EW_Through': 'rrrrGGGGrrrrGGGG',
            'EW_Yellow': 'rrrryyyyrrrryyyy',
            'Left_Turns': 'rrrrrrrrGGGGGGGG',
            'Left_Yellow': 'rrrrrrrryyyyyyyy',
            'All_Red': 'rrrrrrrrrrrrrrrr',
        }
        
        state = patterns.get(phase_name)
        
        if state is None:
            # Generate based on phase type
            if 'green' in phase_name.lower() or 'through' in phase_name.lower():
                state = 'G' * num_lanes
            elif 'yellow' in phase_name.lower():
                state = 'y' * num_lanes
            else:
                state = 'r' * num_lanes
        
        # Adjust to actual number of lanes
        if len(state) < num_lanes:
            state = state + 'r' * (num_lanes - len(state))
        elif len(state) > num_lanes:
            state = state[:num_lanes]
        
        return state
    
    def _add_default_phases(self, tls_element):
        """Add default 4-phase cycle to traffic light"""
        # Determine number of signals from existing state
        current_state = tls_element.get('state', 'GGGGrrrr')
        num_signals = len(current_state)
        half = num_signals // 2
        
        phases = [
            # Phase 0: NS Green (60s)
            {'duration': 60, 'state': 'G' * half + 'r' * half},
            # Phase 1: NS Yellow (3s)
            {'duration': 3, 'state': 'y' * half + 'r' * half},
            # Phase 2: EW Green (60s)
            {'duration': 60, 'state': 'r' * half + 'G' * half},
            # Phase 3: EW Yellow (3s)
            {'duration': 3, 'state': 'r' * half + 'y' * half},
        ]
        
        for phase in phases:
            phase_elem = ET.SubElement(tls_element, 'phase')
            phase_elem.set('duration', str(phase['duration']))
            phase_elem.set('state', phase['state'][:num_signals])
    
    def validate_network(self, net_file: str) -> dict:
        """
        Validate SUMO network file
        
        Args:
            net_file: Path to network file
        
        Returns:
            Dictionary with validation results
        """
        try:
            tree = ET.parse(net_file)
            root = tree.getroot()
            
            # Count elements
            edges = len(root.findall('.//edge'))
            junctions = len(root.findall('.//junction'))
            connections = len(root.findall('.//connection'))
            tls = len(root.findall('.//tlLogic'))
            
            # Get junction types
            junction_types = {}
            for junction in root.findall('.//junction'):
                jtype = junction.get('type', 'unknown')
                junction_types[jtype] = junction_types.get(jtype, 0) + 1
            
            stats = {
                'edges': edges,
                'junctions': junctions,
                'connections': connections,
                'traffic_lights': tls,
                'junction_types': junction_types,
                'is_valid': edges > 0 and junctions > 0
            }
            
            logger.info(f"Network validation: {edges} edges, {junctions} junctions, {tls} TLS")
            
            return stats
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {'is_valid': False, 'error': str(e)}
    
    def create_combined_network(
        self,
        junction_files: List[str],
        output_file: str
    ) -> bool:
        """
        Combine multiple junction networks into one (if needed)
        
        Note: For large area, it's better to convert the full OSM file directly
        """
        # For now, we'll use the full area file
        # This method can be expanded for custom network merging
        logger.info("For multi-junction simulation, use the full area network file")
        return False


def main():
    """Main function to convert OSM to SUMO networks"""
    logger.info("=" * 60)
    logger.info("BANGALORE TRAFFIC RL - SUMO NETWORK CONVERSION")
    logger.info("=" * 60)
    
    converter = SUMONetworkConverter()
    
    # Step 1: Convert all junction OSM files
    logger.info("\nStep 1: Converting OSM files to SUMO networks...")
    results = converter.convert_all_junctions()
    
    successful = sum(1 for v in results.values() if v is not None)
    logger.info(f"Converted {successful}/{len(results)} files successfully")
    
    # Step 2: Configure traffic lights
    logger.info("\nStep 2: Configuring traffic light programs...")
    for junction_id, net_file in results.items():
        if net_file:
            converter.add_traffic_light_programs(net_file, junction_id)
    
    # Step 3: Validate networks
    logger.info("\nStep 3: Validating networks...")
    for junction_id, net_file in results.items():
        if net_file:
            stats = converter.validate_network(net_file)
            if stats.get('is_valid'):
                logger.info(f"  ✓ {junction_id}: {stats['edges']} edges, {stats['traffic_lights']} TLS")
            else:
                logger.warning(f"  ✗ {junction_id}: Validation failed")
    
    # Step 4: Convert full area if available
    full_osm = Path(get_project_root()) / "data" / "osm" / "bangalore_full.osm"
    if full_osm.exists():
        logger.info("\nStep 4: Converting full area network...")
        full_net = Path(get_project_root()) / "maps" / "bangalore_full.net.xml"
        
        if converter.convert_osm_to_net(str(full_osm), str(full_net)):
            stats = converter.validate_network(str(full_net))
            logger.info(f"  Full network: {stats.get('edges', 0)} edges, {stats.get('traffic_lights', 0)} TLS")
    
    logger.info("\n" + "=" * 60)
    logger.info("SUMO NETWORK CONVERSION COMPLETE")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
