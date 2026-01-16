"""
OSM Data Acquisition Script
Downloads real OpenStreetMap data for Bangalore intersections
"""

import os
import sys
import requests
import time
from pathlib import Path
from typing import List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, get_project_root
from src.utils.logger import setup_logger

logger = setup_logger("osm_download")


class OSMDownloader:
    """
    Download OpenStreetMap data using Overpass API
    """
    
    # Overpass API endpoints (with fallbacks)
    OVERPASS_ENDPOINTS = [
        "https://overpass-api.de/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://z.overpass-api.de/api/interpreter",
    ]
    
    def __init__(self, output_dir: str = None):
        """
        Initialize OSM Downloader
        
        Args:
            output_dir: Directory to save downloaded OSM files
        """
        if output_dir is None:
            output_dir = get_project_root() / "data" / "osm"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load junction configuration
        try:
            self.junctions_config = load_config('junctions')
        except FileNotFoundError:
            logger.warning("Junctions config not found, using defaults")
            self.junctions_config = self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Get default configuration for Bangalore junctions"""
        return {
            'osm': {
                'bounding_box': [77.55, 12.88, 77.72, 13.06],
                'junction_boxes': {
                    'silk_board': {
                        'bbox': [77.615, 12.912, 77.630, 12.925],
                        'center': [77.6221, 12.9173]
                    },
                    'tin_factory': {
                        'bbox': [77.635, 12.965, 77.650, 12.980],
                        'center': [77.6412, 12.9716]
                    },
                    'hebbal': {
                        'bbox': [77.590, 13.030, 77.610, 13.045],
                        'center': [77.5972, 13.0358]
                    },
                    'marathahalli': {
                        'bbox': [77.695, 12.952, 77.710, 12.968],
                        'center': [77.7011, 12.9591]
                    }
                }
            }
        }
    
    def _build_overpass_query(
        self,
        bbox: List[float],
        include_roads: bool = True,
        include_traffic_signals: bool = True
    ) -> str:
        """
        Build Overpass QL query for road network
        
        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]
            include_roads: Include road ways
            include_traffic_signals: Include traffic signal nodes
        
        Returns:
            Overpass QL query string
        """
        # Overpass uses [south, west, north, east] format
        south, west, north, east = bbox[1], bbox[0], bbox[3], bbox[2]
        bbox_str = f"{south},{west},{north},{east}"
        
        query_parts = ["[out:xml][timeout:300];", "("]
        
        if include_roads:
            # Include major road types for traffic simulation
            road_types = [
                "motorway", "trunk", "primary", "secondary", "tertiary",
                "motorway_link", "trunk_link", "primary_link", "secondary_link",
                "unclassified", "residential"
            ]
            
            for road_type in road_types:
                query_parts.append(f'  way["highway"="{road_type}"]({bbox_str});')
        
        if include_traffic_signals:
            # Include traffic signal nodes
            query_parts.append(f'  node["highway"="traffic_signals"]({bbox_str});')
        
        query_parts.append(");")
        query_parts.append("(._;>;);")  # Recurse down to get all nodes
        query_parts.append("out body;")
        
        return "\n".join(query_parts)
    
    def _download_from_overpass(
        self,
        query: str,
        output_file: str,
        max_retries: int = 3
    ) -> bool:
        """
        Download data from Overpass API
        
        Args:
            query: Overpass QL query
            output_file: Path to save the data
            max_retries: Maximum number of retry attempts
        
        Returns:
            True if successful, False otherwise
        """
        for endpoint in self.OVERPASS_ENDPOINTS:
            for attempt in range(max_retries):
                try:
                    logger.info(f"Downloading from {endpoint} (attempt {attempt + 1}/{max_retries})...")
                    
                    response = requests.post(
                        endpoint,
                        data={"data": query},
                        timeout=600,  # 10 minute timeout for large queries
                        headers={"User-Agent": "BangaloreTrafficRL/1.0"}
                    )
                    
                    if response.status_code == 200:
                        # Check if response is valid XML
                        if response.text.startswith("<?xml"):
                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(response.text)
                            
                            file_size = os.path.getsize(output_file) / 1024
                            logger.info(f"Successfully downloaded {file_size:.1f} KB to {output_file}")
                            return True
                        else:
                            logger.warning(f"Invalid response (not XML): {response.text[:200]}")
                    
                    elif response.status_code == 429:
                        # Rate limited - wait and retry
                        wait_time = 30 * (attempt + 1)
                        logger.warning(f"Rate limited. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    
                    elif response.status_code == 504:
                        # Gateway timeout - try smaller area or different endpoint
                        logger.warning("Gateway timeout. Trying different endpoint...")
                        break
                    
                    else:
                        logger.warning(f"HTTP {response.status_code}: {response.text[:200]}")
                    
                except requests.Timeout:
                    logger.warning(f"Request timeout on attempt {attempt + 1}")
                    time.sleep(10)
                
                except requests.RequestException as e:
                    logger.error(f"Request failed: {e}")
                    time.sleep(10)
        
        return False
    
    def download_full_area(self) -> Optional[str]:
        """
        Download OSM data for the full Bangalore study area
        
        Returns:
            Path to downloaded file or None if failed
        """
        osm_config = self.junctions_config.get('osm', {})
        bbox = osm_config.get('bounding_box', [77.55, 12.88, 77.72, 13.06])
        
        output_file = self.output_dir / "bangalore_full.osm"
        
        logger.info(f"Downloading full Bangalore area: {bbox}")
        
        query = self._build_overpass_query(bbox)
        
        if self._download_from_overpass(query, str(output_file)):
            return str(output_file)
        
        return None
    
    def download_junction(self, junction_id: str) -> Optional[str]:
        """
        Download OSM data for a specific junction
        
        Args:
            junction_id: Junction identifier (e.g., 'silk_board')
        
        Returns:
            Path to downloaded file or None if failed
        """
        osm_config = self.junctions_config.get('osm', {})
        junction_boxes = osm_config.get('junction_boxes', {})
        
        if junction_id not in junction_boxes:
            logger.error(f"Unknown junction: {junction_id}")
            return None
        
        bbox = junction_boxes[junction_id]['bbox']
        output_file = self.output_dir / f"{junction_id}.osm"
        
        logger.info(f"Downloading junction '{junction_id}': {bbox}")
        
        query = self._build_overpass_query(bbox)
        
        if self._download_from_overpass(query, str(output_file)):
            return str(output_file)
        
        return None
    
    def download_all_junctions(self) -> dict:
        """
        Download OSM data for all target junctions
        
        Returns:
            Dictionary mapping junction_id to file path
        """
        osm_config = self.junctions_config.get('osm', {})
        junction_boxes = osm_config.get('junction_boxes', {})
        
        results = {}
        
        for junction_id in junction_boxes.keys():
            file_path = self.download_junction(junction_id)
            results[junction_id] = file_path
            
            # Rate limiting between downloads
            time.sleep(5)
        
        return results
    
    def validate_osm_file(self, file_path: str) -> Tuple[bool, dict]:
        """
        Validate downloaded OSM file
        
        Args:
            file_path: Path to OSM file
        
        Returns:
            Tuple of (is_valid, stats_dict)
        """
        import xml.etree.ElementTree as ET
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Count elements
            nodes = len(root.findall('.//node'))
            ways = len(root.findall('.//way'))
            relations = len(root.findall('.//relation'))
            
            # Count traffic signals
            traffic_signals = len(root.findall('.//node[@highway="traffic_signals"]'))
            
            # Count road types
            highways = root.findall('.//way/tag[@k="highway"]')
            highway_types = {}
            for hw in highways:
                hw_type = hw.get('v', 'unknown')
                highway_types[hw_type] = highway_types.get(hw_type, 0) + 1
            
            stats = {
                'nodes': nodes,
                'ways': ways,
                'relations': relations,
                'traffic_signals': traffic_signals,
                'highway_types': highway_types
            }
            
            is_valid = nodes > 0 and ways > 0
            
            if is_valid:
                logger.info(f"Valid OSM file: {nodes} nodes, {ways} ways, {traffic_signals} traffic signals")
            else:
                logger.warning(f"Invalid OSM file: insufficient data")
            
            return is_valid, stats
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse OSM file: {e}")
            return False, {}
        except FileNotFoundError:
            logger.error(f"OSM file not found: {file_path}")
            return False, {}


def main():
    """Main function to download OSM data"""
    logger.info("=" * 60)
    logger.info("BANGALORE TRAFFIC RL - OSM DATA ACQUISITION")
    logger.info("=" * 60)
    
    downloader = OSMDownloader()
    
    # Download individual junctions first (smaller, more reliable)
    logger.info("\nStep 1: Downloading individual junctions...")
    junction_results = downloader.download_all_junctions()
    
    successful = sum(1 for v in junction_results.values() if v is not None)
    logger.info(f"\nDownloaded {successful}/{len(junction_results)} junctions successfully")
    
    # Validate downloaded files
    logger.info("\nStep 2: Validating downloaded files...")
    for junction_id, file_path in junction_results.items():
        if file_path:
            is_valid, stats = downloader.validate_osm_file(file_path)
            if is_valid:
                logger.info(f"  ✓ {junction_id}: {stats['nodes']} nodes, {stats['ways']} ways")
            else:
                logger.warning(f"  ✗ {junction_id}: Validation failed")
    
    # Optionally download full area
    logger.info("\nStep 3: Downloading full study area...")
    full_area_file = downloader.download_full_area()
    
    if full_area_file:
        is_valid, stats = downloader.validate_osm_file(full_area_file)
        if is_valid:
            logger.info(f"✓ Full area download successful!")
            logger.info(f"  File: {full_area_file}")
            logger.info(f"  Nodes: {stats['nodes']}")
            logger.info(f"  Ways: {stats['ways']}")
            logger.info(f"  Traffic signals: {stats['traffic_signals']}")
    else:
        logger.warning("Full area download failed. Using individual junction files.")
    
    logger.info("\n" + "=" * 60)
    logger.info("OSM DATA ACQUISITION COMPLETE")
    logger.info("=" * 60)
    
    return junction_results


if __name__ == "__main__":
    main()
