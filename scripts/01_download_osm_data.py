#!/usr/bin/env python3
"""
OSM Data Download Script for Bangalore Junctions
Downloads real road network data from OpenStreetMap for the 4 target intersections.

Target Junctions:
- Silk Board Junction (12.9173° N, 77.6228° E)
- Tin Factory Junction (12.9988° N, 77.6515° E)  
- Hebbal Junction (13.0358° N, 77.5970° E)
- Marathahalli Junction (12.9591° N, 77.7011° E)
"""

import os
import sys
import requests
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BoundingBox:
    """Geographic bounding box for OSM download"""
    south: float  # min latitude
    west: float   # min longitude
    north: float  # max latitude
    east: float   # max longitude
    
    def to_overpass_bbox(self) -> str:
        """Format for Overpass API: south,west,north,east"""
        return f"{self.south},{self.west},{self.north},{self.east}"
    
    def to_osm_bbox(self) -> str:
        """Format for OSM API: left,bottom,right,top (west,south,east,north)"""
        return f"{self.west},{self.south},{self.east},{self.north}"


# ============================================================================
# BANGALORE JUNCTION COORDINATES (WGS84)
# ============================================================================

BANGALORE_JUNCTIONS = {
    'silk_board': {
        'name': 'Silk Board Junction',
        'lat': 12.9173,
        'lon': 77.6228,
        'description': 'One of India\'s busiest junctions - ORR meets Hosur Road',
        'peak_traffic': 'extreme',
        'directions': {
            'north': 'Electronics City',
            'south': 'Hosur',
            'east': 'BTM Layout/Koramangala',
            'west': 'Bannerghatta Road'
        }
    },
    'tin_factory': {
        'name': 'Tin Factory Junction',
        'lat': 12.9988,
        'lon': 77.6515,
        'description': 'Major junction connecting Old Madras Road with ORR',
        'peak_traffic': 'high',
        'directions': {
            'north': 'KR Puram',
            'south': 'Indiranagar',
            'east': 'Whitefield',
            'west': 'MG Road'
        }
    },
    'hebbal': {
        'name': 'Hebbal Junction',
        'lat': 13.0358,
        'lon': 77.5970,
        'description': 'Gateway to North Bangalore - NH44 intersection',
        'peak_traffic': 'high',
        'directions': {
            'north': 'Airport/Bellary Road',
            'south': 'City Center',
            'east': 'Manyata Tech Park',
            'west': 'Yeshwanthpur'
        }
    },
    'marathahalli': {
        'name': 'Marathahalli Junction',
        'lat': 12.9591,
        'lon': 77.7011,
        'description': 'IT corridor junction - Heavy tech office traffic',
        'peak_traffic': 'high',
        'directions': {
            'north': 'Whitefield',
            'south': 'Sarjapur Road',
            'east': 'ITPL',
            'west': 'Koramangala/Silk Board'
        }
    }
}


def get_bounding_box(center_lat: float, center_lon: float, radius_km: float = 0.8) -> BoundingBox:
    """
    Create a bounding box around a center point.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_km: Radius in kilometers (default 0.8km = 800m)
    
    Returns:
        BoundingBox for the area
    """
    # Approximate degrees per km at this latitude
    # 1 degree latitude ≈ 111 km
    # 1 degree longitude ≈ 111 * cos(lat) km
    import math
    
    lat_offset = radius_km / 111.0
    lon_offset = radius_km / (111.0 * math.cos(math.radians(center_lat)))
    
    return BoundingBox(
        south=center_lat - lat_offset,
        west=center_lon - lon_offset,
        north=center_lat + lat_offset,
        east=center_lon + lon_offset
    )


def get_combined_bounding_box() -> BoundingBox:
    """
    Get a bounding box that covers all 4 junctions.
    This is for downloading one large map.
    """
    all_lats = [j['lat'] for j in BANGALORE_JUNCTIONS.values()]
    all_lons = [j['lon'] for j in BANGALORE_JUNCTIONS.values()]
    
    # Add padding (2km)
    padding = 0.02  # ~2km
    
    return BoundingBox(
        south=min(all_lats) - padding,
        west=min(all_lons) - padding,
        north=max(all_lats) + padding,
        east=max(all_lons) + padding
    )


def download_osm_direct(bbox: BoundingBox, output_file: Path) -> bool:
    """
    Download OSM data directly from OSM API.
    Best for small areas (< 0.5 degree box).
    
    Args:
        bbox: Bounding box to download
        output_file: Where to save the .osm file
    
    Returns:
        True if successful
    """
    url = f"https://api.openstreetmap.org/api/0.6/map?bbox={bbox.to_osm_bbox()}"
    
    print(f"Downloading from OSM API...")
    print(f"  URL: {url}")
    print(f"  Bounding Box: {bbox}")
    
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        size_mb = len(response.content) / (1024 * 1024)
        print(f"  Downloaded: {size_mb:.2f} MB")
        print(f"  Saved to: {output_file}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"  Error: {e}")
        return False


def download_osm_overpass(bbox: BoundingBox, output_file: Path) -> bool:
    """
    Download OSM data using Overpass API.
    Better for larger areas and custom queries.
    
    Args:
        bbox: Bounding box to download
        output_file: Where to save the .osm file
    
    Returns:
        True if successful
    """
    # Overpass query for road network (highways, traffic signals, etc.)
    overpass_query = f"""
    [out:xml][timeout:300];
    (
      // All highways (roads)
      way["highway"]({bbox.to_overpass_bbox()});
      
      // Traffic signals
      node["highway"="traffic_signals"]({bbox.to_overpass_bbox()});
      
      // Junction nodes
      node["highway"="junction"]({bbox.to_overpass_bbox()});
      
      // Crossings
      node["highway"="crossing"]({bbox.to_overpass_bbox()});
    );
    out body;
    >;
    out skel qt;
    """
    
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    print(f"Downloading from Overpass API...")
    print(f"  Query area: {bbox}")
    
    try:
        response = requests.post(
            overpass_url,
            data={'data': overpass_query},
            timeout=300
        )
        response.raise_for_status()
        
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        size_mb = len(response.content) / (1024 * 1024)
        print(f"  Downloaded: {size_mb:.2f} MB")
        print(f"  Saved to: {output_file}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"  Error: {e}")
        return False


def download_individual_junctions(data_dir: Path) -> dict:
    """
    Download OSM data for each junction separately.
    Useful for focused analysis of each intersection.
    
    Returns:
        Dict mapping junction name to file path
    """
    results = {}
    
    for junction_id, info in BANGALORE_JUNCTIONS.items():
        print(f"\n{'='*60}")
        print(f"Downloading: {info['name']}")
        print(f"  Location: {info['lat']}, {info['lon']}")
        print(f"  Description: {info['description']}")
        
        bbox = get_bounding_box(info['lat'], info['lon'], radius_km=0.5)
        output_file = data_dir / f"{junction_id}.osm"
        
        success = download_osm_overpass(bbox, output_file)
        
        if success:
            results[junction_id] = output_file
        
        # Be nice to the API
        time.sleep(2)
    
    return results


def download_combined_area(data_dir: Path) -> Optional[Path]:
    """
    Download the entire area covering all 4 junctions.
    This creates one large network file.
    
    Returns:
        Path to downloaded file, or None if failed
    """
    print("\n" + "="*60)
    print("Downloading Combined Bangalore Network")
    print("="*60)
    
    bbox = get_combined_bounding_box()
    output_file = data_dir / "bangalore_combined.osm"
    
    print(f"Coverage area:")
    print(f"  South: {bbox.south:.4f}°")
    print(f"  North: {bbox.north:.4f}°")
    print(f"  West: {bbox.west:.4f}°")
    print(f"  East: {bbox.east:.4f}°")
    
    success = download_osm_overpass(bbox, output_file)
    
    return output_file if success else None


def verify_osm_file(osm_file: Path) -> dict:
    """
    Parse OSM file and extract basic statistics.
    """
    import xml.etree.ElementTree as ET
    
    try:
        tree = ET.parse(osm_file)
        root = tree.getroot()
        
        stats = {
            'nodes': len(root.findall('.//node')),
            'ways': len(root.findall('.//way')),
            'relations': len(root.findall('.//relation')),
            'traffic_signals': 0,
            'highways': {}
        }
        
        # Count traffic signals
        for node in root.findall('.//node'):
            for tag in node.findall('tag'):
                if tag.get('k') == 'highway' and tag.get('v') == 'traffic_signals':
                    stats['traffic_signals'] += 1
        
        # Count highway types
        for way in root.findall('.//way'):
            for tag in way.findall('tag'):
                if tag.get('k') == 'highway':
                    hw_type = tag.get('v')
                    stats['highways'][hw_type] = stats['highways'].get(hw_type, 0) + 1
        
        return stats
        
    except ET.ParseError as e:
        return {'error': str(e)}


def print_junction_info():
    """Print detailed information about target junctions."""
    print("\n" + "="*70)
    print("BANGALORE TARGET JUNCTIONS")
    print("="*70)
    
    for junction_id, info in BANGALORE_JUNCTIONS.items():
        print(f"\n📍 {info['name']} ({junction_id})")
        print(f"   Coordinates: {info['lat']}°N, {info['lon']}°E")
        print(f"   Peak Traffic: {info['peak_traffic'].upper()}")
        print(f"   Description: {info['description']}")
        print(f"   Directions:")
        for direction, destination in info['directions'].items():
            arrow = {'north': '↑', 'south': '↓', 'east': '→', 'west': '←'}[direction]
            print(f"     {arrow} {direction.capitalize()}: {destination}")


def main():
    """Main download workflow."""
    print("="*70)
    print("BANGALORE TRAFFIC NETWORK - OSM DATA DOWNLOAD")
    print("="*70)
    
    # Create data directory
    data_dir = PROJECT_ROOT / "data" / "osm"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Print junction info
    print_junction_info()
    
    # Download options
    print("\n" + "="*70)
    print("DOWNLOAD OPTIONS")
    print("="*70)
    print("1. Download individual junction areas (4 files, ~0.5km radius each)")
    print("2. Download combined area (1 file covering all junctions)")
    print("3. Download both")
    
    choice = input("\nSelect option [1/2/3]: ").strip() or "3"
    
    results = {'individual': {}, 'combined': None}
    
    if choice in ['1', '3']:
        print("\n" + "-"*50)
        print("Downloading Individual Junction Areas...")
        results['individual'] = download_individual_junctions(data_dir)
    
    if choice in ['2', '3']:
        print("\n" + "-"*50)
        print("Downloading Combined Area...")
        results['combined'] = download_combined_area(data_dir)
    
    # Verify downloads
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    for junction_id, osm_file in results['individual'].items():
        if osm_file and osm_file.exists():
            stats = verify_osm_file(osm_file)
            print(f"\n{junction_id}:")
            print(f"  Nodes: {stats.get('nodes', 0)}")
            print(f"  Ways: {stats.get('ways', 0)}")
            print(f"  Traffic Signals: {stats.get('traffic_signals', 0)}")
    
    if results['combined'] and results['combined'].exists():
        stats = verify_osm_file(results['combined'])
        print(f"\nCombined Network:")
        print(f"  Nodes: {stats.get('nodes', 0)}")
        print(f"  Ways: {stats.get('ways', 0)}")
        print(f"  Traffic Signals: {stats.get('traffic_signals', 0)}")
        if stats.get('highways'):
            print(f"  Highway Types: {dict(sorted(stats['highways'].items(), key=lambda x: -x[1])[:5])}")
    
    # Next steps
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Run: python scripts/02_generate_sumo_network.py")
    print("   This converts OSM to SUMO network format")
    print("")
    print("2. Or use SUMO's osmWebWizard for interactive conversion:")
    print("   python $SUMO_HOME/tools/osmWebWizard.py")
    
    return results


if __name__ == "__main__":
    main()
