"""
Main entry point for heuristic traffic control system
"""

import argparse
import os
from run_single_intersection import run_single_intersection
from create_grid_network import create_grid_network

def main():
    parser = argparse.ArgumentParser(
        description='Heuristic Traffic Control System'
    )
    
    parser.add_argument('--network', type=str,
                       default='sumo/networks/simple_intersection.net.xml',
                       help='Network file path')
    parser.add_argument('--routes', type=str,
                       default='sumo/routes/simple_intersection.rou.xml',
                       help='Routes file path')
    parser.add_argument('--duration', type=float, default=900.0,
                       help='Simulation duration (seconds)')
    parser.add_argument('--gui', action='store_true',
                       help='Show SUMO GUI')
    parser.add_argument('--create-grid', type=str,
                       help='Create grid network (e.g., 3x4)')
    
    args = parser.parse_args()
    
    # Create grid if requested
    if args.create_grid:
        try:
            rows, cols = map(int, args.create_grid.split('x'))
            network_file = create_grid_network(rows, cols)
            if network_file:
                args.network = network_file
                args.routes = network_file.replace('networks', 'routes').replace('.net.xml', '.rou.xml')
        except ValueError:
            print("❌ Invalid grid format. Use: --create-grid 3x4")
            return
    
    # Run simulation
    run_single_intersection(
        network_file=args.network,
        route_file=args.routes,
        duration=args.duration,
        gui=args.gui
    )

if __name__ == '__main__':
    main()
