"""
Generate grid networks for testing
"""

import subprocess
import os
import random
import sumolib

def create_grid_network(rows: int, cols: int, output_dir: str = 'sumo/networks'):
    """
    Create a grid network using SUMO netgenerate
    """
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir.replace('networks', 'routes'), exist_ok=True)
    
    output_file = f'{output_dir}/grid_{rows}x{cols}.net.xml'
    
    cmd = [
        'netgenerate',
        '--grid',
        '--grid.x-number', str(cols),
        '--grid.y-number', str(rows),
        '--grid.length', '200',
        '--default.lanenumber', '2',
        '--default.speed', '13.89',
        '--tls.guess', 'true',
        '--tls.yellow.time', '3',
        '--tls.allred.time', '2',
        '--no-internal-links', 'false',
        '--junctions.join', 'false',
        '--output-file', output_file
    ]
    
    print(f"🏗️  Generating {rows}x{cols} grid network...")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Created: {output_file}")
        
        # Generate valid routes using sumolib
        generate_grid_routes(output_file, rows, cols)
        
        return output_file
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        return None

def generate_grid_routes(net_file: str, rows: int, cols: int):
    """Generate random routes for grid using valid edges"""
    
    # Read network to get valid edges
    print(f"   Parsing network to find valid routes...")
    try:
        net = sumolib.net.readNet(net_file)
        valid_edges = [e.getID() for e in net.getEdges() if e.allows("passenger")]
    except Exception as e:
        print(f"   ⚠️ Failed to read network with sumolib ({e}). Using fallback edges.")
        # Fallback if sumolib fails (though it shouldn't)
        valid_edges = []
    
    if not valid_edges:
        print("   ❌ No valid edges found!")
        return

    routes_dir = os.path.dirname(net_file).replace('networks', 'routes')
    route_file = f'{routes_dir}/grid_{rows}x{cols}.rou.xml'
    
    num_cars = rows * cols * 15  # Scale traffic: 12*15 = 180 vehicles for 3x4 (Medium)
    num_emergency = max(2, int(num_cars * 0.05))
    
    print(f"   Generating {num_cars} vehicle trips...")
    
    with open(route_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes>\n')
        f.write('    <vType id="car" accel="2.5" decel="4.5" sigma="0.5" length="4.5" maxSpeed="13.89"/>\n')
        f.write('    <vType id="emergency" accel="3.0" decel="5.0" length="5.0" maxSpeed="16.67" vClass="emergency" color="1,0,0" gap="2.5"/>\n\n')
        
        # Generate random trips
        for i in range(num_cars):
            origin = random.choice(valid_edges)
            dest = random.choice(valid_edges)
            while dest == origin:
                dest = random.choice(valid_edges)
                
            depart = i * 2.0 + random.uniform(0, 5) # Staggered departure
            
            f.write(f'    <trip id="car_{i}" type="car" depart="{depart:.1f}" from="{origin}" to="{dest}" />\n')
            
        # Emergency vehicles
        for i in range(num_emergency):
            origin = random.choice(valid_edges)
            dest = random.choice(valid_edges)
            depart = 50 + i * 100 # Periodic emergency
            
            f.write(f'    <trip id="emergency_{i}" type="emergency" depart="{depart:.1f}" from="{origin}" to="{dest}" />\n')
            
        f.write('</routes>\n')
    
    print(f"✅ Created routes: {route_file}")

if __name__ == '__main__':
    # Create 3x4 grid (12 intersections)
    create_grid_network(3, 4)
    
    # Create 5x5 grid (25 intersections)
    create_grid_network(5, 5)
