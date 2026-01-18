"""
Generate grid networks for testing
"""

import subprocess
import os

def create_grid_network(rows: int, cols: int, output_dir: str = 'sumo/networks'):
    """
    Create a grid network using SUMO netgenerate
    
    Args:
        rows: Number of rows
        cols: Number of columns
        output_dir: Output directory
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
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
        '--output-file', output_file
    ]
    
    print(f"🏗️  Generating {rows}x{cols} grid network...")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Created: {output_file}")
        print(f"   Intersections: {rows * cols}")
        
        # Generate simple routes
        generate_grid_routes(rows, cols, output_dir)
        
        return output_file
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        return None

def generate_grid_routes(rows: int, cols: int, output_dir: str):
    """Generate random routes for grid"""
    
    routes_dir = output_dir.replace('networks', 'routes')
    os.makedirs(routes_dir, exist_ok=True)
    
    route_file = f'{routes_dir}/grid_{rows}x{cols}.rou.xml'
    
    with open(route_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes>\n')
        
        # Vehicle types
        f.write('    <vType id="car" accel="2.5" decel="4.5" sigma="0.5" length="4.5" maxSpeed="13.89"/>\n')
        f.write('    <vType id="emergency" accel="3.0" decel="5.0" length="5.0" maxSpeed="16.67" vClass="emergency" color="1,0,0"/>\n\n')
        
        # Random routes (simplified)
        for i in range(20):
            f.write(f'    <vehicle id="car_{i}" type="car" depart="{i*10}">\n')
            f.write(f'        <route edges="A0toA1 A1toA2"/>\n')
            f.write(f'    </vehicle>\n')
        
        # Emergency vehicles
        for i in range(5):
            f.write(f'    <vehicle id="emergency_{i}" type="emergency" depart="{120 + i*120}">\n')
            f.write(f'        <route edges="A0toA1 A1toB1"/>\n')
            f.write(f'    </vehicle>\n')
        
        f.write('</routes>\n')
    
    print(f"✅ Created: {route_file}")

if __name__ == '__main__':
    # Create 3x4 grid (12 intersections)
    create_grid_network(3, 4)
    
    # Create 5x5 grid (25 intersections)
    create_grid_network(5, 5)
