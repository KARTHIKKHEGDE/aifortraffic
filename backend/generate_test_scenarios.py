"""
Traffic Scenario Generator
Creates SUMO route files for testing Fixed vs Adaptive traffic control
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

class ScenarioGenerator:
    """Generate test scenarios to demonstrate adaptive vs fixed control"""
    
    def __init__(self, output_dir="backend/sumo/routes"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _prettify_xml(self, elem):
        """Return a pretty-printed XML string"""
        rough_string = ET.tostring(elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="    ")
    
    def generate_rush_hour_imbalance(self, filename="rush_hour_imbalance.rou.xml"):
        """
        Scenario 1: Rush Hour Imbalance (80/10/5/5)
        North gets 80% of traffic, others minimal
        """
        root = ET.Element("routes")
        
        # Vehicle types
        vtype = ET.SubElement(root, "vType", {
            "id": "car",
            "accel": "2.6",
            "decel": "4.5",
            "sigma": "0.5",
            "length": "5",
            "maxSpeed": "50"
        })
        
        # Total target: 900 vehicles/hour = 80% North, 10% South, 5% East, 5% West
        # North: 720 veh/hr
        ET.SubElement(root, "flow", {
            "id": "north_heavy",
            "type": "car",
            "from": "north_in",
            "to": "south_out",
            "begin": "0",
            "end": "600",
            "vehsPerHour": "720",
            "departLane": "best",
            "departSpeed": "max"
        })
        
        # South: 90 veh/hr
        ET.SubElement(root, "flow", {
            "id": "south_light",
            "type": "car",
            "from": "south_in",
            "to": "north_out",
            "begin": "0",
            "end": "600",
            "vehsPerHour": "90",
            "departLane": "best",
            "departSpeed": "max"
        })
        
        # East: 45 veh/hr
        ET.SubElement(root, "flow", {
            "id": "east_light",
            "type": "car",
            "from": "east_in",
            "to": "west_out",
            "begin": "0",
            "end": "600",
            "vehsPerHour": "45",
            "departLane": "best",
            "departSpeed": "max"
        })
        
        # West: 45 veh/hr
        ET.SubElement(root, "flow", {
            "id": "west_light",
            "type": "car",
            "from": "west_in",
            "to": "east_out",
            "begin": "0",
            "end": "600",
            "vehsPerHour": "45",
            "departLane": "best",
            "departSpeed": "max"
        })
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            f.write(self._prettify_xml(root))
        
        print(f"✅ Created: {output_path}")
        print(f"   📊 North: 720 veh/hr (80%), Others: 45-90 veh/hr")
        return output_path
    
    def generate_random_bursts(self, filename="random_bursts.rou.xml"):
        """
        Scenario 2: Random Traffic Bursts
        Sudden bursts of vehicles at different times
        """
        root = ET.Element("routes")
        
        vtype = ET.SubElement(root, "vType", {
            "id": "car",
            "accel": "2.6",
            "decel": "4.5",
            "sigma": "0.5",
            "length": "5",
            "maxSpeed": "50"
        })
        
        # Background low traffic
        for direction, edge_from, edge_to in [
            ("north", "north_in", "south_out"),
            ("south", "south_in", "north_out"),
            ("east", "east_in", "west_out"),
            ("west", "west_in", "east_out")
        ]:
            ET.SubElement(root, "flow", {
                "id": f"{direction}_background",
                "type": "car",
                "from": edge_from,
                "to": edge_to,
                "begin": "0",
                "end": "600",
                "vehsPerHour": "60",
                "departLane": "best",
                "departSpeed": "max"
            })
        
        # BURSTS
        bursts = [
            ("north", "north_in", "south_out", 0, 20),    # t=0s: 20 vehicles
            ("east", "east_in", "west_out", 30, 15),       # t=30s: 15 vehicles
            ("south", "south_in", "north_out", 60, 25),    # t=60s: 25 vehicles
            ("west", "west_in", "east_out", 90, 10),       # t=90s: 10 vehicles
            ("north", "north_in", "south_out", 120, 18),   # t=120s: 18 vehicles
            ("east", "east_in", "west_out", 150, 22),      # t=150s: 22 vehicles
        ]
        
        for i, (direction, edge_from, edge_to, start_time, num_vehicles) in enumerate(bursts):
            ET.SubElement(root, "flow", {
                "id": f"burst_{direction}_{i}",
                "type": "car",
                "from": edge_from,
                "to": edge_to,
                "begin": str(start_time),
                "end": str(start_time + 5),
                "number": str(num_vehicles),
                "departLane": "best",
                "departSpeed": "max"
            })
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            f.write(self._prettify_xml(root))
        
        print(f"✅ Created: {output_path}")
        print(f"   💥 6 traffic bursts: 10-25 vehicles each")
        return output_path
    
    def generate_empty_lane(self, filename="empty_lane.rou.xml"):
        """
        Scenario 3: One Lane Completely Empty
        South lane has ZERO traffic
        """
        root = ET.Element("routes")
        
        vtype = ET.SubElement(root, "vType", {
            "id": "car",
            "accel": "2.6",
            "decel": "4.5",
            "sigma": "0.5",
            "length": "5",
            "maxSpeed": "50"
        })
        
        # North, East, West: Normal traffic
        for direction, edge_from, edge_to in [
            ("north", "north_in", "south_out"),
            ("east", "east_in", "west_out"),
            ("west", "west_in", "east_out")
        ]:
            ET.SubElement(root, "flow", {
                "id": f"{direction}_normal",
                "type": "car",
                "from": edge_from,
                "to": edge_to,
                "begin": "0",
                "end": "600",
                "vehsPerHour": "300",
                "departLane": "best",
                "departSpeed": "max"
            })
        
        # South: ZERO traffic (no flow defined!)
        # This is intentional - no south flow = empty lane
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            f.write(self._prettify_xml(root))
        
        print(f"✅ Created: {output_path}")
        print(f"   🚫 South lane: ZERO traffic (25% waste on fixed timer!)")
        return output_path
    
    def generate_emergency_vehicle(self, filename="emergency_vehicle.rou.xml"):
        """
        Scenario 4: Emergency Vehicle
        Ambulance approaches from East
        """
        root = ET.Element("routes")
        
        # Normal car type
        vtype_car = ET.SubElement(root, "vType", {
            "id": "car",
            "accel": "2.6",
            "decel": "4.5",
            "sigma": "0.5",
            "length": "5",
            "maxSpeed": "50"
        })
        
        # Emergency vehicle type
        vtype_emergency = ET.SubElement(root, "vType", {
            "id": "emergency",
            "accel": "3.5",
            "decel": "6.0",
            "sigma": "0",
            "length": "7",
            "maxSpeed": "70",
            "color": "1,0,0",  # Red color
            "vClass": "emergency"
        })
        
        # Background traffic on all lanes
        for direction, edge_from, edge_to in [
            ("north", "north_in", "south_out"),
            ("south", "south_in", "north_out"),
            ("east", "east_in", "west_out"),
            ("west", "west_in", "east_out")
        ]:
            ET.SubElement(root, "flow", {
                "id": f"{direction}_background",
                "type": "car",
                "from": edge_from,
                "to": edge_to,
                "begin": "0",
                "end": "600",
                "vehsPerHour": "400",
                "departLane": "best",
                "departSpeed": "max"
            })
        
        # Ambulance from East at t=30s, 60s, 90s
        for i, depart_time in enumerate([30, 60, 90, 120, 150]):
            ET.SubElement(root, "vehicle", {
                "id": f"ambulance_{i}",
                "type": "emergency",
                "depart": str(depart_time),
                "from": "east_in",
                "to": "west_out",
                "departLane": "best",
                "departSpeed": "max"
            })
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            f.write(self._prettify_xml(root))
        
        print(f"✅ Created: {output_path}")
        print(f"   🚨 5 ambulances from East at t=30s, 60s, 90s, 120s, 150s")
        return output_path
    
    def generate_all_scenarios(self):
        """Generate all test scenarios"""
        print("\n" + "="*60)
        print("🎯 GENERATING ALL TEST SCENARIOS")
        print("="*60 + "\n")
        
        scenarios = []
        
        print("1️⃣  Rush Hour Imbalance (80/10/5/5)")
        scenarios.append(self.generate_rush_hour_imbalance())
        print()
        
        print("2️⃣  Random Traffic Bursts")
        scenarios.append(self.generate_random_bursts())
        print()
        
        print("3️⃣  Empty Lane Scenario")
        scenarios.append(self.generate_empty_lane())
        print()
        
        print("4️⃣  Emergency Vehicle Priority")
        scenarios.append(self.generate_emergency_vehicle())
        print()
        
        print("="*60)
        print(f"✅ Generated {len(scenarios)} test scenarios!")
        print("="*60)
        print("\n📖 To use these scenarios:")
        print("   1. Copy desired .rou.xml to your simulation")
        print("   2. Update dual_simulation_manager.py route_file parameter")
        print("   3. Run simulation and watch the dramatic differences!")
        print()
        
        return scenarios


if __name__ == "__main__":
    generator = ScenarioGenerator()
    generator.generate_all_scenarios()
    
    print("\n💡 TIPS FOR MAXIMUM VISUAL IMPACT:")
    print()
    print("   🔴 Scenario 1 (Rush Hour): Best for showing queue management")
    print("      → Watch North lane in heuristic stay smooth vs fixed gridlock")
    print()
    print("   🟡 Scenario 2 (Bursts): Best for showing responsiveness")  
    print("      → Watch how fast adaptive reacts to sudden traffic")
    print()
    print("   🔴 Scenario 3 (Empty Lane): Best for showing efficiency")
    print("      → Count how many times fixed wastes green on empty South")
    print()
    print("   🔴 Scenario 4 (Emergency): Best for showing life-saving features")
    print("      → Watch ambulance sail through adaptive vs stuck in fixed")
    print()
    print("🎬 Record screen video for MAXIMUM impact demonstration!")
    print()
