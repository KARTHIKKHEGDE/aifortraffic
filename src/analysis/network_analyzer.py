"""
Network Topology Analyzer for SUMO Networks

Analyzes SUMO network topology for RL optimization including:
- Junction and edge parsing
- Graph-based path analysis
- Critical junction identification
- Traffic flow pattern prediction
- Network visualization
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Graph analysis disabled.")


@dataclass
class Lane:
    """Represents a single lane in the network"""
    id: str
    length: float
    speed: float
    width: float = 3.2


@dataclass
class Edge:
    """Represents a road segment (edge) in the network"""
    id: str
    from_junction: str
    to_junction: str
    priority: int
    lanes: List[Lane] = field(default_factory=list)
    
    @property
    def length(self) -> float:
        return self.lanes[0].length if self.lanes else 0
    
    @property
    def max_speed(self) -> float:
        return max(lane.speed for lane in self.lanes) if self.lanes else 13.89


@dataclass
class Junction:
    """Represents a junction (intersection) in the network"""
    id: str
    type: str
    pos: Tuple[float, float]
    incoming: List[str] = field(default_factory=list)
    outgoing: List[str] = field(default_factory=list)
    shape: Optional[str] = None


@dataclass
class TrafficLight:
    """Represents a traffic light controller"""
    id: str
    type: str
    phases: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def n_phases(self) -> int:
        return len(self.phases)
    
    @property
    def cycle_time(self) -> float:
        return sum(p['duration'] for p in self.phases)


@dataclass
class FlowPattern:
    """Traffic flow pattern for a junction"""
    n_incoming: int
    n_outgoing: int
    avg_incoming_priority: float
    avg_outgoing_priority: float
    main_direction: str
    is_bottleneck: bool


class NetworkAnalyzer:
    """
    Analyzes SUMO network topology for RL optimization
    
    Features:
    - Parse network XML structure
    - Build NetworkX graph for path analysis
    - Identify critical junctions using centrality measures
    - Compute Origin-Destination matrices
    - Analyze traffic flow patterns
    - Visualize network topology
    """
    
    def __init__(self, net_file: str):
        """
        Initialize analyzer with SUMO network file
        
        Args:
            net_file: Path to .net.xml file
        """
        self.net_file = Path(net_file)
        
        if not self.net_file.exists():
            raise FileNotFoundError(f"Network file not found: {net_file}")
        
        self.tree = ET.parse(self.net_file)
        self.root = self.tree.getroot()
        
        # Data structures
        self.junctions: Dict[str, Junction] = {}
        self.edges: Dict[str, Edge] = {}
        self.connections: Dict[str, List[Dict]] = {}
        self.traffic_lights: Dict[str, TrafficLight] = {}
        
        # Analysis results
        self.graph = None
        self.od_matrix = None
        self.od_paths = None
        
        # Parse network
        self._parse_network()
        
        # Build graph if networkx available
        if HAS_NETWORKX:
            self._build_graph()
    
    def _parse_network(self):
        """Extract all network elements from XML"""
        self._parse_junctions()
        self._parse_edges()
        self._parse_connections()
        self._parse_traffic_lights()
    
    def _parse_junctions(self):
        """Parse junction elements"""
        for junction in self.root.findall('junction'):
            jid = junction.get('id')
            jtype = junction.get('type', 'unknown')
            
            # Skip internal junctions
            if jid.startswith(':'):
                continue
            
            x = float(junction.get('x', 0))
            y = float(junction.get('y', 0))
            shape = junction.get('shape')
            
            self.junctions[jid] = Junction(
                id=jid,
                type=jtype,
                pos=(x, y),
                incoming=[],
                outgoing=[],
                shape=shape
            )
    
    def _parse_edges(self):
        """Parse edge (road) elements"""
        for edge in self.root.findall('edge'):
            eid = edge.get('id')
            
            # Skip internal edges
            if eid.startswith(':'):
                continue
            
            from_junction = edge.get('from')
            to_junction = edge.get('to')
            priority = int(edge.get('priority', 1))
            
            # Parse lanes
            lanes = []
            for lane in edge.findall('lane'):
                lane_id = lane.get('id')
                length = float(lane.get('length', 100))
                speed = float(lane.get('speed', 13.89))
                width = float(lane.get('width', 3.2))
                
                lanes.append(Lane(
                    id=lane_id,
                    length=length,
                    speed=speed,
                    width=width
                ))
            
            self.edges[eid] = Edge(
                id=eid,
                from_junction=from_junction,
                to_junction=to_junction,
                priority=priority,
                lanes=lanes
            )
            
            # Update junction connections
            if from_junction in self.junctions:
                self.junctions[from_junction].outgoing.append(eid)
            if to_junction in self.junctions:
                self.junctions[to_junction].incoming.append(eid)
    
    def _parse_connections(self):
        """Parse lane connections"""
        for conn in self.root.findall('connection'):
            from_edge = conn.get('from')
            to_edge = conn.get('to')
            from_lane = int(conn.get('fromLane', 0))
            to_lane = int(conn.get('toLane', 0))
            
            if from_edge not in self.connections:
                self.connections[from_edge] = []
            
            self.connections[from_edge].append({
                'to': to_edge,
                'fromLane': from_lane,
                'toLane': to_lane,
                'dir': conn.get('dir', 's'),
                'state': conn.get('state', 'o')
            })
    
    def _parse_traffic_lights(self):
        """Parse traffic light logic"""
        for tl in self.root.findall('tlLogic'):
            tl_id = tl.get('id')
            tl_type = tl.get('type', 'static')
            
            phases = []
            for phase in tl.findall('phase'):
                duration = float(phase.get('duration', 30))
                state = phase.get('state', '')
                min_dur = float(phase.get('minDur', duration))
                max_dur = float(phase.get('maxDur', duration))
                
                phases.append({
                    'duration': duration,
                    'state': state,
                    'minDur': min_dur,
                    'maxDur': max_dur
                })
            
            self.traffic_lights[tl_id] = TrafficLight(
                id=tl_id,
                type=tl_type,
                phases=phases
            )
    
    def _build_graph(self):
        """Create NetworkX graph for path analysis"""
        if not HAS_NETWORKX:
            return
        
        self.graph = nx.DiGraph()
        
        # Add nodes (junctions)
        for jid, jdata in self.junctions.items():
            self.graph.add_node(
                jid,
                type=jdata.type,
                pos=jdata.pos
            )
        
        # Add edges with weights (travel time)
        for eid, edata in self.edges.items():
            from_j = edata.from_junction
            to_j = edata.to_junction
            
            if from_j not in self.junctions or to_j not in self.junctions:
                continue
            
            # Calculate edge weight (travel time in seconds)
            length = edata.length
            speed = edata.max_speed if edata.max_speed > 0 else 10
            travel_time = length / speed
            
            self.graph.add_edge(
                from_j, to_j,
                edge_id=eid,
                weight=travel_time,
                length=length,
                priority=edata.priority
            )
    
    def find_critical_junctions(self, n: int = 4) -> List[Tuple[str, float]]:
        """
        Identify most critical junctions using centrality measures
        
        Args:
            n: Number of top junctions to return
            
        Returns:
            List of (junction_id, score) tuples
        """
        if not HAS_NETWORKX or self.graph is None:
            # Fallback: return junctions with most connections
            scores = {}
            for jid, jdata in self.junctions.items():
                if jdata.type == 'traffic_light':
                    scores[jid] = len(jdata.incoming) + len(jdata.outgoing)
            sorted_junctions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_junctions[:n]
        
        # Betweenness centrality - how many shortest paths go through this junction
        try:
            betweenness = nx.betweenness_centrality(self.graph, weight='weight')
        except:
            betweenness = {jid: 0 for jid in self.junctions}
        
        # Degree centrality - how many roads connect
        try:
            degree = nx.degree_centrality(self.graph)
        except:
            degree = {jid: 0 for jid in self.junctions}
        
        # Traffic light junctions only
        tl_junctions = set(self.traffic_lights.keys())
        
        # Combined score
        scores = {}
        for jid in self.junctions:
            if jid in tl_junctions or self.junctions[jid].type == 'traffic_light':
                b = betweenness.get(jid, 0)
                d = degree.get(jid, 0)
                scores[jid] = 0.7 * b + 0.3 * d
        
        # Sort and return top N
        sorted_junctions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_junctions[:n]
    
    def compute_junction_neighborhoods(
        self, 
        junction_id: str, 
        radius: int = 2
    ) -> Dict[str, int]:
        """
        Find all junctions within N hops
        
        Args:
            junction_id: Starting junction
            radius: Maximum hop distance
            
        Returns:
            Dict mapping neighbor_id -> distance
        """
        if junction_id not in self.junctions:
            return {}
        
        neighbors = {}
        visited = {junction_id}
        queue = [(junction_id, 0)]
        
        while queue:
            current, distance = queue.pop(0)
            
            if distance >= radius:
                continue
            
            # Get neighbors from graph or edge data
            if HAS_NETWORKX and self.graph is not None:
                next_nodes = list(self.graph.neighbors(current))
            else:
                # Fallback: use edge data
                next_nodes = []
                for eid in self.junctions[current].outgoing:
                    if eid in self.edges:
                        next_nodes.append(self.edges[eid].to_junction)
            
            for neighbor in next_nodes:
                if neighbor not in visited and neighbor in self.junctions:
                    visited.add(neighbor)
                    neighbors[neighbor] = distance + 1
                    queue.append((neighbor, distance + 1))
        
        return neighbors
    
    def compute_od_matrix(self) -> Tuple[np.ndarray, Dict]:
        """
        Compute Origin-Destination matrix for traffic routing
        
        Returns:
            Tuple of (od_matrix, od_paths)
        """
        if not HAS_NETWORKX or self.graph is None:
            raise RuntimeError("NetworkX required for OD matrix computation")
        
        # All junctions
        junctions = list(self.junctions.keys())
        n = len(junctions)
        junction_idx = {jid: i for i, jid in enumerate(junctions)}
        
        # OD matrix: od[i][j] = shortest path length from i to j
        od_matrix = np.full((n, n), np.inf)
        od_paths = {}
        
        for i, origin in enumerate(junctions):
            for j, destination in enumerate(junctions):
                if i == j:
                    od_matrix[i][j] = 0
                    continue
                
                try:
                    path = nx.shortest_path(
                        self.graph, origin, destination, weight='weight'
                    )
                    length = nx.shortest_path_length(
                        self.graph, origin, destination, weight='weight'
                    )
                    
                    od_matrix[i][j] = length
                    od_paths[(origin, destination)] = path
                except nx.NetworkXNoPath:
                    pass
        
        self.od_matrix = od_matrix
        self.od_paths = od_paths
        self.junction_indices = junction_idx
        
        return od_matrix, od_paths
    
    def analyze_traffic_flow_patterns(self) -> Dict[str, FlowPattern]:
        """
        Predict natural traffic patterns based on topology
        
        Returns:
            Dict mapping junction_id -> FlowPattern
        """
        flow_patterns = {}
        
        for jid, jdata in self.junctions.items():
            if jdata.type != 'traffic_light':
                continue
            
            incoming_edges = jdata.incoming
            outgoing_edges = jdata.outgoing
            
            # Analyze priorities
            incoming_priorities = [
                self.edges[eid].priority 
                for eid in incoming_edges 
                if eid in self.edges
            ]
            outgoing_priorities = [
                self.edges[eid].priority 
                for eid in outgoing_edges 
                if eid in self.edges
            ]
            
            # Determine main flow direction
            main_direction = self._determine_main_direction(jid)
            
            flow_patterns[jid] = FlowPattern(
                n_incoming=len(incoming_edges),
                n_outgoing=len(outgoing_edges),
                avg_incoming_priority=np.mean(incoming_priorities) if incoming_priorities else 0,
                avg_outgoing_priority=np.mean(outgoing_priorities) if outgoing_priorities else 0,
                main_direction=main_direction,
                is_bottleneck=len(incoming_edges) > len(outgoing_edges)
            )
        
        return flow_patterns
    
    def _determine_main_direction(self, junction_id: str) -> str:
        """
        Determine primary traffic flow direction (N-S, E-W, etc.)
        """
        if junction_id not in self.junctions:
            return 'unknown'
        
        jdata = self.junctions[junction_id]
        incoming = jdata.incoming
        
        if not incoming:
            return 'unknown'
        
        # Get angles of incoming roads
        angles = []
        for edge_id in incoming:
            if edge_id in self.edges:
                from_j = self.edges[edge_id].from_junction
                if from_j in self.junctions:
                    from_pos = self.junctions[from_j].pos
                    to_pos = jdata.pos
                    
                    # Calculate angle
                    dx = to_pos[0] - from_pos[0]
                    dy = to_pos[1] - from_pos[1]
                    
                    if dx != 0 or dy != 0:
                        angle = np.arctan2(dy, dx) * 180 / np.pi
                        angles.append(angle)
        
        if not angles:
            return 'unknown'
        
        angles = np.array(angles)
        
        # Check if primarily N-S (angles near ±90°)
        ns_mask = np.abs(np.abs(angles) - 90) < 30
        if np.sum(ns_mask) > len(angles) / 2:
            return 'north-south'
        
        # Check if primarily E-W (angles near 0° or 180°)
        ew_mask = (np.abs(angles) < 30) | (np.abs(np.abs(angles) - 180) < 30)
        if np.sum(ew_mask) > len(angles) / 2:
            return 'east-west'
        
        return 'mixed'
    
    def get_junction_info(self, junction_id: str) -> Dict[str, Any]:
        """Get comprehensive info about a junction"""
        if junction_id not in self.junctions:
            return {}
        
        jdata = self.junctions[junction_id]
        
        info = {
            'id': junction_id,
            'type': jdata.type,
            'position': jdata.pos,
            'n_incoming': len(jdata.incoming),
            'n_outgoing': len(jdata.outgoing),
            'incoming_edges': jdata.incoming,
            'outgoing_edges': jdata.outgoing,
        }
        
        # Add traffic light info if available
        if junction_id in self.traffic_lights:
            tl = self.traffic_lights[junction_id]
            info['traffic_light'] = {
                'type': tl.type,
                'n_phases': tl.n_phases,
                'cycle_time': tl.cycle_time,
                'phases': tl.phases
            }
        
        # Add neighborhood info
        info['neighbors'] = self.compute_junction_neighborhoods(junction_id, radius=1)
        
        return info
    
    def visualize_network(
        self, 
        highlight_junctions: Optional[List[str]] = None,
        output_file: str = 'network_topology.png',
        show: bool = False
    ):
        """
        Create visualization of network
        
        Args:
            highlight_junctions: List of junction IDs to highlight
            output_file: Path to save visualization
            show: Whether to display plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for visualization")
            return
        
        fig, ax = plt.subplots(figsize=(15, 15))
        
        highlight_set = set(highlight_junctions or [])
        
        # Draw edges first
        for eid, edata in self.edges.items():
            from_j = edata.from_junction
            to_j = edata.to_junction
            
            if from_j in self.junctions and to_j in self.junctions:
                x1, y1 = self.junctions[from_j].pos
                x2, y2 = self.junctions[to_j].pos
                
                # Color based on priority
                alpha = 0.2 + 0.1 * edata.priority
                linewidth = 0.3 + 0.2 * edata.priority
                
                ax.plot([x1, x2], [y1, y2], 'b-', 
                       alpha=min(alpha, 0.6), 
                       linewidth=linewidth, 
                       zorder=1)
        
        # Draw junctions
        for jid, jdata in self.junctions.items():
            x, y = jdata.pos
            
            if jdata.type == 'traffic_light':
                if jid in highlight_set:
                    color = 'red'
                    size = 200
                    ax.annotate(jid, (x, y+80), fontsize=8, ha='center')
                else:
                    color = 'orange'
                    size = 50
            else:
                color = 'gray'
                size = 10
            
            ax.scatter(x, y, c=color, s=size, alpha=0.7, zorder=3)
        
        ax.set_title('SUMO Network Topology', fontsize=14)
        ax.set_xlabel('X coordinate (m)')
        ax.set_ylabel('Y coordinate (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Network visualization saved to {output_file}")
        
        if show:
            plt.show()
        
        plt.close()
        
        return fig, ax
    
    def export_analysis_report(self, output_file: str = 'network_analysis.txt'):
        """Generate comprehensive analysis report"""
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SUMO NETWORK ANALYSIS REPORT\n")
            f.write(f"Network file: {self.net_file}\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic stats
            f.write("NETWORK STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Junctions: {len(self.junctions)}\n")
            f.write(f"Traffic Light Junctions: {len(self.traffic_lights)}\n")
            f.write(f"Total Edges: {len(self.edges)}\n")
            
            total_length = sum(e.length for e in self.edges.values())
            f.write(f"Total Road Length: {total_length:.2f} m ({total_length/1000:.2f} km)\n\n")
            
            # Critical junctions
            f.write("CRITICAL JUNCTIONS (Top 10)\n")
            f.write("-" * 40 + "\n")
            critical = self.find_critical_junctions(n=10)
            for i, (jid, score) in enumerate(critical, 1):
                jdata = self.junctions.get(jid)
                n_in = len(jdata.incoming) if jdata else 0
                n_out = len(jdata.outgoing) if jdata else 0
                f.write(f"{i}. {jid}\n")
                f.write(f"   Score: {score:.4f}\n")
                f.write(f"   Incoming: {n_in}, Outgoing: {n_out}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            
            # Traffic light details
            f.write("\nTRAFFIC LIGHT CONFIGURATIONS\n")
            f.write("-" * 40 + "\n")
            for tl_id, tl in self.traffic_lights.items():
                f.write(f"\n{tl_id}:\n")
                f.write(f"  Type: {tl.type}\n")
                f.write(f"  Phases: {tl.n_phases}\n")
                f.write(f"  Cycle Time: {tl.cycle_time:.1f}s\n")
                for i, phase in enumerate(tl.phases):
                    f.write(f"  Phase {i}: {phase['duration']:.1f}s - {phase['state'][:20]}...\n")
            
            f.write("\n" + "=" * 80 + "\n")
            
            # Flow patterns
            f.write("\nTRAFFIC FLOW PATTERNS\n")
            f.write("-" * 40 + "\n")
            patterns = self.analyze_traffic_flow_patterns()
            for jid, pattern in patterns.items():
                f.write(f"\n{jid}:\n")
                f.write(f"  Incoming roads: {pattern.n_incoming}\n")
                f.write(f"  Outgoing roads: {pattern.n_outgoing}\n")
                f.write(f"  Main direction: {pattern.main_direction}\n")
                f.write(f"  Is Bottleneck: {pattern.is_bottleneck}\n")
        
        print(f"Analysis report saved to {output_file}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get network summary as dictionary"""
        return {
            'n_junctions': len(self.junctions),
            'n_traffic_lights': len(self.traffic_lights),
            'n_edges': len(self.edges),
            'total_length_km': sum(e.length for e in self.edges.values()) / 1000,
            'critical_junctions': self.find_critical_junctions(n=4),
            'traffic_light_ids': list(self.traffic_lights.keys())
        }


# Command-line interface
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze SUMO network topology')
    parser.add_argument('net_file', help='Path to .net.xml file')
    parser.add_argument('--top', type=int, default=4, help='Number of critical junctions')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    parser.add_argument('--report', action='store_true', help='Generate analysis report')
    parser.add_argument('--output', default='network_analysis', help='Output file prefix')
    
    args = parser.parse_args()
    
    print(f"Analyzing network: {args.net_file}")
    analyzer = NetworkAnalyzer(args.net_file)
    
    # Print summary
    summary = analyzer.get_summary()
    print(f"\nNetwork Statistics:")
    print(f"  Junctions: {summary['n_junctions']}")
    print(f"  Traffic Lights: {summary['n_traffic_lights']}")
    print(f"  Edges: {summary['n_edges']}")
    print(f"  Total Length: {summary['total_length_km']:.2f} km")
    
    # Find critical junctions
    print(f"\nTop {args.top} Critical Junctions:")
    for jid, score in summary['critical_junctions'][:args.top]:
        print(f"  {jid}: {score:.4f}")
    
    # Visualize if requested
    if args.visualize:
        critical_ids = [jid for jid, _ in summary['critical_junctions'][:args.top]]
        analyzer.visualize_network(
            highlight_junctions=critical_ids,
            output_file=f"{args.output}.png"
        )
    
    # Generate report if requested
    if args.report:
        analyzer.export_analysis_report(f"{args.output}.txt")
