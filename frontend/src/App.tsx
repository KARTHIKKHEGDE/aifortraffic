import { useState } from 'react';
import { simulationAPI } from './api/simulation';

interface Scenario {
  id: string;
  name: string;
  difficulty: string;
  complexity: string;
  agents: number;
  network: string;
  routes: string;
}

function App() {
  const [isSimulating, setIsSimulating] = useState(false);
  const [comparison, setComparison] = useState<any>(null);

  const scenarios: Scenario[] = [
    {
      id: 'single',
      name: 'JUST ONE INTERSECTION',
      difficulty: 'CRITICAL',
      complexity: 'LOW',
      agents: 1,
      network: 'sumo/networks/simple_intersection.net.xml',
      routes: 'sumo/routes/simple_intersection.rou.xml'
    },
    {
      id: 'grid_3x4',
      name: '12-15 INTERSECTIONS',
      difficulty: 'HIGH',
      complexity: 'EXTREME',
      agents: 12,
      network: 'sumo/networks/grid_3x4.net.xml',
      routes: 'sumo/routes/grid_3x4.rou.xml'
    }
  ];

  const startSimulation = async (scenario: Scenario) => {
    try {
      setIsSimulating(true);

      // Start simulation via API
      await simulationAPI.startSimulation({
        scenario: scenario.id,
        network_file: scenario.network,
        route_file: scenario.routes,
        gui: true
      });

      // Connect WebSocket for updates
      const ws = simulationAPI.connectWebSocket((data) => {
        if (data.status === 'running' && data.data) {
          setComparison(data.data);
        } else if (data.status === 'complete') {
          setIsSimulating(false);
          alert('Simulation complete!');
        }
      });

      return () => ws.close();

    } catch (error) {
      console.error('Failed to start simulation:', error);
      alert('Failed to start simulation');
      setIsSimulating(false);
    }
  };

  return (
    <div className="app-container">
      <div className="selection-screen">
        <div className="top-bar">
          Current_Session :: NETWORK_SELECTION
        </div>

        <h1 className="section-title">SELECT NETWORK TOPOLOGY</h1>

        <div className="cards-container">
          {scenarios.map((scenario) => (
            <div
              key={scenario.id}
              className={`card ${scenario.id === 'single' ? 'critical-border' : 'high-border'
                }`}
              onClick={() => !isSimulating && startSimulation(scenario)}
            >
              <div className="card-header">
                <span className="card-id">SCENARIO-{scenario.id === 'single' ? '01' : '02'}</span>
                <span className={`badge ${scenario.id === 'single' ? 'badge-critical' : 'badge-high'
                  }`}>
                  {scenario.difficulty}
                </span>
              </div>

              <h3>{scenario.name}</h3>

              <div className="card-stats">
                <div className="stat">
                  <span className="label">COMPLEXITY</span>
                  <span className={`value ${scenario.id === 'single' ? 'text-cyan' : 'text-cyan'
                    }`}>
                    {scenario.complexity}
                  </span>
                </div>
                <div className="stat">
                  <span className="label">AGENTS</span>
                  <span className="value">{scenario.agents}{scenario.id !== 'single' && '+'}</span>
                </div>
              </div>

              <p className="card-desc">
                {scenario.id === 'single'
                  ? 'Single agent focusing on optimizing a standalone junction. Ideal for initial policy training and debugging.'
                  : 'Coordinate multiple intersections in a dense urban grid. Requires multi-agent cooperation and advanced handling.'}
              </p>
            </div>
          ))}
        </div>

        {/* Live Comparison Output (Terminal Style) */}
        {comparison && (
          <div className="card" style={{ marginTop: '40px' }}>
            <div className="card-header">
              <span className="card-id">LIVE_METRICS</span>
              <span className="badge badge-critical blink">LIVE</span>
            </div>

            <div className="card-stats" style={{ justifyContent: 'space-around' }}>
              <div className="stat">
                <span className="label">FIXED TIMER (BASELINE)</span>
                <span className="value" style={{ color: '#ff2a2a' }}>{comparison.fixed.avg_wait_time}s</span>
              </div>

              <div className="stat">
                <span className="label">HEURISTIC (AI)</span>
                <span className="value" style={{ color: '#0aff68' }}>{comparison.heuristic.avg_wait_time}s</span>
              </div>

              <div className="stat">
                <span className="label">IMPROVEMENT</span>
                <span className="value text-cyan">{comparison.improvement_percentage}%</span>
              </div>
            </div>

            <div className="terminal-output">
              <div className="terminal-line">&gt; System Time: {comparison.time}s</div>
              <div className="terminal-line">&gt; Throughput Delta: {comparison.throughput_delta > 0 ? '+' : ''}{comparison.throughput_delta} vehicles</div>
              <div className="terminal-line">&gt; Active Agents: {comparison.heuristic.active_vehicles}</div>
            </div>
          </div>
        )}
      </div>

      <div className="footer-stats">
        <div>SYSTEM STATUS: ONLINE</div>
        <div>VERSION: 2.4.0</div>
      </div>
    </div>
  );
}

export default App;
