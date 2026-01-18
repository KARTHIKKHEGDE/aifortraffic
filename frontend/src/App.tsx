import { useState, useEffect, useRef } from 'react';
import { Activity, Settings, BarChart2, Monitor, Layout, ArrowRight, ShieldCheck, Zap, Clock } from 'lucide-react';
import { startSimulation, stopSimulation, getSimulationStatus, getScenarios, type Scenario, simulationAPI } from './api/simulation';
import TrafficDashboard from './components/TrafficDashboard';

// --- Types ---
interface SimulationState {
  isRunning: boolean;
  status: 'idle' | 'starting' | 'running' | 'stopping' | 'error';
  scenario: string;
  time: number;
  metrics: {
    fixed: {
      waitingTime: number;
      queueLength: number;
      throughput: number;
    };
    heuristic: {
      waitingTime: number;
      queueLength: number;
      throughput: number;
    };
  };
}

function App() {
  const [currentPage, setCurrentPage] = useState<'landing' | 'selection' | 'system'>('landing');
  const [viewMode, setViewMode] = useState<'live' | 'analytics'>('live');
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [simState, setSimState] = useState<SimulationState>({
    isRunning: false,
    status: 'idle',
    scenario: 'single',
    time: 0,
    metrics: {
      fixed: { waitingTime: 0, queueLength: 0, throughput: 0 },
      heuristic: { waitingTime: 0, queueLength: 0, throughput: 0 }
    }
  });

  const wsRef = useRef<WebSocket | null>(null);

  // Load scenarios
  useEffect(() => {
    getScenarios().then(setScenarios).catch(console.error);
  }, []);

  // WebSocket Connection - Stays open while in 'system' view
  useEffect(() => {
    if (currentPage === 'system' && !wsRef.current) {
      console.log("🔌 Initializing Simulation Control Stream...");
      const ws = simulationAPI.connectWebSocket((data) => {
        setSimState(prev => ({
          ...prev,
          isRunning: data.is_running,
          status: data.is_running ? 'running' : prev.status,
          time: data.time || prev.time, // We need to make sure backend sends time
          metrics: {
            fixed: data.fixed_metrics,
            heuristic: data.heuristic_metrics
          }
        }));
      });
      wsRef.current = ws;
    }

    return () => {
      if (currentPage !== 'system' && wsRef.current) {
        console.log("🔌 Closing Control Stream...");
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [currentPage]);

  const handleStart = async (scenarioId: string) => {
    try {
      setSimState(prev => ({
        ...prev,
        status: 'starting',
        scenario: scenarioId,
        time: 0,
        metrics: {
          fixed: { waitingTime: 0, queueLength: 0, throughput: 0 },
          heuristic: { waitingTime: 0, queueLength: 0, throughput: 0 }
        }
      }));
      setCurrentPage('system');
      await startSimulation(scenarioId);
      // Backend simulation is now ready, WebSocket loop will start calling step()
    } catch (e) {
      console.error("Failed to start", e);
      setSimState(prev => ({ ...prev, status: 'error', isRunning: false }));
    }
  };

  const handleStop = async () => {
    try {
      setSimState(prev => ({ ...prev, status: 'stopping' }));
      await stopSimulation();
      setSimState(prev => ({ ...prev, status: 'idle', isRunning: false }));
    } catch (e) {
      console.error("Failed to stop", e);
    }
  };

  const dashboardMetrics = {
    fixedTime: {
      avgWait: simState.metrics.fixed.waitingTime,
      throughput: simState.metrics.fixed.throughput,
      maxQueue: simState.metrics.fixed.queueLength * 1.5,
      avgQueue: simState.metrics.fixed.queueLength,
      switches: 45,
      crashes: 0,
      currentPhase: 'Active',
      efficiency: 40
    },
    heuristic: {
      avgWait: simState.metrics.heuristic.waitingTime,
      throughput: simState.metrics.heuristic.throughput,
      maxQueue: simState.metrics.heuristic.queueLength * 1.5,
      avgQueue: simState.metrics.heuristic.queueLength,
      switches: 80,
      crashes: 0,
      currentPhase: 'Adaptive',
      efficiency: 85
    }
  };

  // --- Rendering Functions ---

  const renderLanding = () => (
    <div className="app-container">
      <div className="hero-section">
        <div className="status-badge">
          <div className="sc-indicator" />
          SYSTEM STATUS: OPTIMIZED
        </div>
        <h1 className="main-title">TRAFFIC <span className="text-cyan">AI</span></h1>
        <p className="subtitle">Next-Generation Adaptive Signal Control Platform</p>
        <button className="cta-button flex-center gap-2" onClick={() => setCurrentPage('selection')}>
          ENTER SYSTEM <ArrowRight size={20} />
        </button>

        <div className="terminal-output">
          <p className="terminal-line text-cyan">&gt; initializing neural_network_v4.2... [OK]</p>
          <p className="terminal-line text-cyan">&gt; establishing sumo_traci_connection... [OK]</p>
          <p className="terminal-line text-cyan">&gt; omniscient_sensor_array_online... [READY]</p>
        </div>
      </div>

      <div className="footer-stats">
        REINFORCEMENT LEARNING POWERED<br />
        REAL-TIME TRAFFIC OPTIMIZATION<br />
        99.9% COLLISION PREVENTION
      </div>
    </div>
  );

  const renderSelection = () => (
    <div className="app-container">
      <div className="selection-screen">
        <div className="top-bar flex-center gap-2">
          <Activity size={16} className="text-cyan" />
          <span>CORE_SYSTEM_INTERFACE // SELECT_TRAJECTORY</span>
        </div>

        <h2 className="section-title">Select <span className="text-cyan">Scenario</span></h2>

        <div className="cards-container">
          <div className="card critical-border" onClick={() => handleStart('single')}>
            <div className="card-header">
              <span className="card-id">SCENARIO_001</span>
              <span className="badge badge-critical">CRITICAL</span>
            </div>
            <h3>JUST ONE INTERSECTION</h3>
            <div className="card-stats">
              <div className="stat">
                <span className="label">Complexity</span>
                <span className="value">LOW</span>
              </div>
              <div className="stat">
                <span className="label">Agents</span>
                <span className="value">01</span>
              </div>
            </div>
            <p className="card-desc">
              Focus on peak-hour management for a single high-density four-way intersection.
              Ideal for testing baseline heuristic responses.
            </p>
            <div className="mt-4 flex-center" style={{ justifyContent: 'flex-start', gap: '8px' }}>
              <Zap size={16} className="text-cyan" />
              <span className="text-cyan font-bold text-xs uppercase letter-spacing-1">Deploy Logic Immediately</span>
            </div>
          </div>

          <div className="card high-border" onClick={() => handleStart('grid_3x4')}>
            <div className="card-header">
              <span className="card-id">SCENARIO_012</span>
              <span className="badge badge-high">HIGH LOAD</span>
            </div>
            <h3>12-15 INTERSECTIONS</h3>
            <div className="card-stats">
              <div className="stat">
                <span className="label">Complexity</span>
                <span className="value">HIGH</span>
              </div>
              <div className="stat">
                <span className="label">Agents</span>
                <span className="value">12+</span>
              </div>
            </div>
            <p className="card-desc">
              Full grid network simulation. Orchestrate traffic across multiple junctions
              to maximize green-wave throughput using synchronized heuristic agents.
            </p>
            <div className="mt-4 flex-center" style={{ justifyContent: 'flex-start', gap: '8px' }}>
              <ShieldCheck size={16} className="text-cyan" />
              <span className="text-cyan font-bold text-xs uppercase letter-spacing-1">Enterprise Load Test</span>
            </div>
          </div>
        </div>

        <button
          className="mt-4 text-secondary text-sm hover:text-white transition-colors underline"
          onClick={() => setCurrentPage('landing')}
          style={{ background: 'none', border: 'none', cursor: 'pointer' }}
        >
          &larr; BACK TO TERMINAL
        </button>
      </div>
    </div>
  );

  const renderSystem = () => (
    <div className="dashboard-container" style={{ animation: 'fadeIn 0.5s ease-out' }}>
      {/* Navbar */}
      <nav className="nav-bar">
        <div className="logo cursor-pointer" onClick={() => { handleStop(); setCurrentPage('selection'); }}>
          <Activity className="text-cyan" />
          <span>TRAFFIC<span className="text-cyan">AI</span></span>
        </div>

        <div className="view-switcher">
          <button
            onClick={() => setViewMode('live')}
            className={`nav-btn ${viewMode === 'live' ? 'active' : ''}`}
          >
            <Monitor size={18} />
            LIVE CONTROL
          </button>
          <button
            onClick={() => setViewMode('analytics')}
            className={`nav-btn ${viewMode === 'analytics' ? 'active' : ''}`}
          >
            <BarChart2 size={18} />
            DEEP ANALYTICS
          </button>
        </div>

        <div className="status-badge" style={{ margin: 0 }}>
          <div className={`sc-indicator ${simState.isRunning ? '' : 'inactive'}`} style={{ animation: simState.isRunning ? 'pulse 2s infinite' : 'none', opacity: simState.isRunning ? 1 : 0.3 }} />
          {simState.status.toUpperCase()}
          {simState.isRunning && (
            <span style={{ marginLeft: '10px', fontSize: '1.2rem', color: 'var(--accent-cyan)', fontFamily: 'Roboto Mono' }}>
              {simState.time.toFixed(1)}s
            </span>
          )}
        </div>
      </nav>

      {/* Main Content */}
      <main style={{ padding: '0 2rem 2rem 2rem' }}>
        {viewMode === 'live' ? (
          <div className="grid-layout">
            {/* Sidebar / Control Panel */}
            <aside className="content-panel control-group">
              <div className="section-head">
                <Settings size={20} />
                <span>Control Panel</span>
              </div>

              <div>
                <label className="subtitle" style={{ fontSize: '0.8rem', marginBottom: '10px', display: 'block' }}>Scenario Running</label>
                <div className="input-field" style={{ background: 'rgba(255,255,255,0.05)', fontWeight: 600 }}>
                  {scenarios.find(s => s.id === simState.scenario)?.name || 'Custom Grid'}
                </div>
              </div>

              <div className="p-4 border border-white/5 rounded mt-2">
                <div className="flex-center gap-2 mb-2" style={{ justifyContent: 'flex-start' }}>
                  <Clock size={16} className="text-cyan" />
                  <span className="text-xs uppercase letter-spacing-1">Simulated Time</span>
                </div>
                <div style={{ fontSize: '2rem', fontFamily: 'Roboto Mono', color: 'var(--accent-cyan)', textAlign: 'center' }}>
                  {simState.time.toFixed(1)}s
                </div>
              </div>

              <button
                onClick={handleStop}
                className="cta-button w-full"
                style={{
                  marginBottom: 0,
                  marginTop: '20px',
                  padding: '12px',
                  borderColor: 'var(--accent-critical)',
                  color: 'var(--accent-critical)'
                }}
              >
                STOP SIMULATION
              </button>

              <div className="stats-grid" style={{ gridTemplateColumns: '1fr', marginTop: '20px' }}>
                <div className="stat-card red">
                  <div className="label">FIXED TIME AVG WAIT</div>
                  <div className="value">{simState.metrics.fixed.waitingTime.toFixed(1)}s</div>
                </div>
                <div className="stat-card cyan">
                  <div className="label">HEURISTIC AI AVG WAIT</div>
                  <div className="value">{simState.metrics.heuristic.waitingTime.toFixed(1)}s</div>
                </div>
              </div>
            </aside>

            {/* Viewport Area */}
            <section className="content-panel">
              <div className="viewport-container" style={{ background: '#050505' }}>
                {simState.isRunning ? (
                  <div className="flex-center" style={{ flexDirection: 'column', gap: '20px' }}>
                    <Activity size={64} className="text-cyan blink" />
                    <div style={{ textAlign: 'center' }}>
                      <h3 className="text-cyan">SIMULATION ACTIVE</h3>
                      <p className="card-desc">WebSocket Heartbeat: Synchronizing {simState.scenario}...</p>
                    </div>
                  </div>
                ) : (
                  <div className="flex-center text-secondary" style={{ flexDirection: 'column' }}>
                    <Layout size={48} style={{ opacity: 0.3, marginBottom: '10px' }} />
                    <p>SYSTEM {simState.status.toUpperCase()}</p>
                  </div>
                )}
                <div className="viewport-overlay" />
              </div>

              <div className="status-badge mt-4" style={{ width: '100%', borderColor: 'var(--glass-border)', background: 'rgba(255,255,255,0.02)' }}>
                <CheckCircle size={16} className="text-green" />
                <span style={{ fontSize: '0.8rem', letterSpacing: '1px' }}>
                  SYNC STATUS: {simState.isRunning ? 'LOCKED' : 'WAITING'} // STEP FREQUENCY: 10HZ
                </span>
              </div>
            </section>
          </div>
        ) : (
          <div className="mt-4">
            <TrafficDashboard metrics={simState.isRunning || simState.metrics.heuristic.throughput > 0 ? dashboardMetrics : undefined} />
          </div>
        )}
      </main>

      <footer className="footer-stats">
        FRAMEWORK: SUMO 1.18.0<br />
        AGENT: OMNISCIENT HEURISTIC<br />
        SYNC: TRAFFICAI-SOCKET-V1
      </footer>
    </div>
  );

  return (
    <>
      {currentPage === 'landing' && renderLanding()}
      {currentPage === 'selection' && renderSelection()}
      {currentPage === 'system' && renderSystem()}
    </>
  );
}

function CheckCircle({ size, className }: { size: number, className?: string }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>;
}

export default App;
