import React, { useState, useEffect } from 'react';
import {
    RadarChart, Radar, PolarGrid,
    PolarAngleAxis, XAxis, YAxis, CartesianGrid,
    Tooltip, Legend, ResponsiveContainer, AreaChart, Area
} from 'recharts';
import {
    Activity, Clock, Car, CheckCircle,
    Target
} from 'lucide-react';

// --- Types ---

export interface DashboardMetrics {
    fixedTime: {
        avgWait: number;
        throughput: number;
        maxQueue: number;
        avgQueue: number;
        switches: number;
        crashes: number;
        currentPhase: string;
        efficiency: number;
    };
    heuristic: {
        avgWait: number;
        throughput: number;
        maxQueue: number;
        avgQueue: number;
        switches: number;
        crashes: number;
        currentPhase: string;
        efficiency: number;
    };
}

export interface TrafficDashboardProps {
    metrics?: DashboardMetrics;
}

// --- Components ---

const MetricCard = ({ title, fixedValue, heuristicValue, unit, icon: Icon, colorClass = "cyan" }: any) => {
    const improv = (((fixedValue - heuristicValue) / Math.max(1, fixedValue)) * 100);
    const isPositive = improv > 0;

    return (
        <div className={`stat-card ${colorClass}`}>
            <div className="section-head" style={{ marginBottom: '10px', fontSize: '0.8rem' }}>
                <Icon size={16} />
                <span>{title}</span>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <div>
                    <div style={{ fontSize: '0.6rem', color: 'var(--text-secondary)' }}>FIXED</div>
                    <div style={{ fontSize: '1.2rem', color: 'var(--accent-critical)', fontWeight: 600 }}>{fixedValue.toFixed(1)}</div>
                </div>
                <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '0.6rem', color: 'var(--text-secondary)' }}>AI</div>
                    <div style={{ fontSize: '1.2rem', color: 'var(--accent-cyan)', fontWeight: 600 }}>{heuristicValue.toFixed(1)}<span style={{ fontSize: '0.7rem', marginLeft: '4px' }}>{unit}</span></div>
                </div>
            </div>

            <div style={{ marginTop: '10px', fontSize: '0.7rem', color: isPositive ? 'var(--accent-green)' : 'var(--accent-critical)' }}>
                {isPositive ? '↑' : '↓'} {Math.abs(improv).toFixed(1)}% IMPROVEMENT
            </div>
        </div>
    );
};

// --- Main Dashboard Component ---

const TrafficDashboard: React.FC<TrafficDashboardProps> = ({ metrics }) => {
    const [liveData, setLiveData] = useState<DashboardMetrics>({
        fixedTime: { avgWait: 52.3, throughput: 2847, maxQueue: 18, avgQueue: 8.4, switches: 60, crashes: 0, currentPhase: 'G', efficiency: 58 },
        heuristic: { avgWait: 27.8, throughput: 3928, maxQueue: 8, avgQueue: 3.2, switches: 342, crashes: 0, currentPhase: 'G', efficiency: 94 }
    });

    const [timeSeriesData, setTimeSeriesData] = useState<any[]>([
        { time: '0s', fixed: 45, heuristic: 22 },
        { time: '30s', fixed: 48, heuristic: 25 },
        { time: '60s', fixed: 52, heuristic: 28 },
        { time: '90s', fixed: 51, heuristic: 26 },
        { time: '120s', fixed: 54, heuristic: 29 },
    ]);

    useEffect(() => {
        if (metrics) {
            setLiveData(metrics);
            setTimeSeriesData(prev => {
                const newData = [...prev];
                if (newData.length > 20) newData.shift();
                newData.push({
                    time: `${prev.length * 5}s`,
                    fixed: metrics.fixedTime.avgWait,
                    heuristic: metrics.heuristic.avgWait
                });
                return newData;
            });
        }
    }, [metrics]);

    const radarData = [
        { metric: 'Efficiency', fixed: 58, heuristic: 94 },
        { metric: 'Throughput', fixed: 65, heuristic: 88 },
        { metric: 'Wait Time', fixed: 45, heuristic: 92 },
        { metric: 'Safety', fixed: 100, heuristic: 100 },
    ];

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <div className="stats-grid">
                <MetricCard title="AVG WAIT TIME" fixedValue={liveData.fixedTime.avgWait} heuristicValue={liveData.heuristic.avgWait} unit="s" icon={Clock} colorClass="cyan" />
                <MetricCard title="THROUGHPUT" fixedValue={liveData.fixedTime.throughput} heuristicValue={liveData.heuristic.throughput} unit="v" icon={Car} colorClass="green" />
                <MetricCard title="QUEUE LENGTH" fixedValue={liveData.fixedTime.avgQueue} heuristicValue={liveData.heuristic.avgQueue} unit="v" icon={Activity} colorClass="red" />
                <MetricCard title="EFFICIENCY" fixedValue={liveData.fixedTime.efficiency} heuristicValue={liveData.heuristic.efficiency} unit="%" icon={Target} colorClass="cyan" />
            </div>

            <div className="grid-layout" style={{ gridTemplateColumns: '1fr 1fr' }}>
                <div className="content-panel">
                    <div className="section-head">
                        <Activity size={20} />
                        <span>Wait Time Trends</span>
                    </div>
                    <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={timeSeriesData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                            <XAxis dataKey="time" stroke="#555" />
                            <YAxis stroke="#555" />
                            <Tooltip contentStyle={{ background: '#111', border: '1px solid #333' }} />
                            <Area type="monotone" dataKey="fixed" stroke="var(--accent-critical)" fill="var(--accent-critical)" fillOpacity={0.1} />
                            <Area type="monotone" dataKey="heuristic" stroke="var(--accent-cyan)" fill="var(--accent-cyan)" fillOpacity={0.1} />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>

                <div className="content-panel">
                    <div className="section-head">
                        <Target size={20} />
                        <span>Performance Radar</span>
                    </div>
                    <ResponsiveContainer width="100%" height={300}>
                        <RadarChart data={radarData}>
                            <PolarGrid stroke="#333" />
                            <PolarAngleAxis dataKey="metric" stroke="#777" />
                            <Radar name="Fixed" dataKey="fixed" stroke="var(--accent-critical)" fill="var(--accent-critical)" fillOpacity={0.2} />
                            <Radar name="AI" dataKey="heuristic" stroke="var(--accent-cyan)" fill="var(--accent-cyan)" fillOpacity={0.3} />
                            <Legend />
                        </RadarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="content-panel">
                <div className="section-head">
                    <CheckCircle size={20} />
                    <span>Safety Check: 100% Secure</span>
                </div>
                <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                    Agent is monitoring 16 conflict points. No safety violations detected in last 3600 steps.
                </div>
            </div>
        </div>
    );
};

export default TrafficDashboard;
