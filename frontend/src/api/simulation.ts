// API client for backend communication

const API_URL = 'http://localhost:8000';

export interface Scenario {
    id: string;
    name: string;
    traffic_light_count: number;
    network: string;
    routes: string;
}

export interface SimulationStatus {
    is_running: boolean;
    fixed_metrics: {
        waitingTime: number;
        queueLength: number;
        throughput: number;
    };
    heuristic_metrics: {
        waitingTime: number;
        queueLength: number;
        throughput: number;
    };
}

export async function startSimulation(scenarioId: string) {
    // We need to find the scenario details to get the network/routes
    const scenarios = await getScenarios();
    const scenario = scenarios.find(s => s.id === scenarioId);

    if (!scenario) throw new Error('Scenario not found');

    const response = await fetch(`${API_URL}/api/simulation/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            scenario: scenario.id,
            network_file: scenario.network,
            route_file: scenario.routes
        })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to start simulation');
    }

    return response.json();
}

export async function stopSimulation() {
    const response = await fetch(`${API_URL}/api/simulation/stop`, {
        method: 'POST'
    });
    return response.json();
}

export async function getSimulationStatus(): Promise<SimulationStatus> {
    const response = await fetch(`${API_URL}/api/simulation/status`);
    if (!response.ok) throw new Error('Failed to get status');
    const data = await response.json();

    return {
        is_running: data.status === 'running',
        fixed_metrics: data.data ? {
            waitingTime: data.data.fixed.avg_wait_time,
            queueLength: data.data.fixed.active_vehicles,
            throughput: data.data.fixed.total_arrived
        } : { waitingTime: 0, queueLength: 0, throughput: 0 },
        heuristic_metrics: data.data ? {
            waitingTime: data.data.heuristic.avg_wait_time,
            queueLength: data.data.heuristic.active_vehicles,
            throughput: data.data.heuristic.total_arrived
        } : { waitingTime: 0, queueLength: 0, throughput: 0 }
    };
}

export async function getScenarios(): Promise<Scenario[]> {
    const response = await fetch(`${API_URL}/api/scenarios`);
    if (!response.ok) return [];
    const data = await response.json();
    return data.scenarios || [];
}

export const simulationAPI = {
    startSimulation,
    stopSimulation,
    getStatus: getSimulationStatus,
    getScenarios,
    connectWebSocket(onMessage: (data: any) => void) {
        const ws = new WebSocket(`ws://localhost:8000/ws/simulation`);
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            // Map the data to internal format if needed
            if (data.status === 'running' && data.data) {
                onMessage({
                    is_running: true,
                    time: data.data.time,
                    fixed_metrics: {
                        waitingTime: data.data.fixed.avg_wait_time,
                        queueLength: data.data.fixed.active_vehicles,
                        throughput: data.data.fixed.total_arrived
                    },
                    heuristic_metrics: {
                        waitingTime: data.data.heuristic.avg_wait_time,
                        queueLength: data.data.heuristic.active_vehicles,
                        throughput: data.data.heuristic.total_arrived
                    }
                });
            } else {
                onMessage({
                    is_running: false,
                    fixed_metrics: { waitingTime: 0, queueLength: 0, throughput: 0 },
                    heuristic_metrics: { waitingTime: 0, queueLength: 0, throughput: 0 }
                });
            }
        };
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        return ws;
    },
};
