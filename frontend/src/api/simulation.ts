// API client for backend communication

const API_URL = 'http://localhost:8000';

export interface SimulationConfig {
    scenario: string;
    network_file: string;
    route_file: string;
    duration?: number;
    gui?: boolean;
}

export interface SimulationStatus {
    status: 'stopped' | 'running' | 'complete';
    data?: {
        time: number;
        fixed: {
            avg_wait_time: number;
            total_arrived: number;
            active_vehicles: number;
        };
        heuristic: {
            avg_wait_time: number;
            total_arrived: number;
            active_vehicles: number;
        };
        improvement_percentage: number;
        throughput_delta: number;
    };
}

export const simulationAPI = {
    async startSimulation(config: SimulationConfig) {
        const response = await fetch(`${API_URL}/api/simulation/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        if (!response.ok) {
            throw new Error('Failed to start simulation');
        }

        return response.json();
    },

    async stopSimulation() {
        const response = await fetch(`${API_URL}/api/simulation/stop`, {
            method: 'POST'
        });
        return response.json();
    },

    async getStatus(): Promise<SimulationStatus> {
        const response = await fetch(`${API_URL}/api/simulation/status`);
        return response.json();
    },

    connectWebSocket(onMessage: (data: SimulationStatus) => void) {
        const ws = new WebSocket(`ws://localhost:8000/ws/simulation`);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            onMessage(data);
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        return ws;
    },

    async getScenarios() {
        const response = await fetch(`${API_URL}/api/scenarios`);
        return response.json();
    }
};
