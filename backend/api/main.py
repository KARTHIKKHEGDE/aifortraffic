"""
FastAPI Backend for Dual Simulation
Handles requests from frontend to start/control simulations
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.dual_simulation_manager import DualSimulationManager

app = FastAPI(title="Traffic Control API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global simulation instance
current_simulation: DualSimulationManager = None

class SimulationConfig(BaseModel):
    scenario: str
    network_file: str
    route_file: str
    duration: float = 900.0
    gui: bool = True

@app.post("/api/simulation/start")
async def start_simulation(config: SimulationConfig):
    """
    Start dual simulation (Fixed vs Heuristic)
    """
    global current_simulation
    
    try:
        # Close existing simulation if any
        if current_simulation and current_simulation.is_running:
            current_simulation.close()
        
        # Create new simulation
        current_simulation = DualSimulationManager(
            network_file=config.network_file,
            route_file=config.route_file,
            gui=config.gui
        )
        
        # Start it
        current_simulation.start()
        
        return {
            "status": "success",
            "message": "Dual simulation started",
            "scenario": config.scenario
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulation/stop")
async def stop_simulation():
    """Stop current simulation"""
    global current_simulation
    
    if current_simulation:
        current_simulation.close()
        current_simulation = None
        return {"status": "success", "message": "Simulation stopped"}
    
    return {"status": "error", "message": "No simulation running"}

@app.get("/api/simulation/status")
async def get_status():
    """Get current simulation status"""
    if current_simulation and current_simulation.is_running:
        comparison = current_simulation.get_comparison()
        return {
            "status": "running",
            "data": comparison
        }
    
    return {"status": "stopped"}

@app.websocket("/ws/simulation")
async def simulation_websocket(websocket: WebSocket):
    """
    WebSocket for real-time simulation updates
    """
    await websocket.accept()
    
    try:
        while True:
            if not current_simulation or not current_simulation.is_running:
                await websocket.send_json({
                    "status": "stopped",
                    "message": "No simulation running"
                })
                await asyncio.sleep(1)
                continue
            
            # Step simulation
            comparison = current_simulation.step()
            
            # Send update to frontend
            await websocket.send_json({
                "status": "running",
                "data": comparison
            })
            
            # Check if done (15 minutes)
            if comparison['time'] >= 900:
                await websocket.send_json({
                    "status": "complete",
                    "data": comparison
                })
                break
            
            # Run at 10 FPS
            await asyncio.sleep(0.1)
    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
             pass

@app.get("/api/scenarios")
async def get_scenarios():
    """Get available scenarios"""
    return {
        "scenarios": [
            {
                "id": "single",
                "name": "JUST ONE INTERSECTION",
                "difficulty": "CRITICAL",
                "complexity": "LOW",
                "agents": 1,
                "network": "sumo/networks/simple_intersection.net.xml",
                "routes": "sumo/routes/simple_intersection.rou.xml"
            },
            {
                "id": "grid_3x4",
                "name": "12-15 INTERSECTIONS",
                "difficulty": "HIGH",
                "complexity": "EXTREME",
                "agents": 12,
                "network": "sumo/networks/grid_3x4.net.xml",
                "routes": "sumo/routes/grid_3x4.rou.xml"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
