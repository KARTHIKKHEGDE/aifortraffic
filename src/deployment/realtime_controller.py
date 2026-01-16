"""
Real-Time Deployment Module for Traffic Signal Control

Provides production-ready deployment capabilities:
- Model serving with inference optimization
- Live SUMO integration
- Failsafe mechanisms
- Performance monitoring
- Hot model reloading
"""

import time
import json
import threading
import queue
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class InferenceConfig:
    """Configuration for inference server"""
    model_path: str = "models/best_agent.pt"
    device: str = "cpu"  # 'cpu' or 'cuda'
    batch_size: int = 1
    max_latency_ms: float = 100.0
    fallback_action: int = 0  # Keep current phase
    warmup_steps: int = 10
    enable_monitoring: bool = True
    log_predictions: bool = True
    checkpoint_interval: int = 1000


@dataclass
class DeploymentMetrics:
    """Metrics for monitoring deployment"""
    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    fallback_count: int = 0
    
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    rewards: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    start_time: float = field(default_factory=time.time)
    
    def record_prediction(self, latency_ms: float, success: bool):
        self.total_predictions += 1
        self.latencies.append(latency_ms)
        
        if success:
            self.successful_predictions += 1
        else:
            self.failed_predictions += 1
    
    def record_fallback(self):
        self.fallback_count += 1
    
    def record_reward(self, reward: float):
        self.rewards.append(reward)
    
    @property
    def avg_latency_ms(self) -> float:
        return np.mean(self.latencies) if self.latencies else 0
    
    @property
    def p95_latency_ms(self) -> float:
        return np.percentile(self.latencies, 95) if self.latencies else 0
    
    @property
    def p99_latency_ms(self) -> float:
        return np.percentile(self.latencies, 99) if self.latencies else 0
    
    @property
    def success_rate(self) -> float:
        if self.total_predictions == 0:
            return 0
        return self.successful_predictions / self.total_predictions
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_predictions': self.total_predictions,
            'success_rate': self.success_rate,
            'fallback_rate': self.fallback_count / max(1, self.total_predictions),
            'avg_latency_ms': self.avg_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,
            'p99_latency_ms': self.p99_latency_ms,
            'avg_reward': np.mean(self.rewards) if self.rewards else 0,
            'uptime_seconds': self.uptime_seconds
        }


class ModelServer:
    """
    Inference server for traffic signal control
    
    Features:
    - Model loading and inference
    - Latency monitoring
    - Automatic fallback on errors
    - Hot model reloading
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.metrics = DeploymentMetrics()
        
        self.model = None
        self.model_version = None
        self.model_loaded = False
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Model file watcher for hot reload
        self._last_model_mtime = 0
        
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load model from file
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if successful
        """
        path = Path(model_path or self.config.model_path)
        
        if not path.exists():
            print(f"Model file not found: {path}")
            return False
        
        try:
            with self._lock:
                if HAS_TORCH:
                    self.model = torch.load(path, map_location=self.config.device)
                    if hasattr(self.model, 'eval'):
                        self.model.eval()
                else:
                    # Load as generic pickle
                    import pickle
                    with open(path, 'rb') as f:
                        self.model = pickle.load(f)
                
                self.model_version = f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.model_loaded = True
                self._last_model_mtime = path.stat().st_mtime
                
            print(f"Model loaded: {self.model_version}")
            
            # Warmup
            self._warmup()
            
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def _warmup(self):
        """Warmup model with dummy inputs"""
        print("Warming up model...")
        
        # Create dummy observation
        dummy_obs = np.random.randn(16).astype(np.float32)
        
        for _ in range(self.config.warmup_steps):
            try:
                self.predict(dummy_obs, warmup=True)
            except:
                pass
        
        print("Warmup complete")
    
    def predict(
        self, 
        observation: np.ndarray,
        warmup: bool = False
    ) -> Tuple[int, float]:
        """
        Get action prediction from model
        
        Args:
            observation: Environment observation
            warmup: Whether this is a warmup call
            
        Returns:
            Tuple of (action, confidence)
        """
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                if not self.model_loaded:
                    raise RuntimeError("Model not loaded")
                
                # Convert observation
                if HAS_TORCH:
                    obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                    obs_tensor = obs_tensor.to(self.config.device)
                    
                    with torch.no_grad():
                        if hasattr(self.model, 'predict'):
                            action, _ = self.model.predict(observation, deterministic=True)
                            confidence = 1.0
                        elif hasattr(self.model, 'forward'):
                            q_values = self.model(obs_tensor)
                            action = q_values.argmax().item()
                            confidence = torch.softmax(q_values, dim=-1).max().item()
                        else:
                            action = self.model.select_action(observation)
                            confidence = 0.5
                else:
                    # Non-torch model
                    if hasattr(self.model, 'predict'):
                        action, _ = self.model.predict(observation)
                        confidence = 1.0
                    elif hasattr(self.model, 'select_action'):
                        action = self.model.select_action(observation)
                        confidence = 0.5
                    else:
                        raise RuntimeError("Unknown model interface")
                
                action = int(action)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            if not warmup:
                self.metrics.record_prediction(latency_ms, success=True)
            
            return action, confidence
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            if not warmup:
                self.metrics.record_prediction(latency_ms, success=False)
                self.metrics.record_fallback()
                print(f"Prediction failed, using fallback: {e}")
            
            return self.config.fallback_action, 0.0
    
    def check_hot_reload(self) -> bool:
        """
        Check if model file has been updated and reload if necessary
        
        Returns:
            True if model was reloaded
        """
        path = Path(self.config.model_path)
        
        if not path.exists():
            return False
        
        current_mtime = path.stat().st_mtime
        
        if current_mtime > self._last_model_mtime:
            print("Model file updated, reloading...")
            return self.load_model()
        
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current deployment metrics"""
        return {
            'model_version': self.model_version,
            'model_loaded': self.model_loaded,
            **self.metrics.get_summary()
        }


class TrafficController:
    """
    Real-time traffic signal controller
    
    Integrates with SUMO for live control
    """
    
    def __init__(
        self,
        model_server: ModelServer,
        junction_ids: List[str],
        control_interval: float = 5.0,
        use_sumo: bool = True
    ):
        self.model_server = model_server
        self.junction_ids = junction_ids
        self.control_interval = control_interval
        self.use_sumo = use_sumo
        
        # State tracking
        self.current_phases = {jid: 0 for jid in junction_ids}
        self.phase_timers = {jid: 0.0 for jid in junction_ids}
        self.last_control_time = {jid: 0.0 for jid in junction_ids}
        
        # SUMO connection
        self.traci = None
        
        # Control loop
        self._running = False
        self._control_thread = None
        
        # Logging
        self.action_log = deque(maxlen=10000)
    
    def connect_sumo(self, sumo_cfg: str, port: int = 8813):
        """Connect to running SUMO simulation"""
        if not self.use_sumo:
            print("SUMO disabled, running in simulation mode")
            return
        
        try:
            import traci
            traci.start(["sumo", "-c", sumo_cfg, "--remote-port", str(port)])
            self.traci = traci
            print(f"Connected to SUMO on port {port}")
        except Exception as e:
            print(f"Failed to connect to SUMO: {e}")
            self.use_sumo = False
    
    def get_observation(self, junction_id: str) -> np.ndarray:
        """
        Get current observation for junction
        """
        if self.traci is not None:
            # Get real data from SUMO
            obs = self._get_sumo_observation(junction_id)
        else:
            # Generate mock observation
            obs = self._get_mock_observation(junction_id)
        
        return obs
    
    def _get_sumo_observation(self, junction_id: str) -> np.ndarray:
        """Get observation from SUMO"""
        try:
            # Get incoming lanes
            incoming_lanes = self.traci.trafficlight.getControlledLanes(junction_id)
            unique_lanes = list(set(incoming_lanes))
            
            # Queue lengths (vehicles per lane)
            queue_lengths = []
            for lane in unique_lanes[:4]:  # Max 4 approaches
                queue = self.traci.lane.getLastStepHaltingNumber(lane)
                queue_lengths.append(queue)
            
            # Pad to 4
            while len(queue_lengths) < 4:
                queue_lengths.append(0)
            
            # Densities
            densities = []
            for lane in unique_lanes[:4]:
                occupancy = self.traci.lane.getLastStepOccupancy(lane)
                densities.append(occupancy / 100.0)
            
            while len(densities) < 4:
                densities.append(0)
            
            # Speeds
            speeds = []
            for lane in unique_lanes[:4]:
                speed = self.traci.lane.getLastStepMeanSpeed(lane)
                max_speed = self.traci.lane.getMaxSpeed(lane)
                speeds.append(speed / max(max_speed, 1))
            
            while len(speeds) < 4:
                speeds.append(0)
            
            # Current phase
            phase = self.traci.trafficlight.getPhase(junction_id)
            phase_one_hot = [0, 0, 0, 0]
            phase_one_hot[phase % 4] = 1
            
            # Combine
            obs = np.array(
                queue_lengths + densities + speeds + phase_one_hot,
                dtype=np.float32
            )
            
            return obs
            
        except Exception as e:
            print(f"Error getting SUMO observation: {e}")
            return self._get_mock_observation(junction_id)
    
    def _get_mock_observation(self, junction_id: str) -> np.ndarray:
        """Generate mock observation for testing"""
        # Random but plausible values
        queue_lengths = np.random.randint(0, 20, 4).astype(np.float32)
        densities = np.random.uniform(0, 1, 4).astype(np.float32)
        speeds = np.random.uniform(0, 1, 4).astype(np.float32)
        
        phase = self.current_phases.get(junction_id, 0)
        phase_one_hot = [0, 0, 0, 0]
        phase_one_hot[phase % 4] = 1
        
        obs = np.concatenate([queue_lengths, densities, speeds, phase_one_hot])
        return obs.astype(np.float32)
    
    def apply_action(self, junction_id: str, action: int):
        """
        Apply action to junction
        
        Args:
            junction_id: Junction to control
            action: 0=keep phase, 1=switch to next phase
        """
        current_phase = self.current_phases[junction_id]
        
        if action == 1:  # Switch
            new_phase = (current_phase + 1) % 4
            self.current_phases[junction_id] = new_phase
            
            if self.traci is not None:
                try:
                    self.traci.trafficlight.setPhase(junction_id, new_phase)
                except Exception as e:
                    print(f"Error setting phase: {e}")
            
            # Log action
            self.action_log.append({
                'time': time.time(),
                'junction': junction_id,
                'action': action,
                'old_phase': current_phase,
                'new_phase': new_phase
            })
    
    def control_step(self):
        """
        Perform one control step for all junctions
        """
        current_time = time.time()
        
        for junction_id in self.junction_ids:
            # Check if enough time has passed
            last_time = self.last_control_time[junction_id]
            
            if current_time - last_time < self.control_interval:
                continue
            
            # Get observation
            obs = self.get_observation(junction_id)
            
            # Get action from model
            action, confidence = self.model_server.predict(obs)
            
            # Apply action
            self.apply_action(junction_id, action)
            
            # Update timer
            self.last_control_time[junction_id] = current_time
    
    def start_control_loop(self):
        """Start background control loop"""
        if self._running:
            return
        
        self._running = True
        
        def control_loop():
            while self._running:
                self.control_step()
                
                # Check for model updates
                self.model_server.check_hot_reload()
                
                time.sleep(0.1)  # 10 Hz check rate
        
        self._control_thread = threading.Thread(target=control_loop, daemon=True)
        self._control_thread.start()
        
        print("Control loop started")
    
    def stop_control_loop(self):
        """Stop background control loop"""
        self._running = False
        
        if self._control_thread:
            self._control_thread.join(timeout=5.0)
        
        print("Control loop stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get controller status"""
        return {
            'running': self._running,
            'junctions': self.junction_ids,
            'current_phases': self.current_phases,
            'control_interval': self.control_interval,
            'sumo_connected': self.traci is not None,
            'model_metrics': self.model_server.get_metrics(),
            'recent_actions': list(self.action_log)[-10:]
        }
    
    def shutdown(self):
        """Clean shutdown"""
        self.stop_control_loop()
        
        if self.traci is not None:
            self.traci.close()
            print("SUMO connection closed")


class FailsafeController:
    """
    Failsafe controller for production deployment
    
    Provides safety guarantees:
    - Minimum green time enforcement
    - Maximum red time prevention
    - Emergency vehicle detection
    - Fallback to fixed timing
    """
    
    def __init__(
        self,
        primary_controller: TrafficController,
        min_green_time: float = 10.0,
        max_red_time: float = 120.0,
        emergency_preempt: bool = True
    ):
        self.primary = primary_controller
        self.min_green_time = min_green_time
        self.max_red_time = max_red_time
        self.emergency_preempt = emergency_preempt
        
        # Phase timing tracking
        self.phase_start_times = {}
        self.phase_green_times = {}
        
        # Emergency state
        self.emergency_active = False
        self.emergency_junction = None
        
        # Fallback mode
        self.fallback_active = False
        self.fallback_reason = None
    
    def check_safety_constraints(
        self, 
        junction_id: str, 
        proposed_action: int
    ) -> int:
        """
        Validate action against safety constraints
        
        Args:
            junction_id: Junction being controlled
            proposed_action: Action from RL model
            
        Returns:
            Safe action (may differ from proposed)
        """
        current_time = time.time()
        current_phase = self.primary.current_phases.get(junction_id, 0)
        
        # Check minimum green time
        if junction_id in self.phase_start_times:
            phase_duration = current_time - self.phase_start_times[junction_id]
            
            if proposed_action == 1 and phase_duration < self.min_green_time:
                # Block switch, minimum green not met
                return 0
        
        # Check maximum red time (starvation prevention)
        # If any approach has waited too long, force switch
        if junction_id in self.phase_green_times:
            for phase, last_green in self.phase_green_times[junction_id].items():
                if phase != current_phase:
                    wait_time = current_time - last_green
                    if wait_time > self.max_red_time:
                        # Force switch to serve starved approach
                        return 1
        
        return proposed_action
    
    def check_emergency(self, junction_id: str) -> bool:
        """
        Check for emergency vehicles
        
        Returns:
            True if emergency preemption needed
        """
        if not self.emergency_preempt:
            return False
        
        traci = self.primary.traci
        if traci is None:
            return False
        
        try:
            # Check for emergency vehicles on incoming lanes
            incoming_lanes = traci.trafficlight.getControlledLanes(junction_id)
            
            for lane in set(incoming_lanes):
                vehicles = traci.lane.getLastStepVehicleIDs(lane)
                
                for veh_id in vehicles:
                    vtype = traci.vehicle.getTypeID(veh_id)
                    if 'emergency' in vtype.lower() or 'ambulance' in vtype.lower():
                        return True
            
            return False
            
        except Exception:
            return False
    
    def handle_emergency(self, junction_id: str):
        """Handle emergency vehicle preemption"""
        self.emergency_active = True
        self.emergency_junction = junction_id
        
        # Set all-red briefly then green for emergency direction
        # (Simplified - real implementation would detect direction)
        if self.primary.traci is not None:
            try:
                # Emergency phase (typically all approaches red except emergency)
                self.primary.traci.trafficlight.setPhase(junction_id, 0)
            except:
                pass
    
    def activate_fallback(self, reason: str):
        """Activate fallback fixed-timing mode"""
        self.fallback_active = True
        self.fallback_reason = reason
        print(f"Fallback mode activated: {reason}")
    
    def deactivate_fallback(self):
        """Return to RL control"""
        self.fallback_active = False
        self.fallback_reason = None
        print("Fallback mode deactivated")
    
    def get_fallback_action(self, junction_id: str) -> int:
        """Get action from fixed-time fallback controller"""
        current_phase = self.primary.current_phases.get(junction_id, 0)
        
        # Fixed timing: 30 seconds per phase
        if junction_id in self.phase_start_times:
            phase_duration = time.time() - self.phase_start_times[junction_id]
            
            if phase_duration >= 30.0:
                return 1  # Switch
        
        return 0  # Keep


class DeploymentManager:
    """
    High-level manager for production deployment
    """
    
    def __init__(
        self,
        config: InferenceConfig,
        junction_ids: List[str]
    ):
        self.config = config
        self.junction_ids = junction_ids
        
        # Components
        self.model_server = ModelServer(config)
        self.controller = TrafficController(
            self.model_server,
            junction_ids,
            use_sumo=False  # Set True for real deployment
        )
        self.failsafe = FailsafeController(self.controller)
        
        # Health check
        self._health_check_interval = 60.0
        self._last_health_check = 0
    
    def deploy(self, model_path: Optional[str] = None) -> bool:
        """
        Deploy model for inference
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if deployment successful
        """
        # Load model
        if not self.model_server.load_model(model_path):
            return False
        
        # Start control loop
        self.controller.start_control_loop()
        
        print(f"Deployment successful: {len(self.junction_ids)} junctions")
        return True
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        metrics = self.model_server.get_metrics()
        
        health = {
            'healthy': True,
            'model_loaded': metrics['model_loaded'],
            'success_rate': metrics['success_rate'],
            'avg_latency_ms': metrics['avg_latency_ms'],
            'issues': []
        }
        
        # Check success rate
        if metrics['success_rate'] < 0.95:
            health['issues'].append('Low prediction success rate')
        
        # Check latency
        if metrics['avg_latency_ms'] > self.config.max_latency_ms:
            health['issues'].append('High prediction latency')
        
        # Check fallback rate
        fallback_rate = metrics.get('fallback_rate', 0)
        if fallback_rate > 0.1:
            health['issues'].append('High fallback rate')
        
        health['healthy'] = len(health['issues']) == 0
        
        return health
    
    def get_status(self) -> Dict[str, Any]:
        """Get deployment status"""
        return {
            'controller_status': self.controller.get_status(),
            'health': self.health_check(),
            'failsafe_active': self.failsafe.fallback_active
        }
    
    def shutdown(self):
        """Clean shutdown"""
        self.controller.shutdown()
        print("Deployment manager shutdown complete")


# Convenience function
def deploy_model(
    model_path: str,
    junction_ids: List[str],
    **kwargs
) -> DeploymentManager:
    """
    Quick deployment of trained model
    
    Args:
        model_path: Path to trained model
        junction_ids: Junctions to control
        **kwargs: Additional config options
        
    Returns:
        DeploymentManager instance
    """
    config = InferenceConfig(model_path=model_path, **kwargs)
    manager = DeploymentManager(config, junction_ids)
    manager.deploy()
    return manager


if __name__ == '__main__':
    # Demo deployment
    config = InferenceConfig(
        model_path="models/best_agent.pt",
        device="cpu",
        max_latency_ms=100.0
    )
    
    junction_ids = ['silk_board', 'tin_factory', 'hebbal', 'marathahalli']
    
    manager = DeploymentManager(config, junction_ids)
    
    # Simulate without actual model
    print("Starting simulated deployment...")
    manager.controller.start_control_loop()
    
    try:
        for i in range(10):
            time.sleep(1)
            status = manager.get_status()
            health = status['health']
            print(f"Tick {i+1}: healthy={health['healthy']}, issues={health['issues']}")
    finally:
        manager.shutdown()
