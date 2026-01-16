"""
Logging and Monitoring System for Traffic Control

Provides:
- Structured logging for training and deployment
- Real-time metrics collection
- Alert system
- Performance monitoring dashboards
"""

import os
import sys
import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import deque
from enum import Enum
import numpy as np


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert data structure"""
    level: AlertLevel
    message: str
    metric_name: str = ""
    metric_value: float = 0
    threshold: float = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['level'] = self.level.value
        return d


@dataclass
class MetricConfig:
    """Configuration for a monitored metric"""
    name: str
    description: str = ""
    unit: str = ""
    
    # Alert thresholds
    warning_threshold: Optional[float] = None
    error_threshold: Optional[float] = None
    threshold_type: str = "upper"  # 'upper' or 'lower'
    
    # Aggregation
    window_size: int = 100
    aggregation: str = "mean"  # 'mean', 'max', 'min', 'sum'


class MetricsCollector:
    """
    Collects and aggregates metrics
    """
    
    def __init__(self, log_dir: str = "logs/metrics"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metric storage
        self.metrics: Dict[str, deque] = {}
        self.metric_configs: Dict[str, MetricConfig] = {}
        
        # Timestamps
        self.start_time = time.time()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Alerts
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Logging file
        self._setup_metric_logger()
    
    def _setup_metric_logger(self):
        """Setup metric logging to file"""
        self.metric_log_file = self.log_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    def register_metric(self, config: MetricConfig):
        """Register a metric for monitoring"""
        with self._lock:
            self.metric_configs[config.name] = config
            self.metrics[config.name] = deque(maxlen=config.window_size)
    
    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Record a metric value
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
        """
        timestamp = time.time()
        
        with self._lock:
            if name not in self.metrics:
                # Auto-register with defaults
                self.metrics[name] = deque(maxlen=100)
            
            self.metrics[name].append((timestamp, value))
        
        # Log to file
        log_entry = {
            'timestamp': timestamp,
            'name': name,
            'value': value,
            'tags': tags or {}
        }
        
        with open(self.metric_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Check thresholds
        self._check_thresholds(name, value)
    
    def _check_thresholds(self, name: str, value: float):
        """Check if value exceeds thresholds"""
        if name not in self.metric_configs:
            return
        
        config = self.metric_configs[name]
        
        # Warning threshold
        if config.warning_threshold is not None:
            exceeded = (
                value > config.warning_threshold 
                if config.threshold_type == "upper" 
                else value < config.warning_threshold
            )
            
            if exceeded:
                self._raise_alert(
                    AlertLevel.WARNING,
                    f"Metric {name} exceeded warning threshold",
                    name, value, config.warning_threshold
                )
        
        # Error threshold
        if config.error_threshold is not None:
            exceeded = (
                value > config.error_threshold 
                if config.threshold_type == "upper" 
                else value < config.error_threshold
            )
            
            if exceeded:
                self._raise_alert(
                    AlertLevel.ERROR,
                    f"Metric {name} exceeded error threshold",
                    name, value, config.error_threshold
                )
    
    def _raise_alert(
        self,
        level: AlertLevel,
        message: str,
        metric_name: str,
        value: float,
        threshold: float
    ):
        """Raise an alert"""
        alert = Alert(
            level=level,
            message=message,
            metric_name=metric_name,
            metric_value=value,
            threshold=threshold
        )
        
        self.alerts.append(alert)
        
        # Call handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"Alert handler error: {e}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler callback"""
        self.alert_handlers.append(handler)
    
    def get_metric(
        self, 
        name: str, 
        aggregation: str = "last"
    ) -> Optional[float]:
        """
        Get metric value
        
        Args:
            name: Metric name
            aggregation: 'last', 'mean', 'max', 'min', 'sum'
            
        Returns:
            Aggregated value
        """
        with self._lock:
            if name not in self.metrics or not self.metrics[name]:
                return None
            
            values = [v for _, v in self.metrics[name]]
            
            if aggregation == "last":
                return values[-1]
            elif aggregation == "mean":
                return np.mean(values)
            elif aggregation == "max":
                return np.max(values)
            elif aggregation == "min":
                return np.min(values)
            elif aggregation == "sum":
                return np.sum(values)
            else:
                return values[-1]
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get all metrics with various aggregations"""
        result = {}
        
        with self._lock:
            for name, values in self.metrics.items():
                if not values:
                    continue
                
                vals = [v for _, v in values]
                
                result[name] = {
                    'last': vals[-1],
                    'mean': np.mean(vals),
                    'max': np.max(vals),
                    'min': np.min(vals),
                    'count': len(vals)
                }
        
        return result
    
    def get_metric_history(
        self, 
        name: str, 
        start_time: Optional[float] = None
    ) -> List[Dict[str, float]]:
        """Get metric history"""
        with self._lock:
            if name not in self.metrics:
                return []
            
            history = []
            for ts, val in self.metrics[name]:
                if start_time and ts < start_time:
                    continue
                history.append({'timestamp': ts, 'value': val})
            
            return history
    
    def get_pending_alerts(
        self, 
        min_level: AlertLevel = AlertLevel.WARNING
    ) -> List[Alert]:
        """Get unacknowledged alerts"""
        level_order = [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]
        min_idx = level_order.index(min_level)
        
        return [
            a for a in self.alerts 
            if not a.acknowledged and level_order.index(a.level) >= min_idx
        ]


class TrainingLogger:
    """
    Structured logger for training
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = "experiment",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment directory
        self.exp_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.exp_dir.mkdir(exist_ok=True)
        
        # Setup Python logger
        self.logger = logging.getLogger(f"traffic_control.{experiment_name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.exp_dir / "training.log")
        file_handler.setLevel(file_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Metrics collector
        self.metrics = MetricsCollector(str(self.exp_dir / "metrics"))
        
        # Episode log
        self.episode_log_file = self.exp_dir / "episodes.jsonl"
        
        # Hyperparameters file
        self.hparams_file = self.exp_dir / "hyperparameters.json"
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters"""
        with open(self.hparams_file, 'w') as f:
            json.dump(hparams, f, indent=2, default=str)
        
        self.logger.info(f"Hyperparameters: {hparams}")
    
    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Log episode results"""
        log_data = {
            'episode': episode,
            'reward': reward,
            'length': length,
            'timestamp': time.time(),
            **(metrics or {})
        }
        
        # Write to file
        with open(self.episode_log_file, 'a') as f:
            f.write(json.dumps(log_data) + '\n')
        
        # Record metrics
        self.metrics.record('episode_reward', reward)
        self.metrics.record('episode_length', length)
        
        if metrics:
            for name, value in metrics.items():
                self.metrics.record(f'episode_{name}', value)
        
        # Log to console periodically
        if episode % 100 == 0:
            self.logger.info(
                f"Episode {episode}: reward={reward:.2f}, length={length}"
            )
    
    def log_step(
        self,
        step: int,
        metrics: Dict[str, float]
    ):
        """Log training step metrics"""
        for name, value in metrics.items():
            self.metrics.record(f'step_{name}', value)
        
        # Log periodically
        if step % 10000 == 0:
            self.logger.info(f"Step {step}: {metrics}")
    
    def log_evaluation(
        self,
        step: int,
        metrics: Dict[str, float]
    ):
        """Log evaluation results"""
        log_data = {
            'step': step,
            'timestamp': time.time(),
            **metrics
        }
        
        eval_file = self.exp_dir / "evaluations.jsonl"
        with open(eval_file, 'a') as f:
            f.write(json.dumps(log_data) + '\n')
        
        self.logger.info(f"Evaluation at step {step}: {metrics}")
    
    def log_model_saved(self, path: str, metrics: Dict[str, float]):
        """Log model checkpoint save"""
        self.logger.info(f"Model saved to {path} with metrics {metrics}")
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)


class PerformanceMonitor:
    """
    Monitors system and application performance
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        
        # Timing contexts
        self.timers: Dict[str, float] = {}
        
        # Register performance metrics
        self._register_metrics()
    
    def _register_metrics(self):
        """Register performance metrics"""
        configs = [
            MetricConfig(
                name="inference_latency_ms",
                description="Model inference latency",
                unit="ms",
                warning_threshold=50.0,
                error_threshold=100.0
            ),
            MetricConfig(
                name="step_time_ms",
                description="Environment step time",
                unit="ms",
                warning_threshold=100.0,
                error_threshold=500.0
            ),
            MetricConfig(
                name="memory_usage_mb",
                description="Memory usage",
                unit="MB",
                warning_threshold=4000.0,
                error_threshold=8000.0
            ),
            MetricConfig(
                name="gpu_utilization",
                description="GPU utilization",
                unit="%",
                warning_threshold=95.0
            )
        ]
        
        for config in configs:
            self.metrics.register_metric(config)
    
    def start_timer(self, name: str):
        """Start a timer"""
        self.timers[name] = time.perf_counter()
    
    def stop_timer(self, name: str) -> float:
        """
        Stop timer and record metric
        
        Returns:
            Elapsed time in milliseconds
        """
        if name not in self.timers:
            return 0
        
        elapsed_ms = (time.perf_counter() - self.timers[name]) * 1000
        self.metrics.record(f"{name}_ms", elapsed_ms)
        
        del self.timers[name]
        return elapsed_ms
    
    def time_context(self, name: str):
        """Context manager for timing"""
        return TimerContext(self, name)
    
    def record_memory_usage(self):
        """Record current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self.metrics.record("memory_usage_mb", memory_mb)
            return memory_mb
        except ImportError:
            return None
    
    def record_gpu_stats(self):
        """Record GPU statistics"""
        try:
            import torch
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                memory_cached = torch.cuda.memory_reserved() / (1024 * 1024)
                
                self.metrics.record("gpu_memory_allocated_mb", memory_allocated)
                self.metrics.record("gpu_memory_cached_mb", memory_cached)
                
                return {
                    'allocated_mb': memory_allocated,
                    'cached_mb': memory_cached
                }
        except ImportError:
            pass
        
        return None


class TimerContext:
    """Context manager for timing code blocks"""
    
    def __init__(self, monitor: PerformanceMonitor, name: str):
        self.monitor = monitor
        self.name = name
    
    def __enter__(self):
        self.monitor.start_timer(self.name)
        return self
    
    def __exit__(self, *args):
        self.monitor.stop_timer(self.name)


class HealthChecker:
    """
    Health check system for deployment
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        
        # Health check functions
        self.checks: Dict[str, Callable[[], bool]] = {}
        
        # Health history
        self.health_history: deque = deque(maxlen=100)
    
    def register_check(self, name: str, check_fn: Callable[[], bool]):
        """Register a health check function"""
        self.checks[name] = check_fn
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        all_healthy = True
        
        for name, check_fn in self.checks.items():
            try:
                healthy = check_fn()
                results[name] = {'healthy': healthy}
                
                if not healthy:
                    all_healthy = False
                    
            except Exception as e:
                results[name] = {'healthy': False, 'error': str(e)}
                all_healthy = False
        
        # Record
        health_record = {
            'timestamp': time.time(),
            'overall_healthy': all_healthy,
            'checks': results
        }
        
        self.health_history.append(health_record)
        self.metrics.record("health_check_passed", 1.0 if all_healthy else 0.0)
        
        return health_record
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary"""
        if not self.health_history:
            return {'status': 'unknown', 'checks_run': 0}
        
        latest = self.health_history[-1]
        
        # Calculate uptime
        healthy_count = sum(
            1 for h in self.health_history if h['overall_healthy']
        )
        uptime_pct = healthy_count / len(self.health_history) * 100
        
        return {
            'status': 'healthy' if latest['overall_healthy'] else 'unhealthy',
            'uptime_percent': uptime_pct,
            'last_check': latest,
            'checks_run': len(self.health_history)
        }


# Convenience functions
def setup_training_logger(
    experiment_name: str,
    log_dir: str = "logs"
) -> TrainingLogger:
    """Setup and return training logger"""
    return TrainingLogger(
        log_dir=log_dir,
        experiment_name=experiment_name
    )


def setup_metrics_with_alerts(
    log_dir: str = "logs/metrics"
) -> MetricsCollector:
    """Setup metrics collector with standard alert configs"""
    collector = MetricsCollector(log_dir)
    
    # Add console alert handler
    def console_alert_handler(alert: Alert):
        print(f"[ALERT {alert.level.value.upper()}] {alert.message}")
    
    collector.add_alert_handler(console_alert_handler)
    
    return collector


if __name__ == '__main__':
    # Demo
    logger = setup_training_logger("demo_experiment")
    
    # Log hyperparameters
    logger.log_hyperparameters({
        'algorithm': 'DQN',
        'learning_rate': 0.001,
        'gamma': 0.99,
        'batch_size': 64
    })
    
    # Simulate training
    for episode in range(10):
        reward = np.random.uniform(50, 150)
        length = np.random.randint(100, 500)
        
        logger.log_episode(
            episode=episode,
            reward=reward,
            length=length,
            metrics={
                'avg_wait_time': np.random.uniform(20, 40),
                'throughput': np.random.randint(100, 200)
            }
        )
    
    # Performance monitoring
    monitor = PerformanceMonitor(logger.metrics)
    
    with monitor.time_context("dummy_operation"):
        time.sleep(0.05)  # 50ms
    
    # Health check
    health = HealthChecker(logger.metrics)
    health.register_check("model_loaded", lambda: True)
    health.register_check("sumo_connected", lambda: True)
    
    result = health.run_checks()
    print(f"\nHealth check: {result}")
    
    # Get metrics summary
    print(f"\nMetrics summary: {logger.metrics.get_all_metrics()}")
