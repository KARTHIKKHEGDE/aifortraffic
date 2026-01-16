"""
Logging Module
Provides consistent logging across the project
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Try to import colorama for colored output on Windows
try:
    from colorama import init, Fore, Style
    init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    if COLORAMA_AVAILABLE:
        COLORS = {
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Style.BRIGHT,
        }
        RESET = Style.RESET_ALL
    else:
        COLORS = {}
        RESET = ''
    
    def __init__(self, fmt: str = None, datefmt: str = None):
        super().__init__(fmt, datefmt)
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        return super().format(record)


def setup_logger(
    name: str = "bangalore_traffic",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with consistent formatting
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file (optional)
        console: Whether to output to console
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Log format
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = ColoredFormatter(log_format, date_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "bangalore_traffic") -> logging.Logger:
    """Get an existing logger by name"""
    return logging.getLogger(name)


def create_experiment_logger(experiment_name: str) -> logging.Logger:
    """
    Create a logger for a specific experiment run
    
    Args:
        experiment_name: Name of the experiment
    
    Returns:
        Logger configured with experiment-specific log file
    """
    from .config import get_project_root
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = get_project_root() / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    logger = setup_logger(
        name=experiment_name,
        level=logging.DEBUG,
        log_file=str(log_file),
        console=True
    )
    
    logger.info(f"Experiment started: {experiment_name}")
    logger.info(f"Log file: {log_file}")
    
    return logger


class TrainingLogger:
    """
    Specialized logger for training progress
    Provides structured logging for RL training metrics
    """
    
    def __init__(self, name: str = "training"):
        self.logger = setup_logger(name)
        self.episode_count = 0
        self.step_count = 0
        
    def log_episode(
        self,
        episode: int,
        reward: float,
        avg_waiting_time: float,
        avg_queue_length: float,
        throughput: int,
        epsilon: float = None
    ):
        """Log episode summary"""
        self.episode_count = episode
        
        msg = (
            f"Episode {episode:5d} | "
            f"Reward: {reward:8.2f} | "
            f"Wait: {avg_waiting_time:6.1f}s | "
            f"Queue: {avg_queue_length:5.1f} | "
            f"Throughput: {throughput:4d}"
        )
        
        if epsilon is not None:
            msg += f" | ε: {epsilon:.3f}"
        
        self.logger.info(msg)
    
    def log_step(
        self,
        step: int,
        action: int,
        reward: float,
        state_summary: dict = None
    ):
        """Log individual step (debug level)"""
        self.step_count = step
        
        msg = f"Step {step:6d} | Action: {action} | Reward: {reward:6.2f}"
        
        if state_summary:
            msg += f" | {state_summary}"
        
        self.logger.debug(msg)
    
    def log_emergency(self, vehicle_id: str, junction: str, action: str):
        """Log emergency vehicle events"""
        self.logger.warning(
            f"🚑 EMERGENCY | Vehicle: {vehicle_id} | "
            f"Junction: {junction} | Action: {action}"
        )
    
    def log_weather_change(self, old_state: int, new_state: int):
        """Log weather state changes"""
        weather_names = {0: "Clear", 1: "Rain"}
        self.logger.info(
            f"🌧️ WEATHER CHANGE | {weather_names.get(old_state, 'Unknown')} → "
            f"{weather_names.get(new_state, 'Unknown')}"
        )
    
    def log_training_progress(
        self,
        current_timestep: int,
        total_timesteps: int,
        fps: float,
        loss: float = None
    ):
        """Log overall training progress"""
        progress = (current_timestep / total_timesteps) * 100
        
        msg = (
            f"Progress: {progress:5.1f}% | "
            f"Steps: {current_timestep:,}/{total_timesteps:,} | "
            f"FPS: {fps:.0f}"
        )
        
        if loss is not None:
            msg += f" | Loss: {loss:.4f}"
        
        self.logger.info(msg)
    
    def log_evaluation(
        self,
        controller_name: str,
        metrics: dict
    ):
        """Log evaluation results"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"EVALUATION RESULTS - {controller_name}")
        self.logger.info(f"{'='*60}")
        
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {metric_name}: {value:.3f}")
            else:
                self.logger.info(f"  {metric_name}: {value}")
        
        self.logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    # Test logging
    print("Testing logging module...")
    
    logger = setup_logger("test", level=logging.DEBUG)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test training logger
    training_logger = TrainingLogger("training_test")
    training_logger.log_episode(
        episode=100,
        reward=245.5,
        avg_waiting_time=45.2,
        avg_queue_length=12.3,
        throughput=150,
        epsilon=0.1
    )
    
    training_logger.log_emergency("ambulance_001", "silk_board", "GREEN_OVERRIDE")
    training_logger.log_weather_change(0, 1)
    
    print("\nLogging test complete!")
