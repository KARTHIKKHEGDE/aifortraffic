"""
Evaluation Package
Metrics, baselines, analysis, and statistical testing tools
"""

from .metrics import TrafficMetrics, compute_metrics
from .baselines import FixedTimeController, ActuatedController, run_baseline
from .analyzer import ResultsAnalyzer
from .statistical_testing import (
    StatisticalTester,
    StatisticalResult,
    ExperimentAnalyzer,
    compare_agents,
    is_significantly_better,
)

__all__ = [
    "TrafficMetrics",
    "compute_metrics",
    "FixedTimeController",
    "ActuatedController",
    "run_baseline",
    "ResultsAnalyzer",
    "StatisticalTester",
    "StatisticalResult",
    "ExperimentAnalyzer",
    "compare_agents",
    "is_significantly_better",
]

