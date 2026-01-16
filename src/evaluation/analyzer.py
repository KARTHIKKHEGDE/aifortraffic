"""
Results Analyzer
Analysis and comparison of traffic control strategies
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .metrics import TrafficMetrics, compute_target_achievements
from ..utils.logger import setup_logger

logger = setup_logger("analyzer")


class ResultsAnalyzer:
    """
    Analyze and compare traffic control experiment results
    """
    
    def __init__(self, results_dir: str = "runs"):
        """
        Initialize results analyzer
        
        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.experiments = {}
        self.baselines = {}
        
        # Target improvements
        self.targets = {
            'waiting_time_improvement': 33.0,    # 33% reduction target
            'queue_length_improvement': 33.0,    # 33% reduction target
            'emergency_clearance_improvement': 67.0,  # 67% improvement target
            'throughput_improvement': 15.0,       # 15% improvement target
        }
    
    def load_experiment(self, experiment_path: str) -> Dict[str, Any]:
        """Load experiment results from file"""
        path = Path(experiment_path)
        
        if not path.exists():
            logger.error(f"Experiment file not found: {path}")
            return {}
        
        with open(path, 'r') as f:
            results = json.load(f)
        
        experiment_name = path.stem
        self.experiments[experiment_name] = results
        
        logger.info(f"Loaded experiment: {experiment_name}")
        return results
    
    def load_all_experiments(self) -> Dict[str, Dict]:
        """Load all experiments from results directory"""
        if not self.results_dir.exists():
            logger.warning(f"Results directory not found: {self.results_dir}")
            return {}
        
        for exp_dir in self.results_dir.iterdir():
            if exp_dir.is_dir():
                results_file = exp_dir / "training_results.json"
                if results_file.exists():
                    self.load_experiment(str(results_file))
        
        logger.info(f"Loaded {len(self.experiments)} experiments")
        return self.experiments
    
    def add_baseline_metrics(self, name: str, metrics: TrafficMetrics):
        """Add baseline metrics for comparison"""
        self.baselines[name] = metrics
    
    def compare_agents(
        self,
        agent_metrics: Dict[str, TrafficMetrics],
        baseline_name: str = 'fixed_time'
    ) -> pd.DataFrame:
        """
        Compare multiple agents against baseline
        
        Args:
            agent_metrics: Dict mapping agent name to TrafficMetrics
            baseline_name: Name of baseline to compare against
        
        Returns:
            DataFrame with comparison results
        """
        if baseline_name not in self.baselines:
            logger.error(f"Baseline '{baseline_name}' not found")
            return pd.DataFrame()
        
        baseline = self.baselines[baseline_name]
        baseline_summary = baseline.get_summary()
        
        rows = []
        
        for agent_name, metrics in agent_metrics.items():
            agent_summary = metrics.get_summary()
            comparison = metrics.compare_to_baseline(baseline)
            
            row = {
                'Agent': agent_name,
                'Avg Wait Time (s)': agent_summary.get('avg_waiting_time', 0),
                'Avg Queue Length': agent_summary.get('avg_queue_length', 0),
                'Throughput (veh/hr)': agent_summary.get('avg_throughput_per_hour', 0),
                'Emergency Clearance (s)': agent_summary.get('avg_emergency_clearance', 0),
                'Wait Time Δ (%)': comparison.get('avg_waiting_time_improvement_%', 0),
                'Queue Δ (%)': comparison.get('avg_queue_length_improvement_%', 0),
                'Throughput Δ (%)': comparison.get('avg_throughput_per_hour_improvement_%', 0),
            }
            rows.append(row)
        
        # Add baseline row
        rows.append({
            'Agent': baseline_name,
            'Avg Wait Time (s)': baseline_summary.get('avg_waiting_time', 0),
            'Avg Queue Length': baseline_summary.get('avg_queue_length', 0),
            'Throughput (veh/hr)': baseline_summary.get('avg_throughput_per_hour', 0),
            'Emergency Clearance (s)': baseline_summary.get('avg_emergency_clearance', 0),
            'Wait Time Δ (%)': 0,
            'Queue Δ (%)': 0,
            'Throughput Δ (%)': 0,
        })
        
        return pd.DataFrame(rows)
    
    def check_targets(
        self,
        agent_metrics: TrafficMetrics,
        baseline_metrics: TrafficMetrics
    ) -> Dict[str, Dict[str, Any]]:
        """
        Check if targets are met
        
        Args:
            agent_metrics: RL agent metrics
            baseline_metrics: Baseline controller metrics
        
        Returns:
            Target achievement report
        """
        return compute_target_achievements(
            agent_metrics,
            baseline_metrics,
            self.targets
        )
    
    def generate_report(
        self,
        agent_metrics: Dict[str, TrafficMetrics],
        baseline_metrics: Dict[str, TrafficMetrics],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            agent_metrics: Dict of agent metrics
            baseline_metrics: Dict of baseline metrics
            output_path: Optional path to save report
        
        Returns:
            Report as string
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("BANGALORE ADAPTIVE TRAFFIC SIGNAL CONTROL - EVALUATION REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        
        # Executive Summary
        report_lines.append("\n## EXECUTIVE SUMMARY\n")
        
        best_agent = None
        best_improvement = -float('inf')
        
        for agent_name, metrics in agent_metrics.items():
            if 'fixed_time' in baseline_metrics:
                comparison = metrics.compare_to_baseline(baseline_metrics['fixed_time'])
                wait_improvement = comparison.get('avg_waiting_time_improvement_%', 0)
                
                if wait_improvement > best_improvement:
                    best_improvement = wait_improvement
                    best_agent = agent_name
        
        if best_agent:
            report_lines.append(f"Best performing agent: {best_agent}")
            report_lines.append(f"Waiting time improvement: {best_improvement:.1f}%")
            report_lines.append(f"Target met: {'✓' if best_improvement >= self.targets['waiting_time_improvement'] else '✗'}")
        
        # Baseline Results
        report_lines.append("\n## BASELINE PERFORMANCE\n")
        
        for name, metrics in baseline_metrics.items():
            summary = metrics.get_summary()
            report_lines.append(f"### {name.upper()}")
            report_lines.append(f"  Average Waiting Time: {summary.get('avg_waiting_time', 0):.1f}s")
            report_lines.append(f"  Average Queue Length: {summary.get('avg_queue_length', 0):.1f}")
            report_lines.append(f"  Throughput: {summary.get('avg_throughput_per_hour', 0):.0f} veh/hr")
            report_lines.append(f"  Emergency Clearance: {summary.get('avg_emergency_clearance', 0):.1f}s")
            report_lines.append("")
        
        # Agent Results
        report_lines.append("\n## RL AGENT PERFORMANCE\n")
        
        for agent_name, metrics in agent_metrics.items():
            summary = metrics.get_summary()
            
            report_lines.append(f"### {agent_name.upper()}")
            report_lines.append(f"  Average Waiting Time: {summary.get('avg_waiting_time', 0):.1f}s")
            report_lines.append(f"  Average Queue Length: {summary.get('avg_queue_length', 0):.1f}")
            report_lines.append(f"  Throughput: {summary.get('avg_throughput_per_hour', 0):.0f} veh/hr")
            report_lines.append(f"  Average Reward: {summary.get('avg_episode_reward', 0):.2f}")
            
            # Comparison to fixed-time baseline
            if 'fixed_time' in baseline_metrics:
                comparison = metrics.compare_to_baseline(baseline_metrics['fixed_time'])
                report_lines.append(f"\n  Improvement vs Fixed-Time:")
                report_lines.append(f"    Waiting Time: {comparison.get('avg_waiting_time_improvement_%', 0):+.1f}%")
                report_lines.append(f"    Queue Length: {comparison.get('avg_queue_length_improvement_%', 0):+.1f}%")
                report_lines.append(f"    Throughput: {comparison.get('avg_throughput_per_hour_improvement_%', 0):+.1f}%")
            
            report_lines.append("")
        
        # Target Achievement
        report_lines.append("\n## TARGET ACHIEVEMENT\n")
        report_lines.append(f"Target Improvements:")
        report_lines.append(f"  Waiting Time: ≥{self.targets['waiting_time_improvement']}% reduction")
        report_lines.append(f"  Queue Length: ≥{self.targets['queue_length_improvement']}% reduction")
        report_lines.append(f"  Emergency Clearance: ≥{self.targets['emergency_clearance_improvement']}% faster")
        report_lines.append("")
        
        for agent_name, metrics in agent_metrics.items():
            if 'fixed_time' in baseline_metrics:
                achievements = self.check_targets(metrics, baseline_metrics['fixed_time'])
                
                report_lines.append(f"### {agent_name}")
                for target_name, result in achievements.items():
                    status = '✓ MET' if result['met'] else '✗ NOT MET'
                    report_lines.append(
                        f"  {target_name}: {result['achieved']:.1f}% ({status})"
                    )
                report_lines.append("")
        
        # Per-Junction Analysis
        report_lines.append("\n## PER-JUNCTION ANALYSIS\n")
        
        junctions = ['silk_board', 'tin_factory', 'hebbal', 'marathahalli']
        
        for junction in junctions:
            report_lines.append(f"### {junction.replace('_', ' ').title()}")
            report_lines.append("  (Junction-specific metrics would be populated here)")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("\n## RECOMMENDATIONS\n")
        
        if best_agent and best_improvement >= self.targets['waiting_time_improvement']:
            report_lines.append(f"1. Deploy {best_agent} agent for pilot testing")
            report_lines.append("2. Monitor emergency vehicle response times closely")
            report_lines.append("3. Consider weather-adaptive timing adjustments during monsoon")
        else:
            report_lines.append("1. Continue training with extended curriculum stages")
            report_lines.append("2. Consider fine-tuning reward weights")
            report_lines.append("3. Increase training episodes in high-traffic scenarios")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        # Save report
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to: {output_path}")
        
        return report
    
    def analyze_training_progress(
        self,
        experiment_name: str
    ) -> Dict[str, Any]:
        """
        Analyze training progress from experiment results
        
        Args:
            experiment_name: Name of experiment to analyze
        
        Returns:
            Analysis results
        """
        if experiment_name not in self.experiments:
            logger.error(f"Experiment '{experiment_name}' not found")
            return {}
        
        exp = self.experiments[experiment_name]
        stages = exp.get('stages', [])
        
        analysis = {
            'agent_type': exp.get('agent_type', 'unknown'),
            'total_episodes': exp.get('total_episodes', 0),
            'stages': [],
        }
        
        for stage in stages:
            stage_analysis = {
                'name': stage.get('stage_name', 'unknown'),
                'episodes': len(stage.get('episodes', [])),
                'avg_reward': np.mean(stage.get('rewards', [0])),
                'avg_waiting_time': np.mean(stage.get('waiting_times', [0])),
                'reward_trend': self._calculate_trend(stage.get('rewards', [])),
            }
            analysis['stages'].append(stage_analysis)
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 10:
            return 'insufficient_data'
        
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        improvement = (second_half - first_half) / abs(first_half) if first_half != 0 else 0
        
        if improvement > 0.1:
            return 'improving'
        elif improvement < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def export_to_csv(
        self,
        comparison_df: pd.DataFrame,
        output_path: str
    ):
        """Export comparison results to CSV"""
        comparison_df.to_csv(output_path, index=False)
        logger.info(f"Results exported to: {output_path}")
    
    def get_best_checkpoint(
        self,
        experiment_name: str,
        metric: str = 'reward'
    ) -> Optional[str]:
        """
        Find best checkpoint from experiment
        
        Args:
            experiment_name: Experiment name
            metric: Metric to optimize ('reward', 'waiting_time')
        
        Returns:
            Path to best checkpoint
        """
        if experiment_name not in self.experiments:
            return None
        
        exp = self.experiments[experiment_name]
        exp_dir = self.results_dir / experiment_name
        
        # Find checkpoints
        checkpoints = list(exp_dir.glob("checkpoint_*.pt")) + list(exp_dir.glob("checkpoint_*.zip"))
        
        if not checkpoints:
            # Return final model
            final_model = list(exp_dir.glob("final_model_*"))
            return str(final_model[0]) if final_model else None
        
        # For now, return latest checkpoint
        return str(sorted(checkpoints)[-1])


def create_comparison_summary(
    rl_results: Dict[str, float],
    fixed_time_results: Dict[str, float],
    actuated_results: Dict[str, float]
) -> pd.DataFrame:
    """
    Create summary comparison table
    
    Args:
        rl_results: RL agent results
        fixed_time_results: Fixed-time controller results
        actuated_results: Actuated controller results
    
    Returns:
        Summary DataFrame
    """
    metrics = [
        'avg_waiting_time',
        'avg_queue_length',
        'avg_throughput_per_hour',
        'avg_emergency_clearance'
    ]
    
    metric_names = [
        'Avg Waiting Time (s)',
        'Avg Queue Length',
        'Throughput (veh/hr)',
        'Emergency Clearance (s)'
    ]
    
    data = {
        'Metric': metric_names,
        'RL Agent': [rl_results.get(m, 0) for m in metrics],
        'Fixed-Time': [fixed_time_results.get(m, 0) for m in metrics],
        'Actuated': [actuated_results.get(m, 0) for m in metrics],
    }
    
    df = pd.DataFrame(data)
    
    # Add improvement columns
    df['RL vs Fixed (%)'] = [
        (fixed_time_results.get(m, 1) - rl_results.get(m, 0)) / fixed_time_results.get(m, 1) * 100
        if m in ['avg_waiting_time', 'avg_queue_length', 'avg_emergency_clearance']
        else (rl_results.get(m, 0) - fixed_time_results.get(m, 1)) / fixed_time_results.get(m, 1) * 100
        for m in metrics
    ]
    
    return df


if __name__ == "__main__":
    print("Testing ResultsAnalyzer...")
    print("=" * 50)
    
    # Create analyzer
    analyzer = ResultsAnalyzer()
    
    # Create dummy metrics for testing
    from .metrics import TrafficMetrics
    
    agent_metrics = TrafficMetrics()
    baseline_metrics = TrafficMetrics()
    
    # Simulate some data
    for _ in range(100):
        agent_metrics.add_step_metrics(
            queue_lengths={'lane_1': np.random.randint(5, 15)},
            waiting_times={'lane_1': np.random.uniform(20, 40)},
            speeds={'lane_1': np.random.uniform(8, 12)}
        )
        
        baseline_metrics.add_step_metrics(
            queue_lengths={'lane_1': np.random.randint(10, 25)},
            waiting_times={'lane_1': np.random.uniform(40, 80)},
            speeds={'lane_1': np.random.uniform(5, 10)}
        )
    
    agent_metrics.add_episode_metrics(100, 85.0)
    baseline_metrics.add_episode_metrics(100, 45.0)
    
    # Add baseline
    analyzer.add_baseline_metrics('fixed_time', baseline_metrics)
    
    # Compare
    comparison = agent_metrics.compare_to_baseline(baseline_metrics)
    print("\nComparison to baseline:")
    for key, value in comparison.items():
        print(f"  {key}: {value:.1f}%")
    
    # Check targets
    achievements = analyzer.check_targets(agent_metrics, baseline_metrics)
    print("\nTarget achievements:")
    for target, result in achievements.items():
        status = "✓" if result['met'] else "✗"
        print(f"  {status} {target}: {result['achieved']:.1f}% (target: {result['target']}%)")
    
    # Generate report
    report = analyzer.generate_report(
        {'DQN': agent_metrics},
        {'fixed_time': baseline_metrics}
    )
    print("\nGenerated report preview (first 500 chars):")
    print(report[:500])
    
    print("\nResultsAnalyzer test complete!")
