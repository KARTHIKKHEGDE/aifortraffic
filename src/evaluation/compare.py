"""
RL vs Baseline Comparison System
=================================
Comprehensive comparison of RL agent vs fixed-time baseline.

Features:
- Statistical comparison (t-tests, effect sizes)
- Hourly performance breakdown
- Peak hour analysis
- Improvement metrics
- Publication-quality visualizations

Output:
- Comparison report (TXT)
- Statistical analysis (CSV)
- Visualization plots (PNG)

Usage:
    python compare.py
    python compare.py --rl results/rl_evaluation.csv --baseline results/baseline_evaluation.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Tuple
import argparse

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


# ============================================
# COMPARISON ENGINE
# ============================================

class PerformanceComparator:
    """
    Compares RL agent vs fixed-time baseline with statistical rigor.
    
    Metrics:
    - Absolute improvement
    - Percentage improvement
    - Statistical significance (t-test)
    - Effect size (Cohen's d)
    """
    
    def __init__(self, rl_results_csv: str, baseline_results_csv: str):
        """
        Initialize comparator.
        
        Args:
            rl_results_csv: Path to RL evaluation results
            baseline_results_csv: Path to baseline evaluation results
        """
        # Load data
        self.rl_df = pd.read_csv(rl_results_csv)
        self.baseline_df = pd.read_csv(baseline_results_csv)
        
        # Validate
        if len(self.rl_df) != len(self.baseline_df):
            raise ValueError("RL and baseline results have different number of hours")
        
        if not all(self.rl_df['hour'] == self.baseline_df['hour']):
            raise ValueError("Hour mismatch between RL and baseline results")
        
        print(f"✓ Loaded RL results: {len(self.rl_df)} hours")
        print(f"✓ Loaded baseline results: {len(self.baseline_df)} hours")
    
    def calculate_improvements(self) -> pd.DataFrame:
        """
        Calculate improvement metrics for all hours.
        
        Returns:
            DataFrame with improvement statistics
        """
        comparison = pd.DataFrame()
        comparison['hour'] = self.rl_df['hour']
        
        # Key metrics to compare
        metrics = {
            'avg_queue_length': 'lower_better',
            'avg_vehicle_delay': 'lower_better',
            'total_throughput': 'higher_better',
            'queue_imbalance': 'lower_better',
        }
        
        for metric, direction in metrics.items():
            rl_mean = self.rl_df[f'{metric}_mean']
            baseline_mean = self.baseline_df[f'{metric}_mean']
            
            # Absolute difference
            diff = rl_mean - baseline_mean
            
            # Percentage improvement (note: negative = improvement for "lower_better")
            if direction == 'lower_better':
                pct_improvement = ((baseline_mean - rl_mean) / baseline_mean) * 100
            else:
                pct_improvement = ((rl_mean - baseline_mean) / baseline_mean) * 100
            
            comparison[f'{metric}_rl'] = rl_mean
            comparison[f'{metric}_baseline'] = baseline_mean
            comparison[f'{metric}_diff'] = diff
            comparison[f'{metric}_improvement_pct'] = pct_improvement
        
        return comparison
    
    def statistical_tests(self) -> Dict:
        """
        Perform statistical significance tests.
        
        Returns:
            Dictionary with test results
        """
        results = {}
        
        metrics = ['avg_queue_length', 'avg_vehicle_delay', 'total_throughput', 'queue_imbalance']
        
        for metric in metrics:
            rl_values = self.rl_df[f'{metric}_mean'].values
            baseline_values = self.baseline_df[f'{metric}_mean'].values
            
            # Paired t-test (same hours for both methods)
            t_stat, p_value = stats.ttest_rel(rl_values, baseline_values)
            
            # Effect size (Cohen's d)
            diff = rl_values - baseline_values
            pooled_std = np.sqrt((rl_values.std()**2 + baseline_values.std()**2) / 2)
            cohens_d = diff.mean() / pooled_std
            
            results[metric] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05,
                'effect_size': self._interpret_cohens_d(cohens_d)
            }
        
        return results
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"
    
    def generate_report(self, output_dir: str = "results"):
        """
        Generate comprehensive comparison report.
        
        Args:
            output_dir: Directory to save report
        """
        os.makedirs(output_dir, exist_ok=True)
        report_file = f"{output_dir}/comparison_report.txt"
        
        # Calculate improvements
        comparison_df = self.calculate_improvements()
        
        # Statistical tests
        stat_tests = self.statistical_tests()
        
        # Write report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 90 + "\n")
            f.write("RL AGENT vs FIXED-TIME BASELINE - COMPREHENSIVE COMPARISON\n")
            f.write("=" * 90 + "\n\n")
            
            # Overall summary
            f.write("OVERALL PERFORMANCE IMPROVEMENT:\n")
            f.write("-" * 90 + "\n")
            
            for metric in ['avg_queue_length', 'avg_vehicle_delay', 'total_throughput', 'queue_imbalance']:
                rl_avg = comparison_df[f'{metric}_rl'].mean()
                baseline_avg = comparison_df[f'{metric}_baseline'].mean()
                improvement = comparison_df[f'{metric}_improvement_pct'].mean()
                
                # Get statistical test results
                test = stat_tests[metric]
                sig_marker = "***" if test['p_value'] < 0.001 else "**" if test['p_value'] < 0.01 else "*" if test['p_value'] < 0.05 else ""
                
                f.write(f"\n{metric.replace('_', ' ').title()}:\n")
                f.write(f"  RL Agent:        {rl_avg:>8.2f}\n")
                f.write(f"  Fixed-Time:      {baseline_avg:>8.2f}\n")
                f.write(f"  Improvement:     {improvement:>7.1f}% {sig_marker}\n")
                f.write(f"  Significance:    p={test['p_value']:.4f} (Cohen's d={test['cohens_d']:.2f}, {test['effect_size']})\n")
            
            f.write("\n" + "-" * 90 + "\n")
            f.write("Significance levels: *** p<0.001, ** p<0.01, * p<0.05\n\n")
            
            # Peak hour analysis
            f.write("PEAK HOUR ANALYSIS:\n")
            f.write("-" * 90 + "\n")
            
            peak_hours = [8, 18]  # Morning and evening
            for hour in peak_hours:
                hour_data = comparison_df[comparison_df['hour'] == hour].iloc[0]
                period = "Morning" if hour == 8 else "Evening"
                
                f.write(f"\n{period} Peak ({hour:02d}:00):\n")
                f.write(f"  Queue Length:    RL={hour_data['avg_queue_length_rl']:>6.2f}  Fixed={hour_data['avg_queue_length_baseline']:>6.2f}  "
                       f"Δ={hour_data['avg_queue_length_improvement_pct']:>5.1f}%\n")
                f.write(f"  Vehicle Delay:   RL={hour_data['avg_vehicle_delay_rl']:>6.2f}s Fixed={hour_data['avg_vehicle_delay_baseline']:>6.2f}s "
                       f"Δ={hour_data['avg_vehicle_delay_improvement_pct']:>5.1f}%\n")
            
            # Best/worst hours
            f.write("\n\nBEST IMPROVEMENT HOUR:\n")
            f.write("-" * 90 + "\n")
            best_hour = comparison_df.loc[comparison_df['avg_queue_length_improvement_pct'].idxmax()]
            f.write(f"  Hour: {int(best_hour['hour']):02d}:00\n")
            f.write(f"  Queue improvement: {best_hour['avg_queue_length_improvement_pct']:.1f}%\n")
            f.write(f"  Delay improvement: {best_hour['avg_vehicle_delay_improvement_pct']:.1f}%\n")
            
            f.write("\n\nWORST IMPROVEMENT HOUR:\n")
            f.write("-" * 90 + "\n")
            worst_hour = comparison_df.loc[comparison_df['avg_queue_length_improvement_pct'].idxmin()]
            f.write(f"  Hour: {int(worst_hour['hour']):02d}:00\n")
            f.write(f"  Queue improvement: {worst_hour['avg_queue_length_improvement_pct']:.1f}%\n")
            f.write(f"  Delay improvement: {worst_hour['avg_vehicle_delay_improvement_pct']:.1f}%\n")
            
            f.write("\n" + "=" * 90 + "\n")
        
        # Save detailed comparison
        comparison_df.to_csv(f"{output_dir}/comparison_detailed.csv", index=False)
        
        print(f"📄 Report saved: {report_file}")
        print(f"📊 Detailed comparison: {output_dir}/comparison_detailed.csv")
        
        return comparison_df, stat_tests
    
    def create_visualizations(self, output_dir: str = "results"):
        """
        Create publication-quality comparison plots.
        
        Args:
            output_dir: Directory to save plots
        """
        plots_dir = f"{output_dir}/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        comparison_df = self.calculate_improvements()
        
        # Plot 1: Hourly comparison (Queue Length)
        self._plot_hourly_comparison(comparison_df, plots_dir)
        
        # Plot 2: Improvement heatmap
        self._plot_improvement_heatmap(comparison_df, plots_dir)
        
        # Plot 3: Statistical summary
        self._plot_statistical_summary(comparison_df, plots_dir)
        
        # Plot 4: Peak hour comparison
        self._plot_peak_hour_comparison(comparison_df, plots_dir)
        
        print(f"📊 Visualizations saved to: {plots_dir}/")
    
    def _plot_hourly_comparison(self, df: pd.DataFrame, output_dir: str):
        """Plot hourly queue length comparison."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = df['hour']
        ax.plot(x, df['avg_queue_length_baseline'], 'o-', label='Fixed-Time', linewidth=2, markersize=6, color='#d62728')
        ax.plot(x, df['avg_queue_length_rl'], 's-', label='RL Agent', linewidth=2, markersize=6, color='#2ca02c')
        
        # Shade peak hours
        ax.axvspan(7, 10, alpha=0.1, color='orange', label='Morning Peak')
        ax.axvspan(17, 20, alpha=0.1, color='purple', label='Evening Peak')
        
        ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Queue Length (vehicles)', fontsize=12, fontweight='bold')
        ax.set_title('Hourly Traffic Performance: RL Agent vs Fixed-Time Control', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 2))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hourly_queue_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_improvement_heatmap(self, df: pd.DataFrame, output_dir: str):
        """Plot improvement percentage heatmap."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = [
            ('avg_queue_length_improvement_pct', 'Queue Length Improvement (%)'),
            ('avg_vehicle_delay_improvement_pct', 'Vehicle Delay Improvement (%)'),
            ('total_throughput_improvement_pct', 'Throughput Improvement (%)'),
            ('queue_imbalance_improvement_pct', 'Fairness Improvement (%)')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Create bar plot
            colors = ['green' if x > 0 else 'red' for x in df[metric]]
            ax.bar(df['hour'], df[metric], color=colors, alpha=0.7, edgecolor='black')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax.set_xlabel('Hour of Day', fontsize=10)
            ax.set_ylabel('Improvement (%)', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xticks(range(0, 24, 3))
        
        plt.suptitle('Performance Improvement: RL vs Fixed-Time (Positive = Better)', 
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/improvement_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_summary(self, df: pd.DataFrame, output_dir: str):
        """Plot statistical summary with confidence intervals."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['Queue Length', 'Vehicle Delay', 'Throughput', 'Queue Imbalance']
        metric_cols = ['avg_queue_length', 'avg_vehicle_delay', 'total_throughput', 'queue_imbalance']
        
        rl_means = [df[f'{m}_rl'].mean() for m in metric_cols]
        baseline_means = [df[f'{m}_baseline'].mean() for m in metric_cols]
        rl_stds = [df[f'{m}_rl'].std() for m in metric_cols]
        baseline_stds = [df[f'{m}_baseline'].std() for m in metric_cols]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        # Normalize for comparison (scale each metric 0-100)
        for i in range(len(metrics)):
            max_val = max(rl_means[i], baseline_means[i])
            rl_means[i] = (rl_means[i] / max_val) * 100
            baseline_means[i] = (baseline_means[i] / max_val) * 100
            rl_stds[i] = (rl_stds[i] / max_val) * 100
            baseline_stds[i] = (baseline_stds[i] / max_val) * 100
        
        ax.bar(x - width/2, baseline_means, width, label='Fixed-Time', 
               yerr=baseline_stds, capsize=5, color='#d62728', alpha=0.7)
        ax.bar(x + width/2, rl_means, width, label='RL Agent', 
               yerr=rl_stds, capsize=5, color='#2ca02c', alpha=0.7)
        
        ax.set_ylabel('Normalized Performance (lower = better)', fontsize=11)
        ax.set_title('Statistical Summary: RL vs Baseline (with std dev)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/statistical_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_peak_hour_comparison(self, df: pd.DataFrame, output_dir: str):
        """Focused comparison for peak hours."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        peak_hours = [8, 18]
        peak_names = ['Morning Peak (08:00)', 'Evening Peak (18:00)']
        
        for idx, (hour, name) in enumerate(zip(peak_hours, peak_names)):
            ax = axes[idx]
            hour_data = df[df['hour'] == hour].iloc[0]
            
            metrics = ['Queue\nLength', 'Vehicle\nDelay', 'Throughput', 'Imbalance']
            metric_cols = ['avg_queue_length', 'avg_vehicle_delay', 'total_throughput', 'queue_imbalance']
            
            rl_vals = [hour_data[f'{m}_rl'] for m in metric_cols]
            baseline_vals = [hour_data[f'{m}_baseline'] for m in metric_cols]
            
            # Normalize
            normalized_rl = []
            normalized_baseline = []
            for i in range(len(metrics)):
                max_val = max(rl_vals[i], baseline_vals[i])
                normalized_rl.append((rl_vals[i] / max_val) * 100)
                normalized_baseline.append((baseline_vals[i] / max_val) * 100)
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax.bar(x - width/2, normalized_baseline, width, label='Fixed-Time', color='#d62728', alpha=0.7)
            ax.bar(x + width/2, normalized_rl, width, label='RL Agent', color='#2ca02c', alpha=0.7)
            
            ax.set_ylabel('Normalized Value', fontsize=10)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, fontsize=9)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Peak Hour Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/peak_hour_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()


# ============================================
# COMMAND-LINE INTERFACE
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare RL agent vs fixed-time baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--rl",
        type=str,
        default="results/rl_evaluation.csv",
        help="Path to RL evaluation results"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="results/baseline_evaluation.csv",
        help="Path to baseline evaluation results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip visualization generation"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.rl):
        print(f"❌ Error: RL results not found: {args.rl}")
        print(f"   Run: python evaluate.py --model models/best_model.zip")
        sys.exit(1)
    
    if not os.path.exists(args.baseline):
        print(f"❌ Error: Baseline results not found: {args.baseline}")
        print(f"   Run: python baseline.py")
        sys.exit(1)
    
    print("=" * 90)
    print("🔬 COMPREHENSIVE COMPARISON: RL vs FIXED-TIME")
    print("=" * 90)
    
    # Create comparator
    comparator = PerformanceComparator(args.rl, args.baseline)
    
    # Generate report
    print("\n📊 Generating comparison report...")
    comparison_df, stat_tests = comparator.generate_report(args.output)
    
    # Create visualizations
    if not args.no_plots:
        print("\n📈 Creating visualizations...")
        comparator.create_visualizations(args.output)
    
    # Print key findings
    print("\n" + "=" * 90)
    print("✅ COMPARISON COMPLETE - KEY FINDINGS:")
    print("=" * 90)
    
    for metric in ['avg_queue_length', 'avg_vehicle_delay']:
        improvement = comparison_df[f'{metric}_improvement_pct'].mean()
        test = stat_tests[metric]
        sig = "SIGNIFICANT" if test['significant'] else "not significant"
        
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Improvement: {improvement:+.1f}%")
        print(f"  Statistical significance: {sig} (p={test['p_value']:.4f})")
        print(f"  Effect size: {test['effect_size']} (Cohen's d={test['cohens_d']:.2f})")
    
    print("\n" + "=" * 90)
    print(f"\nFull report: {args.output}/comparison_report.txt")
    if not args.no_plots:
        print(f"Visualizations: {args.output}/plots/")
    print("=" * 90)


if __name__ == "__main__":
    main()