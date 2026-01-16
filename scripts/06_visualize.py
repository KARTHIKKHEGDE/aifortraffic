"""
Visualization Script
Generate plots and visualizations for traffic control results
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("visualize")

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Install with: pip install matplotlib")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class TrafficVisualizer:
    """
    Visualization tools for traffic control experiments
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for visualization")
        
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        if SEABORN_AVAILABLE:
            sns.set_palette("husl")
        
        # Color scheme
        self.colors = {
            'rl_agent': '#2ecc71',      # Green
            'fixed_time': '#e74c3c',     # Red
            'actuated': '#3498db',       # Blue
            'dqn': '#9b59b6',            # Purple
            'ppo': '#f39c12',            # Orange
            'qlearning': '#1abc9c',      # Teal
        }
        
        self.figsize = (12, 8)
    
    def plot_training_curves(
        self,
        history: Dict[str, List],
        title: str = "Training Progress",
        save_path: Optional[str] = None
    ):
        """
        Plot training curves (rewards, loss, epsilon)
        
        Args:
            history: Training history dict
            title: Plot title
            save_path: Optional path to save figure
        """
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # Episode rewards
        ax1 = fig.add_subplot(gs[0, 0])
        if 'episode_rewards' in history:
            rewards = history['episode_rewards']
            ax1.plot(rewards, alpha=0.3, color='blue', label='Raw')
            
            # Smoothed rewards
            window = min(50, len(rewards) // 10)
            if window > 1:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(rewards)), smoothed, color='blue', linewidth=2, label='Smoothed')
            
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title('Episode Rewards')
            ax1.legend()
        
        # Waiting times
        ax2 = fig.add_subplot(gs[0, 1])
        if 'avg_waiting_times' in history:
            waiting_times = history['avg_waiting_times']
            ax2.plot(waiting_times, alpha=0.3, color='red', label='Raw')
            
            window = min(50, len(waiting_times) // 10)
            if window > 1:
                smoothed = np.convolve(waiting_times, np.ones(window)/window, mode='valid')
                ax2.plot(range(window-1, len(waiting_times)), smoothed, color='red', linewidth=2, label='Smoothed')
            
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Waiting Time (s)')
            ax2.set_title('Average Waiting Time')
            ax2.legend()
        
        # Epsilon (exploration)
        ax3 = fig.add_subplot(gs[1, 0])
        if 'epsilons' in history:
            ax3.plot(history['epsilons'], color='green', linewidth=2)
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Epsilon')
            ax3.set_title('Exploration Rate')
            ax3.axhline(y=0.01, color='gray', linestyle='--', label='Min Epsilon')
            ax3.legend()
        
        # Loss
        ax4 = fig.add_subplot(gs[1, 1])
        if 'losses' in history:
            losses = [l for l in history['losses'] if l > 0]
            if losses:
                ax4.plot(losses, alpha=0.3, color='purple', label='Raw')
                
                window = min(50, len(losses) // 10)
                if window > 1:
                    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                    ax4.plot(range(window-1, len(losses)), smoothed, color='purple', linewidth=2, label='Smoothed')
                
                ax4.set_xlabel('Episode')
                ax4.set_ylabel('Loss')
                ax4.set_title('Training Loss')
                ax4.legend()
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training curves saved to: {save_path}")
        
        plt.show()
    
    def plot_comparison_bars(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        title: str = "Performance Comparison",
        save_path: Optional[str] = None
    ):
        """
        Plot bar chart comparing different methods
        
        Args:
            results: Dict mapping method name to metrics
            metrics: List of metrics to plot
            title: Plot title
            save_path: Optional path to save figure
        """
        if metrics is None:
            metrics = ['avg_waiting_time', 'avg_queue_length', 'avg_speed', 'avg_throughput_per_hour']
        
        metric_names = {
            'avg_waiting_time': 'Avg Waiting\nTime (s)',
            'avg_queue_length': 'Avg Queue\nLength',
            'avg_speed': 'Avg Speed\n(m/s)',
            'avg_throughput_per_hour': 'Throughput\n(veh/hr)',
            'avg_emergency_clearance': 'Emergency\nClearance (s)'
        }
        
        methods = list(results.keys())
        n_methods = len(methods)
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            values = [results[m].get(metric, 0) for m in methods]
            colors = [self.colors.get(m.lower().replace('-', '_'), 'gray') for m in methods]
            
            bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor='black')
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel(metric_names.get(metric, metric))
            ax.set_title(metric_names.get(metric, metric).replace('\n', ' '))
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Comparison chart saved to: {save_path}")
        
        plt.show()
    
    def plot_improvement_radar(
        self,
        improvements: Dict[str, float],
        title: str = "Improvement vs Baseline",
        save_path: Optional[str] = None
    ):
        """
        Plot radar chart of improvements
        
        Args:
            improvements: Dict of metric name to improvement percentage
            title: Plot title
            save_path: Optional path to save figure
        """
        labels = list(improvements.keys())
        values = list(improvements.values())
        
        # Number of variables
        N = len(labels)
        
        # Compute angle for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the loop
        
        values += values[:1]  # Complete the loop
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Plot data
        ax.plot(angles, values, 'o-', linewidth=2, color=self.colors['rl_agent'])
        ax.fill(angles, values, alpha=0.25, color=self.colors['rl_agent'])
        
        # Add target line (33%)
        target = [33] * (N + 1)
        ax.plot(angles, target, '--', linewidth=2, color='red', alpha=0.5, label='Target (33%)')
        
        # Zero line
        zero = [0] * (N + 1)
        ax.plot(angles, zero, '-', linewidth=1, color='gray', alpha=0.5)
        
        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([l.replace('_', '\n') for l in labels], size=10)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Radar chart saved to: {save_path}")
        
        plt.show()
    
    def plot_curriculum_progress(
        self,
        stages: List[Dict],
        title: str = "Curriculum Learning Progress",
        save_path: Optional[str] = None
    ):
        """
        Plot curriculum learning stages progress
        
        Args:
            stages: List of stage results
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Combine all stage data
        all_rewards = []
        all_waiting_times = []
        stage_boundaries = [0]
        stage_names = []
        
        for stage in stages:
            rewards = stage.get('rewards', [])
            waiting_times = stage.get('waiting_times', [])
            
            all_rewards.extend(rewards)
            all_waiting_times.extend(waiting_times)
            stage_boundaries.append(len(all_rewards))
            stage_names.append(stage.get('stage_name', 'unknown'))
        
        episodes = range(len(all_rewards))
        
        # Rewards plot
        ax1 = axes[0]
        ax1.plot(episodes, all_rewards, alpha=0.3, color='blue')
        
        # Smoothed
        window = min(50, len(all_rewards) // 20)
        if window > 1:
            smoothed = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(all_rewards)), smoothed, color='blue', linewidth=2)
        
        # Stage boundaries
        colors = plt.cm.Set3(np.linspace(0, 1, len(stages)))
        for i, (start, end) in enumerate(zip(stage_boundaries[:-1], stage_boundaries[1:])):
            ax1.axvspan(start, end, alpha=0.2, color=colors[i], label=stage_names[i])
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Episode Rewards Across Curriculum Stages')
        ax1.legend(loc='upper left')
        
        # Waiting times plot
        ax2 = axes[1]
        ax2.plot(episodes, all_waiting_times, alpha=0.3, color='red')
        
        if window > 1:
            smoothed = np.convolve(all_waiting_times, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(all_waiting_times)), smoothed, color='red', linewidth=2)
        
        for i, (start, end) in enumerate(zip(stage_boundaries[:-1], stage_boundaries[1:])):
            ax2.axvspan(start, end, alpha=0.2, color=colors[i])
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Waiting Time (s)')
        ax2.set_title('Average Waiting Time Across Curriculum Stages')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Curriculum progress saved to: {save_path}")
        
        plt.show()
    
    def plot_junction_heatmap(
        self,
        queue_data: Dict[str, List[float]],
        title: str = "Junction Queue Length Heatmap",
        save_path: Optional[str] = None
    ):
        """
        Plot heatmap of queue lengths across time and lanes
        
        Args:
            queue_data: Dict mapping lane name to queue length over time
            title: Plot title
            save_path: Optional path to save figure
        """
        if not SEABORN_AVAILABLE:
            logger.warning("Seaborn not available for heatmap")
            return
        
        # Convert to 2D array
        lanes = list(queue_data.keys())
        time_steps = len(list(queue_data.values())[0])
        
        data = np.array([queue_data[lane] for lane in lanes])
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        sns.heatmap(
            data,
            ax=ax,
            cmap='YlOrRd',
            yticklabels=lanes,
            xticklabels=False,
            cbar_kws={'label': 'Queue Length'}
        )
        
        # Add time labels
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Lane')
        ax.set_title(title)
        
        # Add x-axis ticks at intervals
        tick_interval = time_steps // 10
        ax.set_xticks(range(0, time_steps, tick_interval))
        ax.set_xticklabels(range(0, time_steps, tick_interval))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Heatmap saved to: {save_path}")
        
        plt.show()
    
    def plot_emergency_response(
        self,
        clearance_times: Dict[str, List[float]],
        title: str = "Emergency Vehicle Response Times",
        save_path: Optional[str] = None
    ):
        """
        Plot emergency response time comparison
        
        Args:
            clearance_times: Dict mapping method to clearance times
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Box plot
        methods = list(clearance_times.keys())
        data = [clearance_times[m] for m in methods]
        colors = [self.colors.get(m.lower().replace('-', '_'), 'gray') for m in methods]
        
        bp = ax1.boxplot(data, patch_artist=True, labels=methods)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Clearance Time (s)')
        ax1.set_title('Distribution of Emergency Clearance Times')
        ax1.tick_params(axis='x', rotation=45)
        
        # Average comparison bar chart
        averages = [np.mean(d) for d in data]
        bars = ax2.bar(methods, averages, color=colors, alpha=0.8, edgecolor='black')
        
        # Add 67% improvement target line
        if 'fixed_time' in methods:
            fixed_idx = methods.index('fixed_time')
            target = averages[fixed_idx] * 0.33  # 67% improvement means 33% of original
            ax2.axhline(y=target, color='red', linestyle='--', linewidth=2, label='67% Improvement Target')
            ax2.legend()
        
        for bar, val in zip(bars, averages):
            height = bar.get_height()
            ax2.annotate(f'{val:.1f}s',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11)
        
        ax2.set_ylabel('Average Clearance Time (s)')
        ax2.set_title('Average Emergency Clearance Time')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Emergency response plot saved to: {save_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations for traffic control results"
    )
    
    parser.add_argument(
        '--results', '-r',
        type=str,
        required=True,
        help='Path to results JSON file or directory'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='visualizations',
        help='Output directory for plots'
    )
    
    parser.add_argument(
        '--plot-type', '-p',
        type=str,
        choices=['all', 'training', 'comparison', 'radar', 'curriculum', 'heatmap', 'emergency'],
        default='all',
        help='Type of plot to generate'
    )
    
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Output format for plots'
    )
    
    args = parser.parse_args()
    
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib is required for visualization. Install with: pip install matplotlib")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results_path = Path(args.results)
    
    if results_path.is_file():
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        logger.error(f"Results file not found: {results_path}")
        sys.exit(1)
    
    # Create visualizer
    viz = TrafficVisualizer()
    
    # Generate plots based on type
    if args.plot_type in ['all', 'training']:
        if 'stages' in results:
            # Curriculum training results
            all_history = {
                'episode_rewards': [],
                'avg_waiting_times': [],
                'epsilons': [],
                'losses': []
            }
            
            for stage in results['stages']:
                all_history['episode_rewards'].extend(stage.get('rewards', []))
                all_history['avg_waiting_times'].extend(stage.get('waiting_times', []))
            
            viz.plot_training_curves(
                all_history,
                title=f"Training Progress - {results.get('agent_type', 'Agent')}",
                save_path=str(output_dir / f"training_curves.{args.format}")
            )
    
    if args.plot_type in ['all', 'curriculum']:
        if 'stages' in results:
            viz.plot_curriculum_progress(
                results['stages'],
                title="Curriculum Learning Progress",
                save_path=str(output_dir / f"curriculum_progress.{args.format}")
            )
    
    if args.plot_type in ['all', 'comparison']:
        if 'baselines' in results and 'agent' in results:
            comparison_data = {
                'RL Agent': results['agent'].get('summary', {}),
                **results.get('baselines', {})
            }
            
            viz.plot_comparison_bars(
                comparison_data,
                title="Performance Comparison",
                save_path=str(output_dir / f"comparison.{args.format}")
            )
    
    if args.plot_type in ['all', 'radar']:
        if 'comparison' in results:
            for baseline_name, improvements in results['comparison'].items():
                viz.plot_improvement_radar(
                    improvements,
                    title=f"Improvement vs {baseline_name}",
                    save_path=str(output_dir / f"radar_{baseline_name}.{args.format}")
                )
    
    logger.info(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
