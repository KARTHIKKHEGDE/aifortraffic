"""
Visualization Dashboard for Traffic Signal Control

Real-time and post-hoc visualization of:
- Training progress
- Traffic state
- Agent decisions
- Performance metrics
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.animation import FuncAnimation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


@dataclass
class TrainingMetrics:
    """Container for training metrics over time"""
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    
    # Traffic-specific metrics
    avg_waiting_times: List[float] = field(default_factory=list)
    avg_queue_lengths: List[float] = field(default_factory=list)
    throughputs: List[float] = field(default_factory=list)
    emergency_response_times: List[float] = field(default_factory=list)
    
    # Timestamps
    timestamps: List[float] = field(default_factory=list)
    
    def add_episode(
        self,
        reward: float,
        length: int,
        loss: float = 0,
        lr: float = 0,
        waiting_time: float = 0,
        queue_length: float = 0,
        throughput: float = 0,
        emergency_time: float = 0
    ):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.losses.append(loss)
        self.learning_rates.append(lr)
        self.avg_waiting_times.append(waiting_time)
        self.avg_queue_lengths.append(queue_length)
        self.throughputs.append(throughput)
        self.emergency_response_times.append(emergency_time)
        self.timestamps.append(time.time())
    
    def to_dict(self) -> Dict[str, List]:
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'learning_rates': self.learning_rates,
            'avg_waiting_times': self.avg_waiting_times,
            'avg_queue_lengths': self.avg_queue_lengths,
            'throughputs': self.throughputs,
            'emergency_response_times': self.emergency_response_times,
            'timestamps': self.timestamps
        }
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingMetrics':
        with open(filepath, 'r') as f:
            data = json.load(f)
        metrics = cls()
        for key, value in data.items():
            setattr(metrics, key, value)
        return metrics


class TrainingVisualizer:
    """
    Visualize training progress with multiple plot types
    """
    
    def __init__(
        self,
        metrics: Optional[TrainingMetrics] = None,
        output_dir: str = "visualizations",
        style: str = "seaborn-v0_8-whitegrid"
    ):
        self.metrics = metrics or TrainingMetrics()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_MATPLOTLIB:
            try:
                plt.style.use(style)
            except:
                pass
    
    def plot_training_curves(
        self,
        window: int = 100,
        save: bool = True,
        show: bool = False
    ) -> Optional[plt.Figure]:
        """
        Plot comprehensive training curves
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib required for plotting")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Episode Rewards
        ax = axes[0, 0]
        rewards = self.metrics.episode_rewards
        if rewards:
            ax.plot(rewards, alpha=0.3, color='blue', label='Raw')
            if len(rewards) >= window:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(rewards)), smoothed, 
                       color='blue', linewidth=2, label=f'MA-{window}')
            ax.axhline(y=np.mean(rewards), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(rewards):.1f}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Episode Lengths
        ax = axes[0, 1]
        lengths = self.metrics.episode_lengths
        if lengths:
            ax.plot(lengths, alpha=0.5, color='green')
            if len(lengths) >= window:
                smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(lengths)), smoothed, 
                       color='darkgreen', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Lengths')
        ax.grid(True, alpha=0.3)
        
        # 3. Loss
        ax = axes[0, 2]
        losses = self.metrics.losses
        if losses and any(l > 0 for l in losses):
            ax.plot(losses, alpha=0.5, color='red')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 4. Waiting Times
        ax = axes[1, 0]
        waiting = self.metrics.avg_waiting_times
        if waiting:
            ax.plot(waiting, color='orange', alpha=0.5)
            if len(waiting) >= window:
                smoothed = np.convolve(waiting, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(waiting)), smoothed, 
                       color='darkorange', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Seconds')
        ax.set_title('Average Waiting Time')
        ax.grid(True, alpha=0.3)
        
        # 5. Queue Lengths
        ax = axes[1, 1]
        queues = self.metrics.avg_queue_lengths
        if queues:
            ax.plot(queues, color='purple', alpha=0.5)
            if len(queues) >= window:
                smoothed = np.convolve(queues, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(queues)), smoothed, 
                       color='darkviolet', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Vehicles')
        ax.set_title('Average Queue Length')
        ax.grid(True, alpha=0.3)
        
        # 6. Throughput
        ax = axes[1, 2]
        throughput = self.metrics.throughputs
        if throughput:
            ax.plot(throughput, color='teal', alpha=0.5)
            if len(throughput) >= window:
                smoothed = np.convolve(throughput, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(throughput)), smoothed, 
                       color='darkcyan', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Vehicles/Episode')
        ax.set_title('Throughput')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / "training_curves.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_reward_distribution(self, save: bool = True) -> Optional[plt.Figure]:
        """Plot reward distribution across episodes"""
        if not HAS_MATPLOTLIB:
            return None
        
        rewards = self.metrics.episode_rewards
        if not rewards:
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Histogram
        ax = axes[0]
        ax.hist(rewards, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(rewards), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(rewards):.1f}')
        ax.axvline(np.median(rewards), color='green', linestyle='--', 
                  label=f'Median: {np.median(rewards):.1f}')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution')
        ax.legend()
        
        # Box plot over time
        ax = axes[1]
        n_bins = min(10, len(rewards) // 10)
        if n_bins > 0:
            bin_size = len(rewards) // n_bins
            bins = [rewards[i*bin_size:(i+1)*bin_size] for i in range(n_bins)]
            ax.boxplot(bins, labels=[f'{i+1}' for i in range(n_bins)])
            ax.set_xlabel('Training Phase')
            ax.set_ylabel('Reward')
            ax.set_title('Reward by Training Phase')
        
        # Cumulative reward
        ax = axes[2]
        cumulative = np.cumsum(rewards)
        ax.plot(cumulative, color='blue')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Cumulative Reward')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / "reward_distribution.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.close()
        return fig
    
    def plot_performance_comparison(
        self,
        baseline_metrics: Dict[str, float],
        agent_metrics: Dict[str, float],
        save: bool = True
    ) -> Optional[plt.Figure]:
        """
        Compare agent performance against baselines
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics_names = list(baseline_metrics.keys())
        x = np.arange(len(metrics_names))
        width = 0.35
        
        baseline_values = [baseline_metrics[m] for m in metrics_names]
        agent_values = [agent_metrics.get(m, 0) for m in metrics_names]
        
        bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', 
                      color='gray', alpha=0.7)
        bars2 = ax.bar(x + width/2, agent_values, width, label='RL Agent', 
                      color='blue', alpha=0.7)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        ax.set_title('Performance Comparison: Baseline vs RL Agent')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / "performance_comparison.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.close()
        return fig


class TrafficStateVisualizer:
    """
    Visualize traffic state at intersections
    """
    
    def __init__(self, n_junctions: int = 4, n_approaches: int = 4):
        self.n_junctions = n_junctions
        self.n_approaches = n_approaches
        
        # Junction names (Bangalore)
        self.junction_names = [
            'Silk Board', 'Tin Factory', 'Hebbal', 'Marathahalli'
        ][:n_junctions]
        
        self.approach_names = ['North', 'East', 'South', 'West'][:n_approaches]
    
    def create_junction_heatmap(
        self,
        queue_lengths: np.ndarray,
        title: str = "Queue Lengths by Junction and Approach",
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Create heatmap of queue lengths
        
        Args:
            queue_lengths: Shape (n_junctions, n_approaches)
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(queue_lengths, cmap='YlOrRd', aspect='auto')
        
        # Labels
        ax.set_xticks(np.arange(len(self.approach_names)))
        ax.set_yticks(np.arange(len(self.junction_names)))
        ax.set_xticklabels(self.approach_names)
        ax.set_yticklabels(self.junction_names)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add values
        for i in range(len(self.junction_names)):
            for j in range(len(self.approach_names)):
                text = ax.text(j, i, f'{queue_lengths[i, j]:.0f}',
                              ha="center", va="center", color="black")
        
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label='Queue Length')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return fig
    
    def create_signal_phase_diagram(
        self,
        phases: List[int],
        phase_durations: List[float],
        junction_name: str = "Junction",
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Create signal timing diagram
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 4))
        
        colors = ['green', 'yellow', 'red', 'green']  # Phase colors
        phase_labels = ['N-S Green', 'Transition', 'E-W Green', 'Transition']
        
        current_time = 0
        for i, (phase, duration) in enumerate(zip(phases, phase_durations)):
            color = colors[phase % len(colors)]
            ax.barh(0, duration, left=current_time, height=0.5, 
                   color=color, edgecolor='black', alpha=0.7)
            
            # Add label
            if duration > 5:
                ax.text(current_time + duration/2, 0, f'{duration:.0f}s',
                       ha='center', va='center', fontsize=9)
            
            current_time += duration
        
        ax.set_xlim(0, current_time)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Time (seconds)')
        ax.set_title(f'Signal Timing: {junction_name}')
        ax.set_yticks([])
        
        # Legend
        patches = [mpatches.Patch(color=c, label=l, alpha=0.7) 
                  for c, l in zip(colors[:3], phase_labels[:3])]
        ax.legend(handles=patches, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return fig


class InteractiveDashboard:
    """
    Interactive dashboard using Plotly
    """
    
    def __init__(self, metrics: TrainingMetrics):
        self.metrics = metrics
    
    def create_dashboard(self, save_html: str = "dashboard.html"):
        """Create interactive HTML dashboard"""
        if not HAS_PLOTLY:
            print("plotly required for interactive dashboard")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Episode Rewards', 'Waiting Times',
                'Queue Lengths', 'Throughput',
                'Reward Distribution', 'Learning Progress'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "scatter"}]
            ]
        )
        
        # Episode Rewards
        rewards = self.metrics.episode_rewards
        if rewards:
            fig.add_trace(
                go.Scatter(y=rewards, mode='lines', name='Reward',
                          line=dict(color='blue', width=1), opacity=0.5),
                row=1, col=1
            )
            # Moving average
            window = min(100, len(rewards) // 5)
            if window > 0:
                ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
                fig.add_trace(
                    go.Scatter(y=ma, mode='lines', name=f'MA-{window}',
                              line=dict(color='red', width=2)),
                    row=1, col=1
                )
        
        # Waiting Times
        waiting = self.metrics.avg_waiting_times
        if waiting:
            fig.add_trace(
                go.Scatter(y=waiting, mode='lines', name='Waiting Time',
                          line=dict(color='orange')),
                row=1, col=2
            )
        
        # Queue Lengths
        queues = self.metrics.avg_queue_lengths
        if queues:
            fig.add_trace(
                go.Scatter(y=queues, mode='lines', name='Queue Length',
                          line=dict(color='purple')),
                row=2, col=1
            )
        
        # Throughput
        throughput = self.metrics.throughputs
        if throughput:
            fig.add_trace(
                go.Scatter(y=throughput, mode='lines', name='Throughput',
                          line=dict(color='green')),
                row=2, col=2
            )
        
        # Reward Distribution
        if rewards:
            fig.add_trace(
                go.Histogram(x=rewards, nbinsx=50, name='Reward Dist',
                            marker_color='blue', opacity=0.7),
                row=3, col=1
            )
        
        # Learning Progress (loss)
        losses = self.metrics.losses
        if losses and any(l > 0 for l in losses):
            fig.add_trace(
                go.Scatter(y=losses, mode='lines', name='Loss',
                          line=dict(color='red')),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=900,
            title_text="Traffic Signal Control Training Dashboard",
            showlegend=True
        )
        
        # Save
        fig.write_html(save_html)
        print(f"Interactive dashboard saved: {save_html}")
        
        return fig
    
    def create_real_time_view(
        self,
        queue_data: List[List[float]],
        junction_names: List[str]
    ):
        """Create real-time traffic state visualization"""
        if not HAS_PLOTLY:
            return
        
        fig = go.Figure()
        
        for i, (name, queues) in enumerate(zip(junction_names, queue_data)):
            fig.add_trace(go.Bar(
                name=name,
                x=['North', 'East', 'South', 'West'],
                y=queues,
                text=[f'{q:.0f}' for q in queues],
                textposition='auto',
            ))
        
        fig.update_layout(
            barmode='group',
            title='Real-Time Queue Lengths by Junction',
            xaxis_title='Approach',
            yaxis_title='Queue Length (vehicles)',
            legend_title='Junction'
        )
        
        return fig


# Convenience functions
def plot_training_results(
    metrics_file: str,
    output_dir: str = "visualizations"
):
    """Load and plot training results from file"""
    metrics = TrainingMetrics.load(metrics_file)
    visualizer = TrainingVisualizer(metrics, output_dir)
    
    visualizer.plot_training_curves(save=True)
    visualizer.plot_reward_distribution(save=True)
    
    print(f"Visualizations saved to {output_dir}")


def create_comparison_report(
    results: Dict[str, TrainingMetrics],
    output_dir: str = "visualizations"
) -> Optional[plt.Figure]:
    """Create comparison report for multiple training runs"""
    if not HAS_MATPLOTLIB:
        return None
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for (name, metrics), color in zip(results.items(), colors):
        rewards = metrics.episode_rewards
        if not rewards:
            continue
        
        # Smoothed rewards
        window = min(50, len(rewards) // 5)
        if window > 0:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(smoothed, label=name, color=color, linewidth=2)
        
        # Final performance
        final_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
        axes[0, 1].boxplot([final_rewards], positions=[list(results.keys()).index(name)],
                          widths=0.6)
        
        # Waiting times
        waiting = metrics.avg_waiting_times
        if waiting:
            axes[1, 0].plot(waiting, label=name, color=color, alpha=0.7)
        
        # Throughput
        throughput = metrics.throughputs
        if throughput:
            axes[1, 1].plot(throughput, label=name, color=color, alpha=0.7)
    
    axes[0, 0].set_title('Learning Curves')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Final Performance Distribution')
    axes[0, 1].set_xticklabels(list(results.keys()), rotation=45)
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Average Waiting Time')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Seconds')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Throughput')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Vehicles')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "comparison_report.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig


if __name__ == '__main__':
    # Demo with synthetic data
    metrics = TrainingMetrics()
    
    for i in range(500):
        reward = -100 + i * 0.5 + np.random.randn() * 20
        metrics.add_episode(
            reward=reward,
            length=100 + np.random.randint(-20, 20),
            loss=max(0.1, 1.0 - i * 0.001 + np.random.randn() * 0.1),
            waiting_time=max(10, 60 - i * 0.08 + np.random.randn() * 5),
            queue_length=max(2, 15 - i * 0.02 + np.random.randn() * 2),
            throughput=min(200, 50 + i * 0.2 + np.random.randn() * 10)
        )
    
    visualizer = TrainingVisualizer(metrics)
    visualizer.plot_training_curves(save=True, show=False)
    visualizer.plot_reward_distribution(save=True)
    
    print("Demo visualizations created!")
