import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional, Dict
import pandas as pd


class TrainingVisualizer:
    """Visualization utilities for training metrics"""

    def __init__(self, style='seaborn-v0_8-darkgrid'):
        plt.style.use(style)
        sns.set_palette("husl")

    def plot_rewards(self, rewards: List[float],
                     window_size: int = 10,
                     save_path: Optional[str] = None):
        """Plot reward progression with moving average"""
        fig, ax = plt.subplots(figsize=(12, 6))

        steps = range(len(rewards))
        ax.plot(steps, rewards, alpha=0.3, label='Raw Rewards')

        # Moving average
        if len(rewards) >= window_size:
            moving_avg = pd.Series(rewards).rolling(
                window=window_size
            ).mean()
            ax.plot(steps, moving_avg, linewidth=2,
                    label=f'{window_size}-Step Moving Average')

        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Reward')
        ax.set_title('Training Reward Progression')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_multiple_metrics(self, metrics_dict: Dict[str, List[float]],
                              save_path: Optional[str] = None):
        """Plot multiple metrics in subplots"""
        n_metrics = len(metrics_dict)
        fig, axes = plt.subplots(n_metrics, 1,
                                 figsize=(12, 4 * n_metrics))

        if n_metrics == 1:
            axes = [axes]

        for ax, (name, values) in zip(axes, metrics_dict.items()):
            steps = range(len(values))
            ax.plot(steps, values, linewidth=2)
            ax.set_xlabel('Training Step')
            ax.set_ylabel(name.replace('_', ' ').title())
            ax.set_title(f'{name.replace("_", " ").title()} Over Time')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_reward_distribution(self, rewards: List[float],
                                 save_path: Optional[str] = None):
        """Plot reward distribution histogram"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(rewards, bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(x=np.mean(rewards), color='r',
                    linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
        ax1.axvline(x=np.median(rewards), color='g',
                    linestyle='--', label=f'Median: {np.median(rewards):.2f}')
        ax1.set_xlabel('Reward Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Reward Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(rewards, vert=True)
        ax2.set_ylabel('Reward Value')
        ax2.set_title('Reward Box Plot')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_accuracy_breakdown(self, format_acc: List[float],
                                answer_acc: List[float],
                                language_acc: List[float],
                                save_path: Optional[str] = None):
        """Plot stacked accuracy metrics"""
        fig, ax = plt.subplots(figsize=(12, 6))

        steps = range(len(format_acc))

        ax.plot(steps, format_acc, label='Format Accuracy',
                marker='o', markersize=3)
        ax.plot(steps, answer_acc, label='Answer Accuracy',
                marker='s', markersize=3)
        ax.plot(steps, language_acc, label='Language Accuracy',
                marker='^', markersize=3)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Metrics Breakdown')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_training_summary(self, metrics: 'TrainingMetrics',
                              save_path: Optional[str] = None):
        """Create comprehensive training summary visualization"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Reward over time
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(metrics.rewards, alpha=0.5)
        if len(metrics.rewards) >= 10:
            moving_avg = pd.Series(metrics.rewards).rolling(10).mean()
            ax1.plot(moving_avg, linewidth=2)
        ax1.set_title('Reward Progression')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)

        # Loss over time
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(metrics.losses, color='orange')
        ax2.set_title('Training Loss')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)

        # KL divergence
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(metrics.kl_divergences, color='red')
        ax3.set_title('KL Divergence')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('KL')
        ax3.grid(True, alpha=0.3)

        # Completion length
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(metrics.completion_lengths, color='green')
        ax4.set_title('Completion Length')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Tokens')
        ax4.grid(True, alpha=0.3)

        # Reward distribution
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.hist(metrics.rewards, bins=30, alpha=0.7, edgecolor='black')
        ax5.axvline(np.mean(metrics.rewards), color='r',
                    linestyle='--', label='Mean')
        ax5.set_title('Reward Distribution')
        ax5.set_xlabel('Reward')
        ax5.set_ylabel('Count')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_comparison_plot(self, baseline_rewards: List[float],
                               trained_rewards: List[float],
                               save_path: Optional[str] = None):
        """Compare baseline vs trained model performance"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Time series comparison
        steps = range(max(len(baseline_rewards), len(trained_rewards)))
        if baseline_rewards:
            ax1.plot(range(len(baseline_rewards)), baseline_rewards,
                     label='Baseline', alpha=0.7)
        if trained_rewards:
            ax1.plot(range(len(trained_rewards)), trained_rewards,
                     label='Trained', alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Reward')
        ax1.set_title('Performance Comparison Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Distribution comparison
        data_to_plot = []
        labels = []
        if baseline_rewards:
            data_to_plot.append(baseline_rewards)
            labels.append('Baseline')
        if trained_rewards:
            data_to_plot.append(trained_rewards)
            labels.append('Trained')

        ax2.boxplot(data_to_plot, labels=labels)
        ax2.set_ylabel('Reward')
        ax2.set_title('Reward Distribution Comparison')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()