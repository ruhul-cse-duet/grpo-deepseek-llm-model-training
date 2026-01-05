import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
import json


class TrainingMetrics:
    """Track and calculate training metrics"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.rewards = []
        self.losses = []
        self.kl_divergences = []
        self.completion_lengths = []
        self.format_accuracy = []
        self.answer_accuracy = []
        self.language_accuracy = []
        self.step_times = []

    def update(self, step_metrics: Dict):
        """Update metrics with new step data"""
        if 'reward' in step_metrics:
            self.rewards.append(step_metrics['reward'])
        if 'loss' in step_metrics:
            self.losses.append(step_metrics['loss'])
        if 'kl' in step_metrics:
            self.kl_divergences.append(step_metrics['kl'])
        if 'completion_length' in step_metrics:
            self.completion_lengths.append(step_metrics['completion_length'])
        if 'format_accuracy' in step_metrics:
            self.format_accuracy.append(step_metrics['format_accuracy'])
        if 'answer_accuracy' in step_metrics:
            self.answer_accuracy.append(step_metrics['answer_accuracy'])
        if 'language_accuracy' in step_metrics:
            self.language_accuracy.append(step_metrics['language_accuracy'])
        if 'step_time' in step_metrics:
            self.step_times.append(step_metrics['step_time'])

    def get_summary(self, last_n_steps: Optional[int] = None) -> Dict:
        """Get summary statistics"""

        def safe_mean(arr):
            return np.mean(arr[-last_n_steps:] if last_n_steps else arr) if arr else 0

        def safe_std(arr):
            return np.std(arr[-last_n_steps:] if last_n_steps else arr) if arr else 0

        return {
            'avg_reward': safe_mean(self.rewards),
            'std_reward': safe_std(self.rewards),
            'avg_loss': safe_mean(self.losses),
            'avg_kl': safe_mean(self.kl_divergences),
            'avg_completion_length': safe_mean(self.completion_lengths),
            'avg_format_accuracy': safe_mean(self.format_accuracy),
            'avg_answer_accuracy': safe_mean(self.answer_accuracy),
            'avg_language_accuracy': safe_mean(self.language_accuracy),
            'avg_step_time': safe_mean(self.step_times),
            'total_steps': len(self.rewards)
        }

    def get_reward_breakdown(self) -> Dict:
        """Analyze reward distribution"""
        if not self.rewards:
            return {}

        rewards_array = np.array(self.rewards)
        return {
            'min': float(np.min(rewards_array)),
            'max': float(np.max(rewards_array)),
            'median': float(np.median(rewards_array)),
            'p25': float(np.percentile(rewards_array, 25)),
            'p75': float(np.percentile(rewards_array, 75)),
            'positive_ratio': float(np.mean(rewards_array > 0)),
            'negative_ratio': float(np.mean(rewards_array < 0)),
            'zero_ratio': float(np.mean(rewards_array == 0))
        }

    def save_to_file(self, filepath: str):
        """Save metrics to JSON file"""
        data = {
            'rewards': self.rewards,
            'losses': self.losses,
            'kl_divergences': self.kl_divergences,
            'completion_lengths': self.completion_lengths,
            'format_accuracy': self.format_accuracy,
            'answer_accuracy': self.answer_accuracy,
            'language_accuracy': self.language_accuracy,
            'step_times': self.step_times,
            'summary': self.get_summary(),
            'reward_breakdown': self.get_reward_breakdown()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str):
        """Load metrics from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        metrics = cls()
        metrics.rewards = data.get('rewards', [])
        metrics.losses = data.get('losses', [])
        metrics.kl_divergences = data.get('kl_divergences', [])
        metrics.completion_lengths = data.get('completion_lengths', [])
        metrics.format_accuracy = data.get('format_accuracy', [])
        metrics.answer_accuracy = data.get('answer_accuracy', [])
        metrics.language_accuracy = data.get('language_accuracy', [])
        metrics.step_times = data.get('step_times', [])

        return metrics


class EvaluationMetrics:
    """Evaluation metrics for model performance"""

    @staticmethod
    def calculate_accuracy(predictions: List[str],
                           ground_truths: List[str]) -> float:
        """Calculate exact match accuracy"""
        if not predictions or not ground_truths:
            return 0.0

        correct = sum(
            pred.strip() == truth.strip()
            for pred, truth in zip(predictions, ground_truths)
        )
        return correct / len(predictions)

    @staticmethod
    def calculate_format_compliance(texts: List[str],
                                    reasoning_start: str = "<think>",
                                    reasoning_end: str = "</think>") -> float:
        """Calculate percentage of properly formatted responses"""
        if not texts:
            return 0.0

        compliant = sum(
            reasoning_start in text and reasoning_end in text
            for text in texts
        )
        return compliant / len(texts)

    @staticmethod
    def calculate_numerical_accuracy(predictions: List[float],
                                     ground_truths: List[float],
                                     tolerance: float = 0.01) -> float:
        """Calculate accuracy with numerical tolerance"""
        if not predictions or not ground_truths:
            return 0.0

        correct = sum(
            abs(pred - truth) <= tolerance
            for pred, truth in zip(predictions, ground_truths)
        )
        return correct / len(predictions)

    @staticmethod
    def calculate_language_consistency(texts: List[str],
                                       target_language: str = "id") -> float:
        """Calculate percentage in target language"""
        import langid

        if not texts:
            return 0.0

        correct_lang = sum(
            langid.classify(text)[0] == target_language
            for text in texts
        )
        return correct_lang / len(texts)

    @staticmethod
    def calculate_average_reasoning_length(texts: List[str],
                                           reasoning_start: str = "<think>",
                                           reasoning_end: str = "</think>") -> float:
        """Calculate average reasoning section length"""
        pattern = rf"{re.escape(reasoning_start)}(.*?){re.escape(reasoning_end)}"

        lengths = []
        for text in texts:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                reasoning = match.group(1).strip()
                lengths.append(len(reasoning.split()))

        return np.mean(lengths) if lengths else 0.0