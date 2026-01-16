"""
Statistical Testing Framework for RL Agent Evaluation

Provides rigorous statistical analysis for:
- Comparing agent performance
- Confidence intervals
- Hypothesis testing
- Effect size calculation
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np

try:
    from scipy import stats
    from scipy.stats import (
        ttest_ind, ttest_rel, mannwhitneyu, wilcoxon,
        shapiro, levene, bootstrap
    )
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class StatisticalResult:
    """Container for statistical test results"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    effect_interpretation: Optional[str] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_sizes: Optional[Tuple[int, int]] = None
    means: Optional[Tuple[float, float]] = None
    stds: Optional[Tuple[float, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'significant': self.significant,
            'effect_size': self.effect_size,
            'effect_interpretation': self.effect_interpretation,
            'confidence_interval': self.confidence_interval,
            'sample_sizes': self.sample_sizes,
            'means': self.means,
            'stds': self.stds
        }
    
    def __str__(self) -> str:
        sig_str = "✓ Significant" if self.significant else "✗ Not significant"
        lines = [
            f"\n{'='*60}",
            f"Statistical Test: {self.test_name}",
            f"{'='*60}",
            f"Test statistic: {self.statistic:.4f}",
            f"P-value: {self.p_value:.6f}",
            f"Result: {sig_str} (α=0.05)",
        ]
        
        if self.means:
            lines.append(f"Means: {self.means[0]:.2f} vs {self.means[1]:.2f}")
        
        if self.stds:
            lines.append(f"Std devs: {self.stds[0]:.2f} vs {self.stds[1]:.2f}")
        
        if self.effect_size is not None:
            lines.append(f"Effect size (Cohen's d): {self.effect_size:.4f} ({self.effect_interpretation})")
        
        if self.confidence_interval:
            lines.append(f"95% CI for difference: [{self.confidence_interval[0]:.2f}, {self.confidence_interval[1]:.2f}]")
        
        lines.append('='*60)
        return '\n'.join(lines)


class StatisticalTester:
    """
    Statistical testing framework for RL experiments
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize tester
        
        Args:
            alpha: Significance level (default 0.05)
        """
        self.alpha = alpha
        
        if not HAS_SCIPY:
            print("Warning: scipy not available. Some tests will be limited.")
    
    def compute_effect_size(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        method: str = 'cohens_d'
    ) -> Tuple[float, str]:
        """
        Compute effect size between two samples
        
        Args:
            data1: First sample
            data2: Second sample
            method: 'cohens_d' or 'hedges_g'
            
        Returns:
            Tuple of (effect_size, interpretation)
        """
        n1, n2 = len(data1), len(data2)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        
        if pooled_std == 0:
            return 0.0, 'undefined'
        
        if method == 'cohens_d':
            d = (mean1 - mean2) / pooled_std
        elif method == 'hedges_g':
            # Hedges' g with correction factor
            d = (mean1 - mean2) / pooled_std
            correction = 1 - (3 / (4*(n1+n2) - 9))
            d *= correction
        else:
            d = (mean1 - mean2) / pooled_std
        
        # Interpretation
        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = 'negligible'
        elif abs_d < 0.5:
            interpretation = 'small'
        elif abs_d < 0.8:
            interpretation = 'medium'
        else:
            interpretation = 'large'
        
        return d, interpretation
    
    def compute_confidence_interval(
        self,
        data: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for mean
        
        Args:
            data: Sample data
            confidence: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n = len(data)
        mean = np.mean(data)
        se = np.std(data, ddof=1) / np.sqrt(n)
        
        if HAS_SCIPY:
            t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        else:
            # Approximate for large samples
            t_value = 1.96 if confidence == 0.95 else 2.576
        
        margin = t_value * se
        return (mean - margin, mean + margin)
    
    def compute_difference_ci(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for difference between means
        """
        n1, n2 = len(data1), len(data2)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        
        # Standard error of difference
        se_diff = np.sqrt(var1/n1 + var2/n2)
        
        # Welch-Satterthwaite degrees of freedom
        df = ((var1/n1 + var2/n2)**2 /
              ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)))
        
        if HAS_SCIPY:
            t_value = stats.t.ppf((1 + confidence) / 2, df)
        else:
            t_value = 1.96
        
        diff = mean1 - mean2
        margin = t_value * se_diff
        
        return (diff - margin, diff + margin)
    
    def test_normality(self, data: np.ndarray) -> Tuple[bool, float]:
        """
        Test if data is normally distributed using Shapiro-Wilk test
        
        Returns:
            Tuple of (is_normal, p_value)
        """
        if not HAS_SCIPY:
            return True, 1.0  # Assume normal if scipy not available
        
        # Shapiro-Wilk test (works best for n < 5000)
        if len(data) > 5000:
            data = np.random.choice(data, 5000, replace=False)
        
        stat, p_value = shapiro(data)
        is_normal = p_value > self.alpha
        
        return is_normal, p_value
    
    def test_equal_variance(
        self,
        data1: np.ndarray,
        data2: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Test if two samples have equal variance using Levene's test
        
        Returns:
            Tuple of (equal_variance, p_value)
        """
        if not HAS_SCIPY:
            return True, 1.0
        
        stat, p_value = levene(data1, data2)
        equal_var = p_value > self.alpha
        
        return equal_var, p_value
    
    def independent_samples_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        names: Tuple[str, str] = ('Group1', 'Group2')
    ) -> StatisticalResult:
        """
        Compare two independent samples
        
        Automatically chooses appropriate test based on assumptions:
        - Normal + equal variance: Independent t-test
        - Normal + unequal variance: Welch's t-test
        - Non-normal: Mann-Whitney U test
        
        Args:
            data1: First sample
            data2: Second sample
            names: Names for the two groups
            
        Returns:
            StatisticalResult with test details
        """
        data1 = np.array(data1)
        data2 = np.array(data2)
        
        # Check assumptions
        normal1, _ = self.test_normality(data1)
        normal2, _ = self.test_normality(data2)
        is_normal = normal1 and normal2
        
        equal_var, _ = self.test_equal_variance(data1, data2)
        
        if HAS_SCIPY:
            if is_normal:
                # Use t-test
                stat, p_value = ttest_ind(data1, data2, equal_var=equal_var)
                test_name = "Welch's t-test" if not equal_var else "Independent t-test"
            else:
                # Use Mann-Whitney U
                stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                test_name = "Mann-Whitney U test"
        else:
            # Simple t-test approximation
            n1, n2 = len(data1), len(data2)
            mean1, mean2 = np.mean(data1), np.mean(data2)
            var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
            
            se = np.sqrt(var1/n1 + var2/n2)
            stat = (mean1 - mean2) / se if se > 0 else 0
            p_value = 0.05  # Placeholder
            test_name = "Approximate t-test"
        
        # Effect size
        effect_size, effect_interp = self.compute_effect_size(data1, data2)
        
        # Confidence interval for difference
        ci = self.compute_difference_ci(data1, data2)
        
        return StatisticalResult(
            test_name=f"{test_name}: {names[0]} vs {names[1]}",
            statistic=float(stat),
            p_value=float(p_value),
            significant=p_value < self.alpha,
            effect_size=effect_size,
            effect_interpretation=effect_interp,
            confidence_interval=ci,
            sample_sizes=(len(data1), len(data2)),
            means=(float(np.mean(data1)), float(np.mean(data2))),
            stds=(float(np.std(data1)), float(np.std(data2)))
        )
    
    def paired_samples_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        names: Tuple[str, str] = ('Before', 'After')
    ) -> StatisticalResult:
        """
        Compare two paired/matched samples
        
        Uses paired t-test or Wilcoxon signed-rank test.
        
        Args:
            data1: First sample
            data2: Second sample (paired with data1)
            names: Names for the conditions
            
        Returns:
            StatisticalResult
        """
        data1 = np.array(data1)
        data2 = np.array(data2)
        
        if len(data1) != len(data2):
            raise ValueError("Paired samples must have equal length")
        
        # Differences
        differences = data1 - data2
        
        # Check normality of differences
        is_normal, _ = self.test_normality(differences)
        
        if HAS_SCIPY:
            if is_normal:
                stat, p_value = ttest_rel(data1, data2)
                test_name = "Paired t-test"
            else:
                stat, p_value = wilcoxon(differences)
                test_name = "Wilcoxon signed-rank test"
        else:
            mean_diff = np.mean(differences)
            se_diff = np.std(differences, ddof=1) / np.sqrt(len(differences))
            stat = mean_diff / se_diff if se_diff > 0 else 0
            p_value = 0.05
            test_name = "Approximate paired t-test"
        
        # Effect size (using differences)
        effect_size = np.mean(differences) / np.std(differences, ddof=1)
        
        abs_d = abs(effect_size)
        if abs_d < 0.2:
            effect_interp = 'negligible'
        elif abs_d < 0.5:
            effect_interp = 'small'
        elif abs_d < 0.8:
            effect_interp = 'medium'
        else:
            effect_interp = 'large'
        
        # CI for mean difference
        ci = self.compute_confidence_interval(differences)
        
        return StatisticalResult(
            test_name=f"{test_name}: {names[0]} vs {names[1]}",
            statistic=float(stat),
            p_value=float(p_value),
            significant=p_value < self.alpha,
            effect_size=effect_size,
            effect_interpretation=effect_interp,
            confidence_interval=ci,
            sample_sizes=(len(data1), len(data2)),
            means=(float(np.mean(data1)), float(np.mean(data2))),
            stds=(float(np.std(data1)), float(np.std(data2)))
        )
    
    def multiple_comparison(
        self,
        groups: Dict[str, np.ndarray],
        correction: str = 'bonferroni'
    ) -> Dict[str, StatisticalResult]:
        """
        Compare multiple groups with correction for multiple comparisons
        
        Args:
            groups: Dict mapping group_name -> data
            correction: 'bonferroni' or 'holm' correction method
            
        Returns:
            Dict mapping comparison_name -> StatisticalResult
        """
        group_names = list(groups.keys())
        n_comparisons = len(group_names) * (len(group_names) - 1) // 2
        
        results = {}
        
        for i, name1 in enumerate(group_names):
            for j, name2 in enumerate(group_names[i+1:], i+1):
                comparison = f"{name1}_vs_{name2}"
                
                result = self.independent_samples_test(
                    groups[name1],
                    groups[name2],
                    names=(name1, name2)
                )
                
                # Apply correction
                if correction == 'bonferroni':
                    adjusted_alpha = self.alpha / n_comparisons
                    result.significant = result.p_value < adjusted_alpha
                
                results[comparison] = result
        
        return results
    
    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic: callable = np.mean,
        n_bootstrap: int = 10000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval
        
        Args:
            data: Sample data
            statistic: Function to compute (default: mean)
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        data = np.array(data)
        n = len(data)
        
        # Bootstrap resampling
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            resample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic(resample))
        
        # Percentile method
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return (lower, upper)
    
    def power_analysis(
        self,
        effect_size: float,
        sample_size: int,
        alpha: float = 0.05
    ) -> float:
        """
        Compute statistical power for given effect size and sample size
        
        Args:
            effect_size: Expected Cohen's d
            sample_size: Sample size per group
            alpha: Significance level
            
        Returns:
            Statistical power (0-1)
        """
        if not HAS_SCIPY:
            return 0.8  # Placeholder
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(sample_size / 2)
        
        # Critical value
        df = 2 * sample_size - 2
        t_crit = stats.t.ppf(1 - alpha/2, df)
        
        # Power
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
        
        return power
    
    def required_sample_size(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05
    ) -> int:
        """
        Compute required sample size for desired power
        
        Args:
            effect_size: Expected Cohen's d
            power: Desired power
            alpha: Significance level
            
        Returns:
            Required sample size per group
        """
        if not HAS_SCIPY:
            return 30  # Rule of thumb
        
        # Binary search for sample size
        for n in range(5, 10000):
            computed_power = self.power_analysis(effect_size, n, alpha)
            if computed_power >= power:
                return n
        
        return 10000


class ExperimentAnalyzer:
    """
    High-level experiment analysis for RL training results
    """
    
    def __init__(
        self,
        results_dir: str = "results",
        output_dir: str = "analysis"
    ):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tester = StatisticalTester()
    
    def load_results(self, pattern: str = "*.json") -> Dict[str, Dict]:
        """Load all result files matching pattern"""
        results = {}
        
        for filepath in self.results_dir.glob(pattern):
            with open(filepath, 'r') as f:
                results[filepath.stem] = json.load(f)
        
        return results
    
    def compare_algorithms(
        self,
        algorithm_results: Dict[str, List[float]],
        metric_name: str = "reward"
    ) -> Dict[str, StatisticalResult]:
        """
        Compare multiple algorithms statistically
        
        Args:
            algorithm_results: Dict mapping algorithm_name -> list of metric values
            metric_name: Name of metric being compared
            
        Returns:
            Dict of pairwise comparison results
        """
        print(f"\n{'='*60}")
        print(f"Statistical Comparison: {metric_name}")
        print(f"{'='*60}\n")
        
        # Summary statistics
        print("Summary Statistics:")
        print("-" * 40)
        for name, values in algorithm_results.items():
            values = np.array(values)
            print(f"{name}:")
            print(f"  n = {len(values)}")
            print(f"  mean = {np.mean(values):.2f}")
            print(f"  std = {np.std(values):.2f}")
            print(f"  95% CI = {self.tester.compute_confidence_interval(values)}")
        
        print()
        
        # Pairwise comparisons
        results = self.tester.multiple_comparison(
            {name: np.array(values) for name, values in algorithm_results.items()},
            correction='bonferroni'
        )
        
        for comparison, result in results.items():
            print(result)
        
        return results
    
    def compare_baseline(
        self,
        agent_results: List[float],
        baseline_results: List[float],
        agent_name: str = "RL Agent",
        baseline_name: str = "Fixed-Time"
    ) -> StatisticalResult:
        """
        Compare agent performance to baseline
        """
        result = self.tester.independent_samples_test(
            np.array(agent_results),
            np.array(baseline_results),
            names=(agent_name, baseline_name)
        )
        
        print(result)
        
        # Improvement percentage
        improvement = (np.mean(agent_results) - np.mean(baseline_results)) / abs(np.mean(baseline_results)) * 100
        print(f"\nImprovement over baseline: {improvement:+.1f}%")
        
        return result
    
    def generate_report(
        self,
        algorithm_results: Dict[str, List[float]],
        output_file: str = "statistical_report.txt"
    ):
        """Generate comprehensive statistical report"""
        filepath = self.output_dir / output_file
        
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STATISTICAL ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary
            f.write("1. SUMMARY STATISTICS\n")
            f.write("-" * 40 + "\n\n")
            
            for name, values in algorithm_results.items():
                values = np.array(values)
                ci = self.tester.compute_confidence_interval(values)
                
                f.write(f"{name}:\n")
                f.write(f"  Sample size: {len(values)}\n")
                f.write(f"  Mean: {np.mean(values):.4f}\n")
                f.write(f"  Std Dev: {np.std(values):.4f}\n")
                f.write(f"  Median: {np.median(values):.4f}\n")
                f.write(f"  Min: {np.min(values):.4f}\n")
                f.write(f"  Max: {np.max(values):.4f}\n")
                f.write(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]\n\n")
            
            # Normality tests
            f.write("\n2. NORMALITY TESTS\n")
            f.write("-" * 40 + "\n\n")
            
            for name, values in algorithm_results.items():
                is_normal, p_value = self.tester.test_normality(np.array(values))
                f.write(f"{name}: {'Normal' if is_normal else 'Non-normal'} (p={p_value:.4f})\n")
            
            # Pairwise comparisons
            f.write("\n3. PAIRWISE COMPARISONS\n")
            f.write("-" * 40 + "\n\n")
            
            results = self.tester.multiple_comparison(
                {name: np.array(values) for name, values in algorithm_results.items()}
            )
            
            for comparison, result in results.items():
                f.write(f"\n{comparison}:\n")
                f.write(f"  Test: {result.test_name}\n")
                f.write(f"  Statistic: {result.statistic:.4f}\n")
                f.write(f"  P-value: {result.p_value:.6f}\n")
                f.write(f"  Significant: {result.significant}\n")
                f.write(f"  Effect size: {result.effect_size:.4f} ({result.effect_interpretation})\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"Report saved: {filepath}")


# Convenience functions
def compare_agents(
    results: Dict[str, List[float]],
    alpha: float = 0.05
) -> Dict[str, StatisticalResult]:
    """Quick comparison of multiple agents"""
    tester = StatisticalTester(alpha=alpha)
    return tester.multiple_comparison(
        {name: np.array(values) for name, values in results.items()}
    )


def is_significantly_better(
    agent_results: List[float],
    baseline_results: List[float],
    alpha: float = 0.05
) -> bool:
    """Check if agent is significantly better than baseline"""
    tester = StatisticalTester(alpha=alpha)
    result = tester.independent_samples_test(
        np.array(agent_results),
        np.array(baseline_results)
    )
    
    # Check if significant AND agent is better
    agent_better = np.mean(agent_results) > np.mean(baseline_results)
    return result.significant and agent_better


if __name__ == '__main__':
    # Demo
    np.random.seed(42)
    
    # Simulated results
    fixed_time = np.random.normal(-50, 15, 100)
    actuated = np.random.normal(-35, 12, 100)
    dqn_agent = np.random.normal(-20, 10, 100)
    ppo_agent = np.random.normal(-15, 8, 100)
    
    analyzer = ExperimentAnalyzer()
    
    results = {
        'Fixed-Time': fixed_time.tolist(),
        'Actuated': actuated.tolist(),
        'DQN': dqn_agent.tolist(),
        'PPO': ppo_agent.tolist()
    }
    
    # Compare all
    comparisons = analyzer.compare_algorithms(results, metric_name="Episode Reward")
    
    # Generate report
    analyzer.generate_report(results)
