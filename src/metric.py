import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score, roc_curve
import json
from pathlib import Path


class ReliableMetrics:
    @staticmethod
    def calculate(small_scores: np.ndarray, large_scores: np.ndarray,
                 router_scores: np.ndarray, dataset_type: str = 'general') -> Dict:
        min_len = min(len(small_scores), len(large_scores), len(router_scores))
        small_scores = small_scores[:min_len]
        large_scores = large_scores[:min_len]
        router_scores = router_scores[:min_len]
        if dataset_type == 'general':
            # general类型：对于评分制数据（alpaca, magpie），比较小模型和大模型分数
            labels = (small_scores >= large_scores).astype(int)
        else:
            # 其他类型：直接使用small_scores作为labels
            labels = small_scores
       
        if len(np.unique(labels)) > 1:
            auroc = roc_auc_score(labels, router_scores)
            fpr, tpr, thresholds = roc_curve(labels, router_scores)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            best_threshold = thresholds[best_idx]
            predictions = (router_scores >= best_threshold).astype(int)
            accuracy = np.mean(predictions == labels)
        else:
            auroc = 0.5
            best_threshold = 0.5
            accuracy = 0.5

        return {
            'auroc': auroc,
            'best_threshold': best_threshold,
            'accuracy': accuracy,
            'num_samples': len(labels),
            'positive_samples': np.sum(labels)
        }


class AdaptiveMetrics:
    @staticmethod
    def calculate(small_scores: np.ndarray, large_scores: np.ndarray,
                 router_scores: np.ndarray,
                #  target_accuracy_band: Tuple[float, float] = (0.7, 0.9),
                 recovery_rate_band: Tuple[float,float] =(0.7,0.9),
                 lpm_call_rate_band: Tuple[float, float] = (0.0, 0.1)) -> Dict:
        min_len = min(len(small_scores), len(large_scores), len(router_scores))
        small_scores = small_scores[:min_len]
        large_scores = large_scores[:min_len]
        router_scores = router_scores[:min_len] 
        small_mean = small_scores.mean()
        large_mean = large_scores.mean()
        
           
        # 分数越低越优先触发大模型调用，因此直接按升序累积
        sorted_indices = np.argsort(router_scores)
        sorted_small = small_scores[sorted_indices]
        sorted_large = large_scores[sorted_indices]
        n_samples = len(router_scores)
        prefix_large = np.cumsum(sorted_large)
        prefix_small = np.cumsum(sorted_small)
        total_small = prefix_small[-1] if n_samples > 0 else 0.0

        call_rates = np.linspace(0, 1, 101)
        accuracies = []
        for rate in call_rates:
            n_large = int(rate * n_samples)

            if n_large == 0:
                mean_acc = total_small / n_samples if n_samples else 0.0
            elif n_large >= n_samples:
                mean_acc = np.mean(sorted_large) if n_samples else 0.0
            else:
                large_sum = prefix_large[n_large - 1]
                small_sum = total_small - prefix_small[n_large - 1]
                mean_acc = (large_sum + small_sum) / n_samples

            accuracies.append(mean_acc)
        accuracies = np.array(accuracies)
        

        # Calculate LPM within指定 call-rate 区间
        lpm_low =  lpm_call_rate_band[0]
        lpm_high = lpm_call_rate_band[1]
        start_idx = np.searchsorted(call_rates, lpm_low, side='left')
        end_idx = np.searchsorted(call_rates, lpm_high, side='right')

        LPM = np.mean(accuracies[start_idx:end_idx])
        lpm_threshold = accuracies[min(end_idx - 1, len(accuracies) - 1)]

        # Calculate HPM
        low_acc_band = small_mean+(large_mean-small_mean)*recovery_rate_band[0]
        high_acc_band = small_mean + (large_mean-small_mean)*recovery_rate_band[1]
        target_mask = (accuracies >= low_acc_band) & (accuracies <= high_acc_band)
        
        if np.any(target_mask):
            target_call_rates = call_rates[target_mask]
            HPM = 1 - np.mean(target_call_rates)
        else:
            HPM = 0.0
        # Calculate MPM
        mpm_mask = (accuracies > lpm_threshold) & (accuracies < low_acc_band)

        if np.any(mpm_mask):
            mpm_acc = accuracies[mpm_mask]
            MPM = np.mean(mpm_acc)
        else:
            MPM = 0.0
      

        return {
            'call_rates': call_rates,
            'accuracies': accuracies,
            'small_mean':small_mean,
            'large_mean':large_mean,
            'LPM': LPM,
            'HPM': HPM,
            'MPM': MPM,
            'recovery_rate_band': recovery_rate_band,
            'lpm_call_rate_band': (lpm_low, lpm_high),
            'max_accuracy': np.max(accuracies),
            'min_accuracy': np.min(accuracies)
        }


class MetricEvaluator:
    def __init__(self):
        self.reliable_metrics = ReliableMetrics()
        self.adaptive_metrics = AdaptiveMetrics()

    def evaluate_from_results(self, small_results: List[Dict], large_results: List[Dict],
                            router_scores: np.ndarray, dataset_type: str = 'general', **kwargs) -> Dict:
        small_scores = np.array([r['score'] for r in small_results])
        large_scores = np.array([r['score'] for r in large_results])

        reliable = self.reliable_metrics.calculate(small_scores, large_scores, router_scores, dataset_type)
        adaptive = self.adaptive_metrics.calculate(small_scores, large_scores, router_scores, **kwargs)
        print(reliable)
        return {
            'reliable_metrics': reliable,
            'adaptive_metrics': adaptive,
            'dataset_info': {
                'num_samples': len(small_results),
                'small_accuracy': np.mean(small_scores),
                'large_accuracy': np.mean(large_scores)
            }
        }

    def evaluate_from_scores(self, small_scores: np.ndarray, large_scores: np.ndarray,
                           router_scores: np.ndarray, dataset_type: str = 'general', **kwargs) -> Dict:
        reliable = self.reliable_metrics.calculate(small_scores, large_scores, router_scores, dataset_type)
        adaptive = self.adaptive_metrics.calculate(small_scores, large_scores, router_scores, **kwargs)

        return {
            'reliable_metrics': reliable,
            'adaptive_metrics': adaptive,
            'dataset_info': {
                'num_samples': len(small_scores),
                'small_accuracy': np.mean(small_scores),
                'large_accuracy': np.mean(large_scores)
            }
        }

    def plot_adaptive_curve(self, adaptive_metrics: Dict, save_path: Optional[str] = None):
        plt.figure(figsize=(6, 5))

        call_rates = adaptive_metrics['call_rates']
        accuracies = adaptive_metrics['accuracies']
        large_mean = adaptive_metrics['large_mean']
        small_mean = adaptive_metrics['small_mean']

        # Calculate random baseline
        random_accuracies = small_mean + (large_mean - small_mean) * call_rates

        plt.plot(call_rates * 100, accuracies * 100, 'b-', linewidth=2, label='Accuracy Curve')
        plt.plot(call_rates * 100, random_accuracies * 100, 'g--', linewidth=2, label='Random Selection')

        max_acc_idx = np.argmax(accuracies)
        plt.plot(call_rates[max_acc_idx] * 100, accuracies[max_acc_idx] * 100,
                'ro', markersize=10, label=f'Max Accuracy ({accuracies[max_acc_idx]*100:.1f}%)')

        plt.xlabel('Call Rate (%)', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        plt.title('Accuracy vs Call Rate', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, frameon=True)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 100)
        plt.ylim(min(accuracies) * 100 - 2, max(accuracies) * 100 + 2)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def save_results(self, results: Dict, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)


class BatchMetricEvaluator:
    def __init__(self, output_dir: str = "metric_results"):
        self.evaluator = MetricEvaluator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def evaluate_multiple_datasets(self, dataset_results: Dict[str, Dict],
                                  router_scores_dict: Dict[str, np.ndarray],router :str,
                                  **kwargs) -> Dict:
        all_results = {}

        for dataset_name, data in dataset_results.items():
            if dataset_name not in router_scores_dict:
                print(f"Warning: No router scores for {dataset_name}")
                continue

            print(f"Evaluating metrics for {dataset_name}...")

            small_results = data['small_results']
            large_results = data['large_results']
            router_scores = router_scores_dict[dataset_name]
            dataset_type = data.get("dataset_type", "general")

            results = self.evaluator.evaluate_from_results(
                small_results, large_results, router_scores, dataset_type=dataset_type, **kwargs
            )

            # Save individual dataset results
            out_file =Path(f"{dataset_name}_{router}")
            dataset_dir = self.output_dir / out_file
            
            dataset_dir.mkdir(exist_ok=True)
            self.evaluator.save_results(results, dataset_dir / "metrics.json")

            # Plot adaptive curve
            plot_path = dataset_dir / "adaptive_curve.png"
            self.evaluator.plot_adaptive_curve(results['adaptive_metrics'], str(plot_path))

            all_results[dataset_name] = results

            # Print summary
            reliable = results['reliable_metrics']
            adaptive = results['adaptive_metrics']
            print(f"{dataset_name}: AUROC={reliable['auroc']:.4f}, "
                  f"LPM={adaptive['LPM']:.4f}, HPM={adaptive['HPM']:.4f}")

        # Save summary
        
        summary_path = self.output_dir / "summary.json"
        self.evaluator.save_results(all_results, str(summary_path))

        print(f"\nResults saved to: {self.output_dir}")
        return all_results


def evaluate_single_dataset(small_scores: np.ndarray, large_scores: np.ndarray,
                           router_scores: np.ndarray, dataset_type: str = 'general', **kwargs) -> Dict:
    evaluator = MetricEvaluator()
    return evaluator.evaluate_from_scores(small_scores, large_scores, router_scores, dataset_type, **kwargs)


def plot_results(adaptive_metrics: Dict, save_path: Optional[str] = None):
    evaluator = MetricEvaluator()
    evaluator.plot_adaptive_curve(adaptive_metrics, save_path)
