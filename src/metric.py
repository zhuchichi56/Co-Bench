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
                 router_scores: np.ndarray) -> Dict:
        min_len = min(len(small_scores), len(large_scores), len(router_scores))
        small_scores = small_scores[:min_len]
        large_scores = large_scores[:min_len]
        router_scores = router_scores[:min_len]
        if np.any(large_scores > 1) or np.any(small_scores > 1):
            # 如果有分数大于1，就看小模型是否 >= 大模型
            labels = (small_scores >= large_scores).astype(int)
        else:
            # 原来的逻辑：处理分数差值
            acc_diff = 1 - small_scores
            labels = (acc_diff <= 0).astype(int)
       

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
                 call_rate_param: float = 0.1) -> Dict:
        min_len = min(len(small_scores), len(large_scores), len(router_scores))
        small_scores = small_scores[:min_len]
        large_scores = large_scores[:min_len]
        router_scores = router_scores[:min_len] 
        small_mean = small_scores.mean()
        large_mean = large_scores.mean()
        
        
        
        #先升序排序，再取对应百分比call rate调用大模型，若遇到相同分数所占百分比大于call rate，随机选择对应比例
        sorted_indices = np.argsort(router_scores)
        sorted_small = small_scores[sorted_indices]
        sorted_large = large_scores[sorted_indices]
        sorted_router = router_scores[sorted_indices]
        n_samples = len(router_scores)
        call_rates = np.linspace(0, 1, 101)  
        accuracies = []
        # 记录每个分数的起止位置
        unique_scores, score_counts = np.unique(sorted_router, return_counts=True)
        score_ranges = []
        start = 0
        for count in score_counts:
            end = start + count
            score_ranges.append((start, end))
            start = end
        for rate in call_rates:
            n_large = int(rate * n_samples)  # 需调用大模型的样本数
            
            if n_large == 0:
               
                mean_acc = np.mean(sorted_small)
            elif n_large == n_samples:
                
                mean_acc = np.mean(sorted_large)
            else:
                selected_indices = []
                remaining = n_large
                
                for score, (start, end) in zip(unique_scores, score_ranges):
                    if remaining <= 0:
                        break
                    available = end - start
                    if available <= remaining:
                        selected_indices.extend(range(start, end))
                        remaining -= available
                    else:
                        selected_indices.extend(
                            np.random.choice(range(start, end), remaining, replace=False)
                        )
                        remaining = 0
                large_acc = np.mean(sorted_large[selected_indices])
                small_acc = np.mean(np.delete(sorted_small, selected_indices))
                mean_acc = (large_acc * n_large + small_acc * (n_samples - n_large)) / n_samples
            
            accuracies.append(mean_acc)
        accuracies = np.array(accuracies)
        

        # Calculate LPM
        # low_call_mask = call_rates <= call_rate_param
        # LPM = np.mean(accuracies[low_call_mask]) if np.any(low_call_mask) else accuracies[0]
        max_idx = np.searchsorted(call_rates, call_rate_param, side='left')
        LPM =np.mean(accuracies[:max_idx])

        # Calculate HPM
        low_acc_band = small_mean+(large_mean-small_mean)*recovery_rate_band[0]
        high_acc_band = small_mean + (large_mean-small_mean)*recovery_rate_band[1]
        print(f"low{low_acc_band}")
        print(f"high{high_acc_band}")
        target_mask = (accuracies >= low_acc_band) & (accuracies <= high_acc_band)
        
        if np.any(target_mask):
            target_call_rates = call_rates[target_mask]
            HPM = 1 - np.mean(target_call_rates)
        else:
            HPM = 0.0
        # Calculate MPM
        lpm_threshold = accuracies[max_idx]  # LPM的阈值点
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
            'call_rate_param': call_rate_param,
            'max_accuracy': np.max(accuracies),
            'min_accuracy': np.min(accuracies)
        }


class MetricEvaluator:
    def __init__(self):
        self.reliable_metrics = ReliableMetrics()
        self.adaptive_metrics = AdaptiveMetrics()

    def evaluate_from_results(self, small_results: List[Dict], large_results: List[Dict],
                            router_scores: np.ndarray, **kwargs) -> Dict:
        small_scores = np.array([r['score'] for r in small_results])
        large_scores = np.array([r['score'] for r in large_results])

        reliable = self.reliable_metrics.calculate(small_scores, large_scores, router_scores)
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
                           router_scores: np.ndarray, **kwargs) -> Dict:
        reliable = self.reliable_metrics.calculate(small_scores, large_scores, router_scores)
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
        plt.figure(figsize=(10, 6))

        call_rates = adaptive_metrics['call_rates']
        accuracies = adaptive_metrics['accuracies']
        large_mean = adaptive_metrics['large_mean']
        small_mean = adaptive_metrics['small_mean']
        

        plt.plot(call_rates * 100, accuracies * 100, 'b-', linewidth=2, label='Accuracy Curve')
        plt.axhspan(small_mean * 100, large_mean * 100,
                   alpha=0.2, color='red', label=f'Target Band [{small_mean*100:.0f}%, {large_mean*100:.0f}%]')

        max_acc_idx = np.argmax(accuracies)
        plt.plot(call_rates[max_acc_idx] * 100, accuracies[max_acc_idx] * 100,
                'ro', markersize=8, label=f'Max Accuracy ({accuracies[max_acc_idx]*100:.1f}%)')

        plt.xlabel('Large Model Call Rate (%)')
        plt.ylabel('Overall Accuracy (%)')
        plt.title('Adaptive Framework Accuracy Curve')
        plt.legend()
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

            results = self.evaluator.evaluate_from_results(
                small_results, large_results, router_scores, **kwargs
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
                           router_scores: np.ndarray, **kwargs) -> Dict:
    evaluator = MetricEvaluator()
    return evaluator.evaluate_from_scores(small_scores, large_scores, router_scores, **kwargs)


def plot_results(adaptive_metrics: Dict, save_path: Optional[str] = None):
    evaluator = MetricEvaluator()
    evaluator.plot_adaptive_curve(adaptive_metrics, save_path)