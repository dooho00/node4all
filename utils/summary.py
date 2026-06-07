import os
import datetime
import statistics
from collections import defaultdict
from typing import Dict, List, Sequence

import pandas as pd

def print_final_summary(all_summaries, dataset_list, results_dir=None):
    """Save final summary tables for all datasets to both text and CSV files."""
    
    # Use provided results_dir or create a new timestamped one
    if results_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = f'results/{timestamp}'
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filenames
    txt_filename = f'{results_dir}/summary_all_datasets.txt'
    
    # Save separate CSV files for each experiment
    save_separate_experiment_csvs(all_summaries, dataset_list, results_dir)
    
    # Then save text format (keeping original functionality)
    save_results_to_text(all_summaries, dataset_list, txt_filename, results_dir)
    
    print(f"Summary saved to:")
    print(f"  Text: {txt_filename}")
    print(f"  CSV files saved in: {results_dir}")

def save_separate_experiment_csvs(all_summaries, dataset_list, results_dir):
    """Save results in separate CSV files for each experiment type."""
    
    # 1. Adaptation Results
    adaptation_data = []
    for dataset in dataset_list:
        if dataset not in all_summaries:
            continue
        s = all_summaries[dataset]
        if 'Agg_best_val' in s:
            row = {
                'Dataset': dataset,
                'Agg_Val_Acc': s['Agg_best_val'] * 100,
                'Agg_Test_Acc': s['Agg_best_test'] * 100,
                'Agg_Best_H': s.get('best_agg_h', ''),
            }
            if 'Agg_best_val_std' in s:
                row['Agg_Val_Acc_std'] = s['Agg_best_val_std'] * 100
            if 'Agg_best_test_std' in s:
                row['Agg_Test_Acc_std'] = s['Agg_best_test_std'] * 100
            adaptation_data.append(row)
    
    if adaptation_data:
        df_lp = pd.DataFrame(adaptation_data)
        df_lp = df_lp.round(2)
        df_lp.to_csv(f'{results_dir}/adaptation_results.csv', index=False)

def save_results_to_text(all_summaries, dataset_list, txt_filename, results_dir):
    """Save results in original text format."""
    
    # Collect all output in a list
    output_lines = []
    
    # Main results summary (Adaptation)
    output_lines.append('\n=== Adaptation Summary Across All Datasets ===')
    header = (
        f"{'Dataset':<15} | {'Agg Val (%)':<12} | {'Val Std (%)':<12} | "
        f"{'Agg h':<6} | {'Agg Test (%)':<12} | {'Test Std (%)':<12}"
    )
    output_lines.append(header)
    output_lines.append('-' * len(header))
    
    for ds in dataset_list:
        if ds in all_summaries:
            s = all_summaries[ds]
            val_std = s.get('Agg_best_val_std', 0.0)
            test_std = s.get('Agg_best_test_std', 0.0)
            output_lines.append(
                f"{ds:<15} | "
                f"{100 * s['Agg_best_val']:<12.2f} | "
                f"{100 * val_std:<12.2f} | "
                f"{str(s['best_agg_h']):<6} | "
                f"{100 * s['Agg_best_test']:<12.2f} | "
                f"{100 * test_std:<12.2f}"
            )

    # Average across all datasets
    avg_agg_val = sum(s['Agg_best_val'] for s in all_summaries.values() if 'Agg_best_val' in s) / len(all_summaries)
    avg_agg_test = sum(s['Agg_best_test'] for s in all_summaries.values() if 'Agg_best_test' in s) / len(all_summaries)
    
    output_lines.append(
        f"{'Average':<15} | "
        f"{100 * avg_agg_val:<12.2f} | "
        f"{'':<12} | "
        f"{'':<6} | "
        f"{100 * avg_agg_test:<12.2f} | "
        f"{'':<12}"
    )
    
    # Add a note about the results directory
    output_lines.append(f"\nResults saved to: {results_dir}")
    
    # Write all output to file
    with open(txt_filename, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Summary saved to: {txt_filename}")



def save_stage2_overall_summary(
    results_dir: str,
    node_results: Dict[str, Dict] = None,
    graph_results: Dict[str, Dict] = None,
):
    """Write a single summary that aggregates all Stage 2 tasks."""

    if not results_dir:
        return

    node_results = node_results or {}
    graph_results = graph_results or {}

    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, 'stage2_summary.txt')

    lines: List[str] = []
    lines.append('=== Stage 2 Cross-Task Summary ===')
    lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append('')

    lines.extend(_format_node_section(node_results))
    lines.extend(_format_graph_section(graph_results))

    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'Stage 2 overall summary saved to: {filename}')


def _format_node_section(node_results: Dict[str, Dict]) -> List[str]:
    if not node_results:
        return ['Node level task: no results.', '']

    lines = ['Node Classification Task']
    header = "{:<25} | {:<16} | {:<16} | {:<5}".format('Dataset', 'Val (%)', 'Test (%)', 'h')
    lines.append(header)
    lines.append('-' * len(header))
    heldout_values: List[float] = []

    for dataset in sorted(node_results):
        summary = node_results.get(dataset, {}) or {}
        val = summary.get('Agg_best_val')
        test = summary.get('Agg_best_test')
        val_std = summary.get('Agg_best_val_std')
        test_std = summary.get('Agg_best_test_std')
        best_h = summary.get('best_agg_h', '')
        if isinstance(val, (int, float)):
            val_str = f"{100 * val:.2f}"
            if isinstance(val_std, (int, float)):
                val_str = f"{val_str} +/- {100 * val_std:.2f}"
        else:
            val_str = 'n/a'
        if isinstance(test, (int, float)):
            test_str = f"{100 * test:.2f}"
            if isinstance(test_std, (int, float)):
                test_str = f"{test_str} +/- {100 * test_std:.2f}"
        else:
            test_str = 'n/a'
        lines.append("{:<25} | {:<16} | {:<16} | {:<5}".format(dataset, val_str, test_str, best_h))
        if isinstance(test, (int, float)):
            heldout_values.append(float(test))

    lines.append(_format_holdout_line('Held-out mean +/- std (test %)', heldout_values, multiplier=100))
    lines.append('')
    return lines


def _format_graph_section(graph_results: Dict[str, Dict]) -> List[str]:
    if not graph_results:
        return ['Graph level task: no results.', '']

    lines = ['Graph Level Task']
    header = "{:<25} | {:<12} | {:<10} | {:<10}".format('Dataset', 'Metric', 'Val', 'Test')
    lines.append(header)
    lines.append('-' * len(header))
    metric_groups = defaultdict(list)

    for dataset in sorted(graph_results):
        summary = graph_results.get(dataset, {}) or {}
        metric_name = summary.get('metric_name') or 'metric'
        val = summary.get('val_metric')
        test = summary.get('test_metric')
        val_str = f"{val:.4f}" if isinstance(val, (int, float)) else 'n/a'
        test_str = f"{test:.4f}" if isinstance(test, (int, float)) else 'n/a'
        lines.append("{:<25} | {:<12} | {:<10} | {:<10}".format(dataset, metric_name, val_str, test_str))
        if isinstance(test, (int, float)):
            metric_groups[metric_name].append(float(test))

    if metric_groups:
        lines.append('Held-out mean +/- std per metric (test):')
        for metric_name in sorted(metric_groups):
            stats = _compute_mean_std(metric_groups[metric_name])
            if stats is None:
                continue
            mean_val, std_val = stats
            lines.append(f"  {metric_name}: {mean_val:.4f} +/- {std_val:.4f}")
    else:
        lines.append('Held-out mean +/- std per metric (test): n/a')

    lines.append('')
    return lines


def _format_holdout_line(label: str, values: Sequence[float], multiplier: float = 1.0) -> str:
    stats = _compute_mean_std(values)
    if stats is None:
        return f'{label}: n/a'
    mean_val, std_val = stats
    return f"{label}: {mean_val * multiplier:.2f} +/- {std_val * multiplier:.2f}"


def _compute_mean_std(values: Sequence[float]):
    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    if not numeric:
        return None
    if len(numeric) == 1:
        return numeric[0], 0.0
    return statistics.mean(numeric), statistics.pstdev(numeric)
