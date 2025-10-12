# generate_charts.py

import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_metrics(file_path):
    """Loads metrics from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract the last value for metrics that are lists
        metrics = {
            "Training Time (s)": data.get("train_time_per_epoch", [0])[-1],
            "Inference Latency (s)": data.get("inference_latency", 0),
            "Inference Throughput (tokens/s)": data.get("inference_throughput", 0)
        }
        return metrics
    except (FileNotFoundError, json.JSONDecodeError, IndexError) as e:
        print(f"Error reading or parsing {file_path}: {e}")
        return None

def generate_plots(baseline_metrics, optimized_metrics):
    """Generates and saves comparison charts."""
    if not baseline_metrics or not optimized_metrics:
        print("Missing metrics data. Cannot generate charts.")
        return

    output_dir = 'docs'
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Chart 1: Training Time ---
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = ['Baseline (1x T4)', 'Optimized (4x T4s + DeepSpeed)']
    times = [baseline_metrics["Training Time (s)"], optimized_metrics["Training Time (s)"]]
    bars = ax.bar(labels, times, color=['#4c72b0', '#55a868'])
    
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Training Time Reduction per Epoch', fontsize=16, fontweight='bold')
    ax.bar_label(bars, fmt='%.0f s')
    
    improvement = ((times[0] - times[1]) / times[0]) * 100
    plt.text(0.5, 0.5, f'{improvement:.0f}% Time Reduction',
             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
             fontsize=18, color='white', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="#55a868", ec="b", lw=2))
             
    plt.tight_layout()
    training_chart_path = os.path.join(output_dir, 'training_time_comparison.png')
    plt.savefig(training_chart_path)
    print(f"Saved training time chart to {training_chart_path}")
    plt.close()

    # --- Chart 2: Inference Performance ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.bar(labels, [baseline_metrics["Inference Latency (s)"], optimized_metrics["Inference Latency (s)"]], color=['#4c72b0', '#55a868'])
    ax1.set_ylabel('Latency (seconds)')
    ax1.set_title('Inference Latency Improvement', fontsize=14, fontweight='bold')
    ax1.bar_label(ax1.containers[0], fmt='%.2f s')
    
    ax2.bar(labels, [baseline_metrics["Inference Throughput (tokens/s)"], optimized_metrics["Inference Throughput (tokens/s)"]], color=['#4c72b0', '#55a868'])
    ax2.set_ylabel('Throughput (tokens/second)')
    ax2.set_title('Inference Throughput Boost', fontsize=14, fontweight='bold')
    ax2.bar_label(ax2.containers[0], fmt='%.1f tok/s')
    
    fig.suptitle('Inference Performance Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    inference_chart_path = os.path.join(output_dir, 'inference_performance_comparison.png')
    plt.savefig(inference_chart_path)
    print(f"Saved inference performance chart to {inference_chart_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate performance charts from training metric files.")
    parser.add_argument('--baseline-json', type=str, required=True, help="Path to the baseline run's training_metrics.json")
    parser.add_argument('--optimized-json', type=str, required=True, help="Path to the optimized run's training_metrics.json")
    args = parser.parse_args()

    baseline_data = load_metrics(args.baseline_json)
    optimized_data = load_metrics(args.optimized_json)

    generate_plots(baseline_data, optimized_data)
