# generate_charts.py

import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_metrics(file_path):
    """Loads training metrics from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        epochs = data.get("epochs", [])
        last_epoch = epochs[-1] if epochs else {}
        metrics = {
            "Training Time (s)": last_epoch.get("epoch_wall_time_sec", 0),
            "Tokens/sec": last_epoch.get("tokens_per_sec_global", 0),
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

    # --- Chart 2: Training Throughput ---
    fig, ax = plt.subplots(figsize=(8, 6))
    throughput = [baseline_metrics["Tokens/sec"], optimized_metrics["Tokens/sec"]]
    bars = ax.bar(labels, throughput, color=['#4c72b0', '#55a868'])
    ax.set_ylabel('Tokens/sec')
    ax.set_title('Training Throughput per Epoch', fontsize=16, fontweight='bold')
    ax.bar_label(bars, fmt='%.0f tok/s')
    plt.tight_layout()
    throughput_chart_path = os.path.join(output_dir, 'training_throughput_comparison.png')
    plt.savefig(throughput_chart_path)
    print(f"Saved training throughput chart to {throughput_chart_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate performance charts from training metric files.")
    parser.add_argument('--baseline-json', type=str, required=True, help="Path to the baseline run's training_metrics.json")
    parser.add_argument('--optimized-json', type=str, required=True, help="Path to the optimized run's training_metrics.json")
    args = parser.parse_args()

    baseline_data = load_metrics(args.baseline_json)
    optimized_data = load_metrics(args.optimized_json)

    generate_plots(baseline_data, optimized_data)
