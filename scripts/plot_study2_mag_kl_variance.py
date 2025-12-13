"""
Study 2: Variance of mag_kl
Computes and plots rolling variance and standard deviation across seeds at each timestep.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd


def load_log_file(log_file):
    """Load a JSON log file and extract step and mag_kl data."""
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    steps = []
    mag_kl = []
    
    for entry in data.get('train', []):
        if 'step' in entry and 'mag_kl' in entry:
            steps.append(entry['step'])
            mag_kl.append(entry['mag_kl'])
    
    return np.array(steps), np.array(mag_kl)


def plot_mag_kl_variance(log_files, window_size=100, output_file=None, labels=None):
    """
    Plot rolling variance and std across seeds for mag_kl.
    
    Args:
        log_files: List of paths to log JSON files (each representing a different seed)
        window_size: Window size for rolling variance calculation
        output_file: Optional path to save the plot
        labels: Optional list of labels for each log file
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Load all data
    all_data = {}
    for i, log_file in enumerate(log_files):
        steps, mag_kl = load_log_file(log_file)
        label = labels[i] if labels and i < len(labels) else Path(log_file).stem
        all_data[label] = pd.DataFrame({'step': steps, 'mag_kl': mag_kl})
        
        # Plot individual trajectories in the background
        axes[0].plot(steps, mag_kl, alpha=0.2, color='gray', linewidth=0.5)
    
    # Compute rolling variance for each seed
    for label, df in all_data.items():
        df['rolling_var'] = df['mag_kl'].rolling(window=window_size, min_periods=1).var()
        axes[1].plot(df['step'], df['rolling_var'], label=label, alpha=0.7)
    
    # Compute mean and std across seeds at each timestep
    # Align data by step
    all_steps = sorted(set().union(*[set(df['step']) for df in all_data.values()]))
    
    mean_mag_kl = []
    std_mag_kl = []
    
    for step in all_steps:
        values_at_step = []
        for df in all_data.values():
            step_data = df[df['step'] == step]
            if not step_data.empty:
                values_at_step.append(step_data['mag_kl'].values[0])
        
        if values_at_step:
            mean_mag_kl.append(np.mean(values_at_step))
            std_mag_kl.append(np.std(values_at_step))
        else:
            mean_mag_kl.append(np.nan)
            std_mag_kl.append(np.nan)
    
    mean_mag_kl = np.array(mean_mag_kl)
    std_mag_kl = np.array(std_mag_kl)
    all_steps = np.array(all_steps)
    
    # Plot mean with std bands
    axes[0].plot(all_steps, mean_mag_kl, 'b-', linewidth=2, label='Mean across seeds')
    axes[0].fill_between(all_steps, 
                          mean_mag_kl - std_mag_kl, 
                          mean_mag_kl + std_mag_kl, 
                          alpha=0.3, label='±1 std')
    
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('mag_kl')
    axes[0].set_title('Study 2a: mag_kl with Standard Deviation Across Seeds')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('Rolling Variance')
    axes[1].set_title(f'Study 2b: Rolling Variance of mag_kl (window={window_size})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot variance of mag_kl')
    parser.add_argument('log_files', nargs='+', help='Path(s) to log JSON file(s) (each should be a different seed)')
    parser.add_argument('--output', '-o', help='Output file path for the plot')
    parser.add_argument('--labels', '-l', nargs='+', help='Labels for each log file')
    parser.add_argument('--window', '-w', type=int, default=100, help='Window size for rolling variance (default: 100)')
    
    args = parser.parse_args()
    
    plot_mag_kl_variance(args.log_files, args.window, args.output, args.labels)


if __name__ == '__main__':
    main()
