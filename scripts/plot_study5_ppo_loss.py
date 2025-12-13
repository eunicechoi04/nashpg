"""
Study 5: ppo_loss vs Training Steps
Plots ppo_loss over time across multiple log files.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_log_file(log_file):
    """Load a JSON log file and extract step and ppo_loss data."""
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    steps = []
    ppo_loss = []
    
    for entry in data.get('train', []):
        if 'step' in entry and 'ppo_loss' in entry:
            steps.append(entry['step'])
            ppo_loss.append(entry['ppo_loss'])
    
    return np.array(steps), np.array(ppo_loss)


def plot_ppo_loss(log_files, output_file=None, labels=None):
    """
    Plot ppo_loss vs training steps for multiple log files.
    
    Args:
        log_files: List of paths to log JSON files
        output_file: Optional path to save the plot
        labels: Optional list of labels for each log file
    """
    plt.figure(figsize=(10, 6))
    
    for i, log_file in enumerate(log_files):
        steps, ppo_loss = load_log_file(log_file)
        label = labels[i] if labels and i < len(labels) else Path(log_file).stem
        plt.plot(steps, ppo_loss, label=label, alpha=0.7)
    
    plt.xlabel('Training Steps')
    plt.ylabel('ppo_loss')
    plt.title('Study 5: PPO Loss vs Training Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot ppo_loss vs training steps')
    parser.add_argument('log_files', nargs='+', help='Path(s) to log JSON file(s)')
    parser.add_argument('--output', '-o', help='Output file path for the plot')
    parser.add_argument('--labels', '-l', nargs='+', help='Labels for each log file')
    
    args = parser.parse_args()
    
    plot_ppo_loss(args.log_files, args.output, args.labels)


if __name__ == '__main__':
    main()
