"""
Study 4: clip_frac vs Training Steps
Plots clip_frac over time across multiple log files.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_log_file(log_file):
    """Load a JSON log file and extract step and clip_frac data."""
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    steps = []
    clip_frac = []
    
    for entry in data.get('train', []):
        if 'step' in entry and 'clip_frac' in entry:
            steps.append(entry['step'])
            clip_frac.append(entry['clip_frac'])
    
    return np.array(steps), np.array(clip_frac)


def plot_clip_frac(log_files, output_file=None, labels=None):
    """
    Plot clip_frac vs training steps for multiple log files.
    
    Args:
        log_files: List of paths to log JSON files
        output_file: Optional path to save the plot
        labels: Optional list of labels for each log file
    """
    plt.figure(figsize=(10, 6))
    
    for i, log_file in enumerate(log_files):
        steps, clip_frac = load_log_file(log_file)
        label = labels[i] if labels and i < len(labels) else Path(log_file).stem
        plt.plot(steps, clip_frac, label=label, alpha=0.7)
    
    plt.xlabel('Training Steps')
    plt.ylabel('clip_frac')
    plt.title('Study 4: PPO Clipping Fraction vs Training Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot clip_frac vs training steps')
    parser.add_argument('log_files', nargs='+', help='Path(s) to log JSON file(s)')
    parser.add_argument('--output', '-o', help='Output file path for the plot')
    parser.add_argument('--labels', '-l', nargs='+', help='Labels for each log file')
    
    args = parser.parse_args()
    
    plot_clip_frac(args.log_files, args.output, args.labels)


if __name__ == '__main__':
    main()