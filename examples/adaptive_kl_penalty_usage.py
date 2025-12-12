#!/usr/bin/env python3
"""
Example script demonstrating Adaptive KL Penalty usage.

This script shows how to configure and monitor the adaptive KL penalty mechanism.
"""

# Example 1: Basic usage with default adaptive settings
# The default configuration already has adaptive KL penalty enabled:
# - adaptive_kl_penalty: true
# - target_kl: 0.01
# - kl_adjustment_factor: 1.5
# - min_mag_coef: 0.001
# - max_mag_coef: 1.0

# To run with default adaptive settings:
# python scripts/run_training.py algorithm=nash_pg

# Example 2: Disable adaptive KL penalty (use fixed mag_coef)
# python scripts/run_training.py algorithm=nash_pg algorithm.adaptive_kl_penalty=false

# Example 3: Custom adaptive settings - stricter KL constraint
# python scripts/run_training.py algorithm=nash_pg \
#     algorithm.target_kl=0.005 \
#     algorithm.kl_adjustment_factor=2.0

# Example 4: Custom adaptive settings - more lenient KL constraint
# python scripts/run_training.py algorithm=nash_pg \
#     algorithm.target_kl=0.02 \
#     algorithm.kl_adjustment_factor=1.2

# Example 5: Custom bounds for mag_coef
# python scripts/run_training.py algorithm=nash_pg \
#     algorithm.min_mag_coef=0.01 \
#     algorithm.max_mag_coef=0.5

# Monitoring the adaptive mechanism:
# When you run training, watch these metrics in your logs/tensorboard:
# - mag_kl: Shows the actual KL divergence between current and magnetic policy
# - mag_coef: Shows how the penalty coefficient adapts over time
# - approx_kl: Shows KL between consecutive policies (standard PPO metric)

# Expected behavior:
# - mag_kl should oscillate around target_kl (e.g., 0.01)
# - mag_coef should adjust up when mag_kl > target_kl
# - mag_coef should adjust down when mag_kl < target_kl
# - mag_coef stays within [min_mag_coef, max_mag_coef] bounds


def explain_parameters():
    """Explain each parameter of the adaptive KL penalty mechanism."""
    
    params = {
        "adaptive_kl_penalty": {
            "type": "bool",
            "default": True,
            "description": "Enable or disable the adaptive KL penalty mechanism. "
                         "When False, uses fixed mag_coef value."
        },
        "target_kl": {
            "type": "float",
            "default": 0.01,
            "description": "Target KL divergence between current and magnetic policy. "
                         "Lower values enforce stricter similarity. "
                         "Typical range: 0.005 - 0.02"
        },
        "kl_adjustment_factor": {
            "type": "float",
            "default": 1.5,
            "description": "Multiplicative factor for adjusting mag_coef. "
                         "Higher values = more aggressive adjustments. "
                         "Typical range: 1.2 - 2.0"
        },
        "min_mag_coef": {
            "type": "float",
            "default": 0.001,
            "description": "Minimum allowed value for mag_coef. "
                         "Prevents penalty from becoming too weak."
        },
        "max_mag_coef": {
            "type": "float",
            "default": 1.0,
            "description": "Maximum allowed value for mag_coef. "
                         "Prevents penalty from becoming too strong."
        }
    }
    
    print("Adaptive KL Penalty Parameters")
    print("=" * 70)
    for name, info in params.items():
        print(f"\n{name}:")
        print(f"  Type: {info['type']}")
        print(f"  Default: {info['default']}")
        print(f"  Description: {info['description']}")


def compare_fixed_vs_adaptive():
    """Compare fixed vs adaptive mag_coef approaches."""
    
    comparison = """
    Fixed mag_coef (Original Paper):
    ✗ Requires manual tuning per environment
    ✗ May be too strong/weak during training
    ✗ Same value used throughout training
    ✓ Simple and predictable
    
    Adaptive KL Penalty (This Implementation):
    ✓ Automatically tunes to environment
    ✓ Adapts to different training phases
    ✓ Maintains target KL divergence
    ✓ No manual hyperparameter search needed
    ✗ Slightly more complex (5 parameters vs 1)
    
    Recommendation:
    - Use adaptive by default for most cases
    - Use fixed only if you have a well-tuned value
    - Monitor mag_coef in logs to verify adaptation
    """
    print(comparison)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Adaptive KL Penalty - Usage Guide")
    print("="*70 + "\n")
    
    explain_parameters()
    
    print("\n" + "="*70)
    print("Fixed vs Adaptive Comparison")
    print("="*70 + "\n")
    
    compare_fixed_vs_adaptive()
    
    print("\n" + "="*70)
    print("Quick Start Examples")
    print("="*70 + "\n")
    
    print("1. Use default adaptive settings:")
    print("   python scripts/run_training.py algorithm=nash_pg\n")
    
    print("2. Disable adaptive (use fixed mag_coef=0.2):")
    print("   python scripts/run_training.py algorithm=nash_pg \\")
    print("       algorithm.adaptive_kl_penalty=false\n")
    
    print("3. Stricter KL constraint (more conservative):")
    print("   python scripts/run_training.py algorithm=nash_pg \\")
    print("       algorithm.target_kl=0.005\n")
    
    print("4. More lenient KL constraint (more exploratory):")
    print("   python scripts/run_training.py algorithm=nash_pg \\")
    print("       algorithm.target_kl=0.02\n")
