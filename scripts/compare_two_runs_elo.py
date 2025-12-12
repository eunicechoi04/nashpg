#!/usr/bin/env python3
"""
Compare ELO ratings between two specific Nash-PG runs on the same game.

This script loads two different Nash-PG training runs (e.g., adaptive KL vs fixed KL)
and computes head-to-head ELO ratings by having them play against each other
at each checkpoint step.

Usage:
  uv run scripts/compare_two_runs_elo.py \
      --env kuhn_poker \
      --run1 "kuhn_poker/nash_pg_adaptive" \
      --run2 "kuhn_poker/nash_pg_fixed" \
      --label1 "Adaptive KL" \
      --label2 "Fixed KL"
      
  uv run scripts/compare_two_runs_elo.py \
      --env kuhn_poker \
      --run1 "kuhn_poker/nash_pg_adaptive" \
      --run2 "kuhn_poker/nash_pg_fixed" \
      --games-per-pairing 100 \
      --output results/elo_comparison.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Environment setup
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose logging
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('orbax').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore', message='.*Sharding info not provided when restoring.*')

from flax import nnx
from envs import create_env
from agents import create_agent
from scripts.compute_elo.elo_agents import AgentInfo


def find_checkpoint_dir(run_name: str) -> Optional[Path]:
    """Find the checkpoint directory for a given run name.
    
    Args:
        run_name: Run name (e.g., "kuhn_poker/nash_pg_adaptive")
        
    Returns:
        Path to checkpoint directory if found, None otherwise
    """
    checkpoint_path = Path("checkpoints") / run_name
    
    if checkpoint_path.exists():
        return checkpoint_path
    
    logger.error(f"Checkpoint directory not found: {checkpoint_path}")
    return None


def get_checkpoint_steps(checkpoint_dir: Path) -> List[int]:
    """Get all available checkpoint steps from a directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        Sorted list of checkpoint steps
    """
    steps = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            steps.append(int(item.name))
    
    return sorted(steps)


def load_agent_at_step(env_name: str, checkpoint_dir: Path, step: int, rngs: nnx.Rngs) -> Optional[object]:
    """Load an agent from a specific checkpoint step.
    
    Args:
        env_name: Environment name
        checkpoint_dir: Path to checkpoint directory
        step: Checkpoint step to load
        rngs: Random number generator
        
    Returns:
        Loaded agent or None if loading fails
    """
    try:
        agent = create_agent(env_name, key=rngs)
        checkpoint_path = checkpoint_dir / str(step)
        agent.load_checkpoint(checkpoint_path)
        return agent
    except Exception as e:
        logger.error(f"Failed to load agent at step {step} from {checkpoint_dir}: {e}")
        return None


def play_game(env, agent1, agent2, key):
    """Play a single game between two agents.
    
    Args:
        env: Game environment
        agent1: First agent
        agent2: Second agent
        key: Random key
        
    Returns:
        Tuple of (agent1_reward, agent2_reward)
    """
    import jax
    import jax.numpy as jnp
    
    # Reset environment
    key, reset_key = jax.random.split(key)
    state, timestep = env.reset(reset_key)
    
    agents = [agent1, agent2]
    
    while not timestep.last():
        # Get current player
        current_player = int(state.current_player)
        current_agent = agents[current_player]
        
        # Get action from agent
        action = current_agent.get_action(
            timestep.observation,
            timestep.legal_action_mask,
            deterministic=False
        )
        
        # Step environment
        key, step_key = jax.random.split(key)
        state, timestep = env.step(state, action, step_key)
    
    # Return rewards for both players
    return float(timestep.rewards[0]), float(timestep.rewards[1])


def compute_head_to_head_winrate(
    env,
    agent1,
    agent2,
    num_games: int,
    key
) -> Dict:
    """Compute head-to-head win rate between two agents.
    
    Args:
        env: Game environment
        agent1: First agent
        agent2: Second agent
        num_games: Number of games to play
        key: Random key
        
    Returns:
        Dictionary with win/loss/draw statistics
    """
    import jax
    
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    
    for _ in range(num_games):
        key, game_key = jax.random.split(key)
        r1, r2 = play_game(env, agent1, agent2, game_key)
        
        if r1 > r2:
            agent1_wins += 1
        elif r2 > r1:
            agent2_wins += 1
        else:
            draws += 1
    
    return {
        "agent1_wins": agent1_wins,
        "agent2_wins": agent2_wins,
        "draws": draws,
        "agent1_winrate": agent1_wins / num_games,
        "agent2_winrate": agent2_wins / num_games,
        "draw_rate": draws / num_games
    }


def compare_elo_over_training(
    env_name: str,
    run1_name: str,
    run2_name: str,
    label1: str = "Run 1",
    label2: str = "Run 2",
    games_per_pairing: int = 50,
    output_file: Optional[str] = None
) -> Dict:
    """Compare ELO ratings between two runs over training.
    
    Args:
        env_name: Environment name
        run1_name: First run name
        run2_name: Second run name
        label1: Label for first run
        label2: Label for second run
        games_per_pairing: Number of games to play per comparison
        output_file: Optional output file to save results
        
    Returns:
        Dictionary with comparison results
    """
    import jax
    
    # Create environment and RNG
    env = create_env(env_name)
    rngs = nnx.Rngs(0)
    key = jax.random.key(0)
    
    # Find checkpoint directories
    checkpoint1 = find_checkpoint_dir(run1_name)
    checkpoint2 = find_checkpoint_dir(run2_name)
    
    if not checkpoint1 or not checkpoint2:
        logger.error("One or both checkpoint directories not found!")
        return {}
    
    # Get available steps (use common steps between both runs)
    steps1 = set(get_checkpoint_steps(checkpoint1))
    steps2 = set(get_checkpoint_steps(checkpoint2))
    common_steps = sorted(steps1.intersection(steps2))
    
    if not common_steps:
        logger.error("No common checkpoint steps found between the two runs!")
        return {}
    
    logger.info(f"Found {len(common_steps)} common checkpoint steps")
    logger.info(f"Steps: {common_steps}")
    
    # Compare at each step
    comparison_results = []
    
    for step in common_steps:
        logger.info(f"\nComparing at step {step}...")
        
        # Load agents
        agent1 = load_agent_at_step(env_name, checkpoint1, step, rngs)
        agent2 = load_agent_at_step(env_name, checkpoint2, step, rngs)
        
        if not agent1 or not agent2:
            logger.warning(f"Skipping step {step} due to loading failure")
            continue
        
        # Compute head-to-head statistics
        key, subkey = jax.random.split(key)
        stats = compute_head_to_head_winrate(
            env, agent1, agent2, games_per_pairing, subkey
        )
        
        result = {
            "step": step,
            **stats
        }
        comparison_results.append(result)
        
        logger.info(f"  {label1} win rate: {stats['agent1_winrate']:.3f}")
        logger.info(f"  {label2} win rate: {stats['agent2_winrate']:.3f}")
        logger.info(f"  Draw rate: {stats['draw_rate']:.3f}")
    
    # Compute overall statistics
    if comparison_results:
        avg_run1_winrate = sum(r['agent1_winrate'] for r in comparison_results) / len(comparison_results)
        avg_run2_winrate = sum(r['agent2_winrate'] for r in comparison_results) / len(comparison_results)
        
        final_result = comparison_results[-1]
    else:
        avg_run1_winrate = 0
        avg_run2_winrate = 0
        final_result = {}
    
    comparison = {
        "run1": {
            "name": run1_name,
            "label": label1,
            "avg_winrate": avg_run1_winrate,
            "final_winrate": final_result.get('agent1_winrate', 0)
        },
        "run2": {
            "name": run2_name,
            "label": label2,
            "avg_winrate": avg_run2_winrate,
            "final_winrate": final_result.get('agent2_winrate', 0)
        },
        "games_per_pairing": games_per_pairing,
        "comparison_results": comparison_results
    }
    
    # Print summary
    print("\n" + "="*70)
    print("ELO COMPARISON (HEAD-TO-HEAD)")
    print("="*70)
    print(f"\n{label1} ({run1_name}):")
    print(f"  Average win rate: {avg_run1_winrate:.3f}")
    print(f"  Final win rate:   {final_result.get('agent1_winrate', 0):.3f}")
    
    print(f"\n{label2} ({run2_name}):")
    print(f"  Average win rate: {avg_run2_winrate:.3f}")
    print(f"  Final win rate:   {final_result.get('agent2_winrate', 0):.3f}")
    
    print(f"\nNumber of comparisons: {len(comparison_results)}")
    print(f"Games per comparison: {games_per_pairing}")
    
    if avg_run1_winrate > avg_run2_winrate:
        advantage = (avg_run1_winrate - avg_run2_winrate) * 100
        print(f"\n{label1} performs {advantage:.1f}% better on average")
    elif avg_run2_winrate > avg_run1_winrate:
        advantage = (avg_run2_winrate - avg_run1_winrate) * 100
        print(f"\n{label2} performs {advantage:.1f}% better on average")
    else:
        print(f"\nBoth runs perform equally on average")
    
    print("="*70 + "\n")
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Saved comparison results to {output_path}")
    
    return comparison


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare ELO ratings between two Nash-PG runs via head-to-head play",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Compare adaptive vs fixed KL
  python scripts/compare_two_runs_elo.py \\
      --env kuhn_poker \\
      --run1 "kuhn_poker/nash_pg_adaptive" \\
      --run2 "kuhn_poker/nash_pg_fixed" \\
      --label1 "Adaptive KL" \\
      --label2 "Fixed KL"
  
  # Use more games and save results
  python scripts/compare_two_runs_elo.py \\
      --env kuhn_poker \\
      --run1 "kuhn_poker/nash_pg_adaptive" \\
      --run2 "kuhn_poker/nash_pg_fixed" \\
      --games-per-pairing 100 \\
      --output results/elo_comparison.json
        """
    )
    
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Environment name (e.g., kuhn_poker, leduc_poker, tictactoe)"
    )
    
    parser.add_argument(
        "--run1",
        type=str,
        required=True,
        help="First run name (e.g., 'kuhn_poker/nash_pg_adaptive')"
    )
    
    parser.add_argument(
        "--run2",
        type=str,
        required=True,
        help="Second run name (e.g., 'kuhn_poker/nash_pg_fixed')"
    )
    
    parser.add_argument(
        "--label1",
        type=str,
        default="Run 1",
        help="Label for first run (default: 'Run 1')"
    )
    
    parser.add_argument(
        "--label2",
        type=str,
        default="Run 2",
        help="Label for second run (default: 'Run 2')"
    )
    
    parser.add_argument(
        "--games-per-pairing",
        type=int,
        default=50,
        help="Number of games to play per comparison (default: 50)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file to save comparison results"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    logger.info(f"Environment: {args.env}")
    logger.info(f"Run 1: {args.run1} ({args.label1})")
    logger.info(f"Run 2: {args.run2} ({args.label2})")
    logger.info(f"Games per pairing: {args.games_per_pairing}")
    
    # Compare ELO
    comparison = compare_elo_over_training(
        args.env,
        args.run1,
        args.run2,
        args.label1,
        args.label2,
        args.games_per_pairing,
        args.output
    )
    
    if not comparison:
        logger.error("Failed to complete ELO comparison!")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
