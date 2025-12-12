# Adaptive KL Penalty Implementation

## Overview
This implementation adds **Adaptive KL Penalties** to the Nash-PG algorithm, following the approach from standard PPO (Schulman et al. 2017). Instead of using a fixed `mag_coef` (α) value, the coefficient is automatically adjusted based on how far the policy drifts from the magnetic agent.

## Motivation
The original paper claims that decaying α causes instability, so they fix it to a large constant (e.g., 0.2). However, this approach has limitations:
- A fixed value may be too strong or too weak depending on the training phase
- Different games/environments may require different penalty strengths
- The optimal value is not known a priori

**Adaptive KL Penalties solve this by:**
- Automatically tuning α up or down based on observed KL divergence
- If the agent drifts too far (KL > target), increase the penalty → Increase α
- If the agent barely changes (KL < target), decrease the penalty → Decrease α

## Implementation Details

### 1. Configuration Parameters (`conf/algorithm/nash_pg.yaml`)
```yaml
# Adaptive KL Penalty settings
adaptive_kl_penalty: true        # Enable/disable adaptive mechanism
target_kl: 0.01                 # Target KL divergence threshold
kl_adjustment_factor: 1.5       # Multiplicative adjustment factor
min_mag_coef: 0.001             # Minimum value for mag_coef
max_mag_coef: 1.0               # Maximum value for mag_coef
```

**Parameters:**
- `adaptive_kl_penalty`: Toggle to enable/disable the mechanism
- `target_kl`: The desired KL divergence between current and magnetic policy
- `kl_adjustment_factor`: How aggressively to adjust (1.5 = 50% increase/decrease)
- `min_mag_coef` / `max_mag_coef`: Safety bounds to prevent extreme values

### 2. LearnerState Extension (`train/nash_pg.py`)
Added `mag_coef` to the `LearnerState` dataclass to track the adaptive coefficient:
```python
@chex.dataclass
class LearnerState:
    ...
    mag_coef: float  # adaptive KL penalty coefficient
```

### 3. Adaptive Update Function
```python
def update_adaptive_mag_coef(learner_state: LearnerState, config: DictConfig) -> LearnerState:
    """
    Update mag_coef based on observed KL divergence (Adaptive KL Penalty from PPO).
    
    If KL > target_kl: penalty too weak → increase mag_coef
    If KL < target_kl: penalty too strong → decrease mag_coef
    """
```

**Algorithm:**
1. Extract current KL divergence from training metrics (`mag_kl`)
2. Compare with `target_kl`
3. If KL is too high: `mag_coef *= adjustment_factor`
4. If KL is too low: `mag_coef /= adjustment_factor`
5. Clamp to `[min_mag_coef, max_mag_coef]` range

### 4. Training Loop Integration
The adaptive update is called after each logging interval:
```python
# logging
log_metrics(learner_state, logger, cur_num_update)

# update adaptive mag_coef based on KL divergence
learner_state = update_adaptive_mag_coef(learner_state, config)
```

### 5. Enhanced Logging
The current `mag_coef` value is now logged alongside other training metrics, allowing you to:
- Monitor how the coefficient evolves during training
- Verify the adaptive mechanism is working correctly
- Analyze the relationship between `mag_coef` and training stability

## Usage

### Enable Adaptive KL Penalty
Set `adaptive_kl_penalty: true` in your config (default in `nash_pg.yaml`).

### Disable Adaptive KL Penalty
Set `adaptive_kl_penalty: false` to use the fixed `mag_coef` value.

### Tuning Parameters
- **target_kl**: Lower values (e.g., 0.005) enforce stricter similarity to magnetic agent. Higher values (e.g., 0.02) allow more drift.
- **kl_adjustment_factor**: Higher values (e.g., 2.0) cause more aggressive adjustments. Lower values (e.g., 1.2) cause gentler adjustments.
- **min_mag_coef / max_mag_coef**: Adjust based on your observation of what works for your environment.

## Expected Behavior

### During Training
- **Early training**: `mag_coef` may decrease if policies are similar (low KL)
- **Mid training**: `mag_coef` should stabilize around the value that maintains `KL ≈ target_kl`
- **Late training**: `mag_coef` may increase if policies diverge too much

### Advantages over Fixed α
1. **Automatic tuning**: No need to manually search for optimal α
2. **Adaptive to training phase**: Adjusts as the agent learns
3. **Stable convergence**: Prevents excessive drift while allowing learning
4. **Environment-agnostic**: Works across different games without manual tuning

## Monitoring

Watch the following metrics in your logs:
- `mag_kl`: The actual KL divergence between current and magnetic policy
- `mag_coef`: The adaptive penalty coefficient (should vary over time)
- `approx_kl`: The KL between old and new policy (for PPO stability)

**Healthy behavior:**
- `mag_kl` oscillates around `target_kl`
- `mag_coef` adjusts gradually, not wildly
- Training remains stable without large policy jumps

## Comparison to Original Paper

| Original Paper | This Implementation |
|---------------|---------------------|
| Fixed α = 0.2 | Adaptive α starting from 0.2 |
| Claims decaying α causes instability | Uses adaptive adjustment, not decay |
| Manual tuning required | Automatic tuning based on KL |
| Same α for all environments | Adapts per environment/training phase |

## References
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms" - Section on Adaptive KL Penalty
- Original Nash-PG paper approach with fixed magnetic coefficient
