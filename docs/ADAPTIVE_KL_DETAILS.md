# Adaptive KL Penalty - Technical Details

## How It Works

### Algorithm Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Collect trajectories using current policy               │
│     - Agent interacts with environment                      │
│     - Stores observations, actions, rewards                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Update agent with PPO + Magnetic Regularization         │
│     Loss = PPO_loss + ent_coef*H + mag_coef*KL_divergence  │
│                                            ↑                │
│                                  (adaptive coefficient)     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Measure KL divergence to magnetic policy                │
│     mag_kl = KL(π_current || π_magnetic)                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Adaptive Update Decision                                │
│                                                             │
│     if mag_kl > target_kl:                                 │
│         mag_coef *= adjustment_factor  (increase penalty)   │
│     else:                                                   │
│         mag_coef /= adjustment_factor  (decrease penalty)   │
│                                                             │
│     mag_coef = clip(mag_coef, min_mag_coef, max_mag_coef)  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    (loop back to step 1)
```

## Intuition

### Why Adaptive?

The magnetic regularization term `mag_coef * KL(π || π_mag)` encourages the current policy to stay close to the magnetic policy (previous outer loop policy).

**Problem with Fixed Coefficient:**
- Too large → Agent can't explore effectively
- Too small → Agent drifts too far from magnetic policy
- Optimal value varies by environment and training phase

**Solution with Adaptive Coefficient:**
- Automatically finds the right balance
- Adapts as training progresses
- Environment-agnostic

### The Feedback Loop

```
High KL Divergence (policy drifting too far)
    ↓
Increase mag_coef (strengthen penalty)
    ↓
Policy updates stay closer to magnetic policy
    ↓
Lower KL Divergence
    ↓
Decrease mag_coef (weaken penalty)
    ↓
Policy can explore more
    ↓
(potentially) Higher KL Divergence
    ↓
... cycle continues until equilibrium around target_kl
```

## Mathematical Formulation

### Standard PPO Loss
```
L_PPO = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
```

### Nash-PG with Fixed Coefficient
```
L_total = L_PPO + α_ent * H(π) + α_mag * KL(π || π_mag)
                                  ↑
                            (fixed α_mag)
```

### Nash-PG with Adaptive Coefficient
```
L_total = L_PPO + α_ent * H(π) + α_mag(t) * KL(π || π_mag)
                                  ↑
                            (adaptive α_mag)

Where α_mag is updated after each training step:
    
    α_mag(t+1) = {
        α_mag(t) * β,     if KL(π || π_mag) > KL_target
        α_mag(t) / β,     if KL(π || π_mag) ≤ KL_target
    }
    
    α_mag(t+1) = clip(α_mag(t+1), α_min, α_max)
```

## Hyperparameter Guidance

### target_kl

**What it controls:** How similar the policy should stay to the magnetic policy

**Typical values:**
- `0.005`: Very conservative, minimal policy drift
- `0.01`: Balanced (default)
- `0.02`: More exploratory, allows more drift

**How to tune:**
- Start with default (0.01)
- If training is too conservative (slow progress): increase to 0.015-0.02
- If training is unstable (large policy jumps): decrease to 0.005-0.008

### kl_adjustment_factor

**What it controls:** How aggressively mag_coef is adjusted

**Typical values:**
- `1.2`: Gentle adjustments
- `1.5`: Balanced (default)
- `2.0`: Aggressive adjustments

**How to tune:**
- Start with default (1.5)
- If mag_coef changes too slowly: increase to 1.8-2.0
- If mag_coef oscillates wildly: decrease to 1.2-1.3

### min_mag_coef / max_mag_coef

**What they control:** Safety bounds for the adaptive coefficient

**Typical values:**
- `min_mag_coef: 0.001` (prevents penalty from vanishing)
- `max_mag_coef: 1.0` (prevents penalty from dominating)

**How to tune:**
- Usually don't need to change
- If mag_coef hits bounds frequently, expand the range
- If mag_coef never reaches bounds, tighten the range

## Monitoring and Debugging

### Key Metrics to Watch

1. **mag_kl**: Should oscillate around `target_kl`
   - Too high → mag_coef will increase
   - Too low → mag_coef will decrease

2. **mag_coef**: Should vary over time
   - Flat line = not adapting (check adaptive_kl_penalty=true)
   - Wild oscillations = adjustment_factor too high
   - Stuck at bounds = need wider [min, max] range

3. **approx_kl**: Standard PPO KL (old policy → new policy)
   - Independent of mag_kl
   - Should stay below ~0.03 for PPO stability

### Healthy Training Patterns

```
Early Training:
    mag_kl: 0.005-0.01 (low, policies similar)
    mag_coef: 0.1-0.2 (decreasing, less penalty needed)

Mid Training:
    mag_kl: 0.008-0.015 (moderate, policies diverging)
    mag_coef: 0.2-0.5 (increasing, more penalty needed)

Late Training:
    mag_kl: 0.01-0.02 (stable around target)
    mag_coef: 0.3-0.6 (stable, found equilibrium)
```

### Troubleshooting

**Problem:** mag_coef always at max_mag_coef
- **Cause:** Policy drifts too far from magnetic policy
- **Solution:** Increase max_mag_coef or increase target_kl

**Problem:** mag_coef always at min_mag_coef
- **Cause:** Policy too similar to magnetic policy
- **Solution:** Decrease min_mag_coef or decrease target_kl

**Problem:** mag_coef oscillates wildly
- **Cause:** Adjustment factor too aggressive
- **Solution:** Decrease kl_adjustment_factor

**Problem:** Training is unstable
- **Cause:** Multiple possible causes
- **Solutions:**
  - Decrease target_kl (more conservative)
  - Increase initial mag_coef
  - Check PPO hyperparameters (clip_eps, lr)

## Comparison: Fixed vs Adaptive

### Scenario 1: Simple Game (Tic-Tac-Toe)

**Fixed α = 0.2:**
- Works well if manually tuned
- May be too strong → slow convergence
- May be too weak → unstable training

**Adaptive:**
- Starts at 0.2
- Discovers optimal value automatically
- Adapts to training phases

### Scenario 2: Complex Game (Poker)

**Fixed α = 0.2:**
- May need different value (0.1? 0.5?)
- Requires expensive hyperparameter search
- Same value may not work throughout training

**Adaptive:**
- No hyperparameter search needed
- Automatically adjusts to game complexity
- Finds right balance at each training stage

## Implementation Notes

### Why Update After Logging?

```python
# logging
log_metrics(learner_state, logger, cur_num_update)

# update adaptive mag_coef based on KL divergence
learner_state = update_adaptive_mag_coef(learner_state, config)
```

We update after logging so that:
1. The logged mag_coef corresponds to the value used in training
2. We can compute metrics before adjusting the coefficient
3. The KL divergence used for adjustment is from the current training step

### Clamping Behavior

```python
new_mag_coef = jnp.clip(new_mag_coef, min_mag_coef, max_mag_coef)
```

Clamping ensures:
- Coefficient never becomes too small (would allow unbounded drift)
- Coefficient never becomes too large (would prevent learning)
- Numerical stability (avoids extreme values)

### JIT Compatibility

The adaptive update function is **not JIT-compiled** because:
- It's called infrequently (once per log interval)
- It involves Python conditionals on metric values
- The overhead is negligible compared to training step

The core training loop remains JIT-compiled for performance.

## Future Extensions

Possible improvements to explore:

1. **Exponential Moving Average of KL:**
   Instead of reacting to single KL measurement, use EMA for smoother adaptation

2. **Per-Player Adaptive Coefficients:**
   Different mag_coef for each player in multi-agent setting

3. **Automatic target_kl Selection:**
   Learn optimal target_kl from training dynamics

4. **Adaptive adjustment_factor:**
   Adjust the adjustment factor itself based on stability metrics
