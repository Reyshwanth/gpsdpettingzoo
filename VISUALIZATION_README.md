# GPSD Policy Visualization Tools

This directory contains tools to visualize and compare trained GPSD policies.

## Available Scripts

### 1. `test_gpsd.py` - Visualize a Single Policy

Test and visualize a single trained policy or random baseline.

**Usage:**

```bash
# List all available trained policies
python test_gpsd.py --policy list

# Test with random policy (baseline)
python test_gpsd.py

# Test a specific trained policy with visualization
python test_gpsd.py --policy runs/gpsd__train_gpsd_ppo__1__1770837747/gpsd_ppo_agent.pt

# You can also just provide the directory (auto-completes to gpsd_ppo_agent.pt)
python test_gpsd.py --policy runs/gpsd__train_gpsd_ppo__1__1770837747/

# Run multiple episodes
python test_gpsd.py --policy runs/gpsd__train_gpsd_ppo__1__1770837747/gpsd_ppo_agent.pt --num-episodes 3

# Run without rendering (faster)
python test_gpsd.py --policy runs/gpsd__train_gpsd_ppo__1__1770837747/gpsd_ppo_agent.pt --no-render
```

**Arguments:**
- `--policy PATH`: Path to policy .pt file, 'random', or 'list'
- `--num-episodes N`: Number of episodes to run (default: 1)
- `--max-cycles N`: Maximum cycles per episode (default: 100)
- `--no-render`: Disable rendering window

**Output:**
- Visual rendering of the environment (if rendering enabled)
- Episode statistics including rewards and coverage
- Agent positions, belief errors, and covariance traces

---

### 2. `compare_policies.py` - Compare Multiple Policies

Quantitatively compare multiple trained policies and generate visualizations.

**Usage:**

```bash
# Compare all available policies
python compare_policies.py --policies all --num-episodes 10

# Compare specific policies
python compare_policies.py --policies \
    runs/gpsd__train_gpsd_ppo__1__1770837747/gpsd_ppo_agent.pt \
    runs/gpsd__train_gpsd_ppo__1__1770836752/gpsd_ppo_agent.pt \
    --num-episodes 10

# Include random baseline for comparison
python compare_policies.py --policies all --include-random --num-episodes 10

# Save comparison plot
python compare_policies.py --policies all --num-episodes 10 --save-plot comparison.png

# Set random seed for reproducibility
python compare_policies.py --policies all --num-episodes 20 --seed 42
```

**Arguments:**
- `--policies [PATHS...]`: Paths to policy files or 'all' for all policies
- `--num-episodes N`: Number of episodes per policy (default: 10)
- `--max-cycles N`: Maximum cycles per episode (default: 100)
- `--seed N`: Random seed for reproducibility (default: 42)
- `--include-random`: Include random baseline policy
- `--save-plot FILE`: Save comparison plot to file

**Output:**
- Statistical summary of each policy's performance
- Comparison table sorted by average reward
- Box plots showing:
  - Episode rewards distribution
  - Coverage performance
  - Episode lengths
  - Average rewards with error bars

---

## Available Trained Policies

To see all available trained policies:

```bash
python test_gpsd.py --policy list
```

Current saved policies in `runs/`:
```
runs/gpsd__train_gpsd_ppo__1__1770830610/gpsd_ppo_agent.pt
runs/gpsd__train_gpsd_ppo__1__1770830678/gpsd_ppo_agent.pt
runs/gpsd__train_gpsd_ppo__1__1770830792/gpsd_ppo_agent.pt
runs/gpsd__train_gpsd_ppo__1__1770830985/gpsd_ppo_agent.pt
runs/gpsd__train_gpsd_ppo__1__1770836752/gpsd_ppo_agent.pt
runs/gpsd__train_gpsd_ppo__1__1770837700/gpsd_ppo_agent.pt
runs/gpsd__train_gpsd_ppo__1__1770837747/gpsd_ppo_agent.pt
```

---

## Example Workflows

### Quick Test of Latest Policy

```bash
# Find the most recent policy
LATEST=$(ls -t runs/*/gpsd_ppo_agent.pt | head -1)
echo "Testing: $LATEST"

# Visualize it
python test_gpsd.py --policy "$LATEST"
```

### Compare Top 3 Policies

```bash
# Get the 3 most recent policies
POLICIES=$(ls -t runs/*/gpsd_ppo_agent.pt | head -3 | tr '\n' ' ')

# Compare them
python compare_policies.py --policies $POLICIES --num-episodes 15 --include-random
```

### Benchmark All Policies

```bash
# Comprehensive evaluation
python compare_policies.py --policies all \
    --num-episodes 20 \
    --include-random \
    --seed 42 \
    --save-plot policy_comparison.png
```

---

## Understanding the Metrics

**Rewards:**
- Higher is better
- Rewards are influenced by:
  - POI coverage (+10/(1+covariance))
  - Path penalty (-0.1 per step)
  - Agent communication bonus (+0.5 per agent in range)
  - High covariance penalty (if near POI but covariance too high)

**Coverage:**
- Percentage of POIs successfully covered
- A POI is covered when an agent is within range AND has low covariance
- 100% coverage = all 16 POIs covered

**Steps:**
- Number of individual agent actions taken
- Fewer steps = more efficient policy (if coverage is similar)

**Belief Error:**
- Euclidean distance between true position and believed position
- Lower is better
- Increases when agents are in GPS-denied zone

**Covariance Trace:**
- Sum of diagonal elements of position covariance matrix
- Measure of position uncertainty
- Must be < 0.015 to cover a POI
- Increases in GPS-denied zone, resets outside

---

## Troubleshooting

**"No trained policies found":**
- Make sure you've trained at least one policy with `train_gpsd_ppo.py`
- Check that `--save-model` was enabled during training

**"Policy file not found":**
- Verify the path is correct
- Use `--policy list` to see available policies

**Import errors:**
- Ensure PettingZoo is installed in editable mode: `pip install -e PettingZoo/`
- Make sure PyTorch is installed: `pip install torch`
- For comparison plots: `pip install matplotlib`

**Model loading errors:**
- The scripts automatically handle both checkpoint formats (with optimizer state) and raw state dict formats
- If you see state_dict errors, ensure the model was saved correctly during training

**Rendering issues:**
- If window doesn't appear, check your display settings
- Use `--no-render` flag to run without visualization
- On headless systems, use `compare_policies.py` instead
