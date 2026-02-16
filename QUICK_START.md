# Quick Start Guide - GPSD Policy Visualization

## 🚀 Quick Commands

### List All Available Policies
```bash
./visualize.sh list
```

### Test Latest Trained Policy (with visual rendering)
```bash
./visualize.sh latest
```

### Test Random Baseline
```bash
./visualize.sh random
```

### Compare All Policies (Statistical Analysis)
```bash
./visualize.sh compare
```

### Compare 3 Most Recent Policies
```bash
./visualize.sh compare-recent 3
```

---

## 📊 What Each Tool Does

### 1. **Test Single Policy** (`test_gpsd.py`)
- **What it does:** Runs a specific policy and shows visualization
- **Use when:** You want to watch how a specific trained agent behaves
- **Output:** 
  - Live rendering of agents moving in environment
  - Real-time statistics (positions, covariance, coverage)
  - Episode summary

### 2. **Compare Policies** (`compare_policies.py`)
- **What it does:** Evaluates multiple policies and compares performance
- **Use when:** You want to find the best policy or analyze training progress
- **Output:**
  - Statistical comparison table
  - Box plots showing performance distributions
  - Quantitative metrics (rewards, coverage, efficiency)

### 3. **Helper Script** (`visualize.sh`)
- **What it does:** Easy-to-use wrapper for common tasks
- **Use when:** You want quick access without remembering exact commands

---

## 📁 Files Created

- `test_gpsd.py` - Single policy visualization tool
- `compare_policies.py` - Multi-policy comparison tool
- `visualize.sh` - Convenient helper script
- `VISUALIZATION_README.md` - Detailed documentation
- `QUICK_START.md` - This file

---

## 🎯 Common Workflows

### After Training a New Model
```bash
# See how it performs visually
./visualize.sh latest

# Compare it to previous models
./visualize.sh compare-recent 5
```

### Finding the Best Model
```bash
# Compare all trained models
./visualize.sh compare --save-plot comparison.png

# The comparison will rank them by average reward
```

### Testing a Specific Model
```bash
# If you know the path
./visualize.sh test runs/gpsd__train_gpsd_ppo__1__1770837747/gpsd_ppo_agent.pt

# Run it for 3 episodes
./visualize.sh test runs/gpsd__train_gpsd_ppo__1__1770837747/gpsd_ppo_agent.pt --num-episodes 3
```

### Baseline Comparison
```bash
# Include random baseline when comparing
./visualize.sh compare --include-random
```

---

## 💡 Tips

1. **Use `list` first** to see what policies are available
   ```bash
   ./visualize.sh list
   ```

2. **Test visually first** to understand behavior
   ```bash
   ./visualize.sh latest
   ```

3. **Then compare quantitatively** for rigorous evaluation
   ```bash
   ./visualize.sh compare
   ```

4. **Save comparison plots** for reports/papers
   ```bash
   ./visualize.sh compare --save-plot results.png
   ```

5. **Run without rendering** for faster evaluation
   ```bash
   python3 test_gpsd.py --policy runs/.../gpsd_ppo_agent.pt --no-render --num-episodes 10
   ```

---

## 🔍 Understanding the Output

### Key Metrics

**Average Reward**
- Higher = better overall performance
- Combines coverage success, efficiency, and collaboration

**Coverage %**
- Percentage of POIs successfully covered
- 100% = all 16 POIs covered
- Most important metric for task success

**Average Steps**
- Efficiency metric
- Fewer steps with high coverage = better

**Belief Error**
- How much the agent's position estimate differs from reality
- Increases in GPS-denied zone

**Covariance Trace**
- Position uncertainty metric
- Must be < 0.015 to cover a POI

### What to Look For

✅ **Good Policy:**
- High coverage % (>80%)
- Good rewards
- Low belief errors outside GPS zone
- Agents stay connected (communication links)

❌ **Poor Policy:**
- Low coverage % (<50%)
- Negative rewards
- Agents wander aimlessly
- High covariance when trying to cover POIs

---

## 🐛 Troubleshooting

**"No trained policies found"**
- Train a model first: `python3 train_gpsd_ppo.py`

**"Command not found: python3"**
- Use `python` instead of `python3` in scripts

**Rendering window doesn't appear**
- Use `--no-render` flag for headless operation
- Or use `compare_policies.py` which doesn't need rendering

**Import errors**
- Make sure dependencies are installed:
  ```bash
  pip install torch numpy matplotlib
  pip install -e PettingZoo/
  ```

---

## 📚 More Information

See `VISUALIZATION_README.md` for:
- Detailed command line options
- Advanced usage examples
- Metric explanations
- Full troubleshooting guide

---

**Happy Visualizing! 🎉**
