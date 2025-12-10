# Reinforcement Learning Results Summary

## Main Takeaway (Best Models)

### PPO — Best Overall
- Mean reward: ~295.87  
- Standard deviation: ±1.81  
- Success rate: 100%  
- Very stable and reliable.  
**Use PPO’s best model.**

### TD3 — Second Place
- Mean reward: ~257.49  
- Standard deviation: ±60.69  
- One very bad episode caused instability.  
**Good performance but less stable than PPO.**

### SAC — Poor Performance
- Mean reward: ~–9.73  
- Success rate: 0%  
- Failed to learn the task.  
**Not useful without major tuning.**

## Final Checkpoints

### TD3 Final Checkpoint
- Score: ~277.23  
- Standard deviation: ±2.74  
- More stable than its own best model.

### PPO Final Checkpoint
- Score: ~243.60  
- Standard deviation: ±103.72  
- Unstable and worse than the best model.  
**Always use PPO’s best model, not the final one.**
