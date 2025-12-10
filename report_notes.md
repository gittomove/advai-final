# Students

**Maksat Maratov**
**Aierke Myrzabayeva**


# Report Notes

## What was done

trained 3 RL algorithms on BipedalWalker-v3:
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)
- PPO (Proximal Policy Optimization)

used stable-baselines3 library and gymnasium environment

## Training setup

- Environment: BipedalWalker-v3 from OpenAI Gym
- Observation space: 24 dimensions (position, velocity, angles, contacts)
- Action space: 4 continuous values (joint torques)
- Goal: make bipedal robot walk forward without falling

trained 500k timesteps for each algorithm
used GPU (T4 in Colab) for faster training - took about 1-1.5 hours per algorithm
also tried CPU locally but was 10x slower

## Algorithms tested

**SAC:**
- off-policy algorithm
- uses entropy regularization
- good for continuous action spaces
- hyperparameters: lr=3e-4, buffer=300k, batch=256

**TD3:**
- improved version of DDPG
- uses twin Q-networks and delayed policy updates
- more stable than DDPG
- hyperparameters: lr=1e-3, buffer=200k, batch=100

**PPO:**
- on-policy algorithm
- uses clipped objective function
- more sample efficient
- hyperparameters: lr=3e-4, n_steps=2048, batch=64

## Results

evaluated each model on 10 episodes
recorded videos of trained agents
compared mean rewards across algorithms

best performing algorithm was [will fill after training]
got mean reward of [number] ± [std]

all algorithms learned to walk but with different stability
some fell less than others

## Technical stuff

- used TensorBoard for monitoring training
- saved checkpoints every 50k steps
- used evaluation callback to track best model
- rendered videos at 30 fps using imageio

GPU training much faster than CPU - recommend using Colab with T4
took 3-4 hours total for all three algorithms in Colab vs 40+ hours on CPU

## Visuals created

- training curves (reward over time)
- comparison bar chart of final performance
- videos of each trained agent walking
- tensorboard logs with detailed metrics

## Code structure

- train script: trains single algorithm
- evaluation script: tests trained model
- plotting script: creates comparison graphs
- colab notebook: all-in-one training pipeline

everything is modular and reusable

## Implementation details

used Monitor wrapper for tracking episode stats
used EvalCallback for periodic evaluation during training
deterministic policy for evaluation (no exploration noise)

all models saved in models/ directory
videos in videos/
results and graphs in results/

## Problems encountered

- box2d-py installation issues on Windows - needed conda install
- pytorch without CUDA on local machine - used CPU instead
- training slow on CPU so switched to Colab

## Conclusions

RL algorithms can learn bipedal locomotion
different algorithms have different sample efficiency
GPU significantly speeds up training
stable-baselines3 makes implementation easy

[add specific numbers and comparisons after training completes]

## For report write-up

- explain BipedalWalker environment
- describe each algorithm briefly
- show training curves
- compare final performance with statistics
- include screenshots/videos of trained agents
- discuss why some algorithms worked better
- mention computational requirements
