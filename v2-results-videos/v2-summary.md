# V2 Training Results - Google Colab

Training completed on Google Colab using T4 GPU with 500,000 timesteps per algorithm on BipedalWalker-v3 environment.

# Training Setup

Trained three reinforcement learning algorithms: SAC, TD3, and PPO. Each algorithm was trained for 500,000 steps using GPU acceleration. The environment used was BipedalWalker-v3 from Gymnasium.

# SAC Results

SAC achieved the best performance with a mean reward of 284.82 across 10 evaluation episodes. The algorithm showed very stable performance after the first episode, with 9 out of 10 episodes scoring above 308 points. The first episode scored only 65.93, but all subsequent episodes were highly consistent around 309 points. Success rate was 90 percent with only one episode below 200 points.

# TD3 Results

TD3 achieved the second best performance with a mean reward of 267.04. Most episodes were very consistent, scoring between 281 and 289 points. However, the last evaluation episode had a significant drop to 107.59 which affected the overall average. Without this outlier, TD3 would have achieved similar performance to SAC. Success rate was 90 percent.

# PPO Results

PPO failed to learn effective walking behavior within 500,000 steps. The algorithm achieved a mean reward of negative 109.58, with all 10 evaluation episodes resulting in negative rewards. The best episode only reached negative 57.79. The walker was unable to maintain balance and likely fell immediately in most episodes. Success rate was 0 percent.

# Key Findings

SAC demonstrated the most reliable learning and consistent performance on this task. TD3 showed competitive results but with occasional instability. PPO requires significantly more training steps or hyperparameter tuning to solve BipedalWalker. The 500,000 timestep training duration was sufficient for SAC and TD3 but insufficient for PPO.

# Videos

Three demonstration videos are included showing the best learned policies: v2-colab-sac500.mp4, v2-colab-td3500.mp4, and v2-colab-ppo500.mp4.

# Visualizations

Multiple analysis charts are included: comprehensive analysis panel, episode-by-episode performance graphs, reward distribution plots, success rate comparison, and statistical summary tables.
