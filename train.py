"""
Train RL agents on BipedalWalker-v3
"""
import gymnasium as gym
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import argparse
from datetime import datetime


def train(algorithm='SAC', timesteps=500000, save_freq=50000):
    print(f"\n{'='*60}\nTraining {algorithm} - {timesteps:,} steps\n{'='*60}\n")

    env = Monitor(gym.make('BipedalWalker-v3'))
    eval_env = Monitor(gym.make('BipedalWalker-v3'))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/{algorithm.lower()}_{timestamp}"
    log_dir = f"tensorboard_logs/{algorithm.lower()}_{timestamp}"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    eval_cb = EvalCallback(eval_env, best_model_save_path=f"{model_dir}/best",
                           log_path=f"logs/{algorithm.lower()}", eval_freq=10000,
                           deterministic=True, render=False, n_eval_episodes=5)

    checkpoint_cb = CheckpointCallback(save_freq=save_freq, save_path=f"{model_dir}/checkpoints",
                                       name_prefix=f"{algorithm.lower()}_model")

    if algorithm == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1, learning_rate=3e-4, buffer_size=300000,
                   batch_size=256, gamma=0.99, tau=0.02, tensorboard_log=log_dir, device='cuda')
    elif algorithm == 'TD3':
        model = TD3('MlpPolicy', env, verbose=1, learning_rate=1e-3, buffer_size=200000,
                   batch_size=100, gamma=0.99, tau=0.005, tensorboard_log=log_dir, device='cuda')
    elif algorithm == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1, learning_rate=3e-4, n_steps=2048,
                   batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                   clip_range=0.2, tensorboard_log=log_dir, device='cuda')

    print("Starting training...")
    model.learn(total_timesteps=timesteps, callback=[eval_cb, checkpoint_cb], progress_bar=True)

    final_path = f"{model_dir}/final_model"
    model.save(final_path)
    print(f"\nModel saved: {final_path}\n")

    env.close()
    eval_env.close()

    return model, model_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train BipedalWalker')
    parser.add_argument('--algorithm', type=str, default='SAC', choices=['SAC', 'TD3', 'PPO'])
    parser.add_argument('--timesteps', type=int, default=500000)
    parser.add_argument('--save-freq', type=int, default=50000)

    args = parser.parse_args()
    train(args.algorithm, args.timesteps, args.save_freq)
