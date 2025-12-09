"""
Evaluate trained models
"""
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, TD3, PPO
import imageio
import argparse


def evaluate(model_path, algorithm='SAC', n_episodes=10, save_video=False):
    print(f"\nEvaluating {algorithm}...")

    env = gym.make('BipedalWalker-v3', render_mode='rgb_array' if save_video else None)

    if algorithm == 'SAC':
        model = SAC.load(model_path)
    elif algorithm == 'TD3':
        model = TD3.load(model_path)
    elif algorithm == 'PPO':
        model = PPO.load(model_path)

    rewards = []
    frames = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False
        ep_reward = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward

            if save_video and ep == 0:
                frames.append(env.render())

        rewards.append(ep_reward)
        print(f"  Episode {ep + 1}: {ep_reward:.2f}")

    env.close()

    if save_video and frames:
        video_path = f"videos/{algorithm.lower()}_eval.mp4"
        imageio.mimsave(video_path, frames, fps=30)
        print(f"\nVideo saved: {video_path}")

    mean = np.mean(rewards)
    std = np.std(rewards)

    print(f"\nResults:")
    print(f"  Mean reward: {mean:.2f} ± {std:.2f}")
    print(f"  Range: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]\n")

    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--algorithm', type=str, default='SAC', choices=['SAC', 'TD3', 'PPO'])
    parser.add_argument('--n-episodes', type=int, default=10)
    parser.add_argument('--save-video', action='store_true')

    args = parser.parse_args()
    evaluate(args.model_path, args.algorithm, args.n_episodes, args.save_video)
