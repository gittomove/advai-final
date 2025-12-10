"""
Record videos of best models in action
"""
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, TD3, PPO
import imageio


def record_video(model_path, algorithm, output_path, n_episodes=1):
    """Record video of model performance"""
    print(f"Recording {algorithm}...")

    env = gym.make('BipedalWalker-v3', render_mode='rgb_array')

    # Load model
    if algorithm == 'SAC':
        model = SAC.load(model_path)
    elif algorithm == 'TD3':
        model = TD3.load(model_path)
    elif algorithm == 'PPO':
        model = PPO.load(model_path)

    frames = []
    total_reward = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False
        ep_reward = 0

        while not (done or truncated):
            # Add text overlay to frame
            frame = env.render()
            frames.append(frame)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward

        total_reward += ep_reward
        print(f"  Episode {ep + 1}: {ep_reward:.2f}")

    env.close()

    # Save video
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=30)
    print(f"  Saved: {output_path}")
    print(f"  Mean reward: {total_reward / n_episodes:.2f}\n")

    return total_reward / n_episodes


def record_all_best_models():
    """Record videos for all best models"""
    models = {
        'PPO': 'models/ppo_20251209_223634/best/best_model',
        'SAC': 'models/sac_20251209_223638/best/best_model',
        'TD3': 'models/td3_20251209_223718/best/best_model'
    }

    print("\n" + "="*70)
    print("RECORDING VIDEOS OF BEST MODELS")
    print("="*70 + "\n")

    os.makedirs('videos', exist_ok=True)

    results = {}

    for alg, model_path in models.items():
        try:
            output_path = f'videos/{alg.lower()}_best_model.mp4'
            reward = record_video(model_path, alg, output_path, n_episodes=1)
            results[alg] = {
                'status': 'success',
                'reward': reward,
                'video': output_path
            }
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")
            results[alg] = {'status': 'failed', 'error': str(e)}

    print("="*70)
    print("SUMMARY")
    print("="*70 + "\n")

    for alg, res in results.items():
        if res['status'] == 'success':
            print(f"[OK] {alg}: {res['reward']:.2f} - {res['video']}")
        else:
            print(f"[FAIL] {alg}: Failed - {res.get('error', 'Unknown error')}")

    print()
    return results


if __name__ == "__main__":
    record_all_best_models()
