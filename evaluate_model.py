"""
Скрипт для оценки и визуализации обученных моделей
"""
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, TD3, PPO
import imageio
import argparse
import os


def evaluate_model(model_path, algorithm='SAC', n_episodes=10, render=True, save_video=False):
    """
    Оценка производительности модели

    Args:
        model_path: Путь к сохраненной модели
        algorithm: Тип алгоритма ('SAC', 'TD3', 'PPO')
        n_episodes: Количество эпизодов для оценки
        render: Рендерить ли окружение
        save_video: Сохранить ли видео
    """
    print(f"\n{'='*60}")
    print(f"Оценка модели: {model_path}")
    print(f"Алгоритм: {algorithm}")
    print(f"Количество эпизодов: {n_episodes}")
    print(f"{'='*60}\n")

    # Создание окружения
    if render:
        env = gym.make('BipedalWalker-v3', render_mode='human')
    else:
        env = gym.make('BipedalWalker-v3', render_mode='rgb_array')

    # Загрузка модели
    if algorithm == 'SAC':
        model = SAC.load(model_path)
    elif algorithm == 'TD3':
        model = TD3.load(model_path)
    elif algorithm == 'PPO':
        model = PPO.load(model_path)
    else:
        raise ValueError(f"Неизвестный алгоритм: {algorithm}")

    # Статистика
    episode_rewards = []
    episode_lengths = []

    # Для сохранения видео
    if save_video:
        frames = []
        video_path = f"videos/{algorithm.lower()}_evaluation.mp4"
        os.makedirs("videos", exist_ok=True)

    # Запуск эпизодов
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0

        while not (done or truncated):
            # Получение действия от модели
            action, _states = model.predict(obs, deterministic=True)

            # Шаг в окружении
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            # Сохранение кадра для видео
            if save_video and episode == 0:  # Сохраняем только первый эпизод
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Эпизод {episode + 1}: Награда = {episode_reward:.2f}, Длина = {episode_length}")

    # Сохранение видео
    if save_video and frames:
        print(f"\nСохранение видео в {video_path}...")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Видео сохранено!")

    # Вывод статистики
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    print(f"\n{'='*60}")
    print(f"Результаты оценки:")
    print(f"Средняя награда: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Средняя длина эпизода: {mean_length:.2f}")
    print(f"Минимальная награда: {np.min(episode_rewards):.2f}")
    print(f"Максимальная награда: {np.max(episode_rewards):.2f}")
    print(f"{'='*60}\n")

    env.close()

    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'episode_rewards': episode_rewards
    }


def main():
    parser = argparse.ArgumentParser(description='Оценка модели BipedalWalker')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Путь к сохраненной модели')
    parser.add_argument('--algorithm', type=str, default='SAC',
                       choices=['SAC', 'TD3', 'PPO'],
                       help='Тип алгоритма')
    parser.add_argument('--n-episodes', type=int, default=10,
                       help='Количество эпизодов для оценки')
    parser.add_argument('--render', action='store_true',
                       help='Показывать визуализацию')
    parser.add_argument('--save-video', action='store_true',
                       help='Сохранить видео')

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        algorithm=args.algorithm,
        n_episodes=args.n_episodes,
        render=args.render,
        save_video=args.save_video
    )


if __name__ == "__main__":
    main()
