"""
Скрипт для тренировки агента BipedalWalker с разными RL алгоритмами
"""
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import argparse
from datetime import datetime


def make_env():
    """Создание окружения BipedalWalker"""
    env = gym.make('BipedalWalker-v3')
    env = Monitor(env)
    return env


def train_agent(algorithm='SAC', timesteps=500000, save_freq=50000):
    """
    Тренировка агента с указанным алгоритмом

    Args:
        algorithm: Название алгоритма ('SAC', 'TD3', 'PPO')
        timesteps: Количество шагов обучения
        save_freq: Частота сохранения чекпоинтов
    """
    print(f"\n{'='*60}")
    print(f"Начало тренировки с алгоритмом {algorithm}")
    print(f"Количество шагов: {timesteps}")
    print(f"{'='*60}\n")

    # Создание окружения
    env = make_env()
    eval_env = make_env()

    # Определение пути для сохранения
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/{algorithm.lower()}_{timestamp}"
    log_dir = f"tensorboard_logs/{algorithm.lower()}_{timestamp}"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Настройка коллбэков
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best",
        log_path=f"logs/{algorithm.lower()}",
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=f"{model_dir}/checkpoints",
        name_prefix=f"{algorithm.lower()}_model"
    )

    # Создание модели в зависимости от алгоритма
    if algorithm == 'SAC':
        model = SAC(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=300000,
            batch_size=256,
            gamma=0.99,
            tau=0.02,
            tensorboard_log=log_dir,
            device='cuda'
        )
    elif algorithm == 'TD3':
        model = TD3(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=1e-3,
            buffer_size=200000,
            batch_size=100,
            gamma=0.99,
            tau=0.005,
            tensorboard_log=log_dir,
            device='cuda'
        )
    elif algorithm == 'PPO':
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=log_dir,
            device='cuda'
        )
    else:
        raise ValueError(f"Неизвестный алгоритм: {algorithm}")

    # Обучение
    print(f"Начало обучения...")
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    # Сохранение финальной модели
    final_model_path = f"{model_dir}/final_model"
    model.save(final_model_path)
    print(f"\nФинальная модель сохранена: {final_model_path}")

    # Закрытие окружений
    env.close()
    eval_env.close()

    print(f"\n{'='*60}")
    print(f"Тренировка завершена!")
    print(f"Модели сохранены в: {model_dir}")
    print(f"Логи TensorBoard: {log_dir}")
    print(f"{'='*60}\n")

    return model, model_dir


def main():
    parser = argparse.ArgumentParser(description='Тренировка BipedalWalker')
    parser.add_argument('--algorithm', type=str, default='SAC',
                       choices=['SAC', 'TD3', 'PPO'],
                       help='Алгоритм RL для тренировки')
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Количество шагов обучения')
    parser.add_argument('--save-freq', type=int, default=50000,
                       help='Частота сохранения чекпоинтов')

    args = parser.parse_args()

    # Тренировка
    train_agent(
        algorithm=args.algorithm,
        timesteps=args.timesteps,
        save_freq=args.save_freq
    )


if __name__ == "__main__":
    main()
