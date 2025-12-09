"""
Скрипт для построения графиков результатов обучения
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
import glob


def load_tensorboard_data(log_dir):
    """Загрузка данных из логов TensorBoard"""
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))

    if not event_files:
        print(f"Предупреждение: не найдены файлы событий в {log_dir}")
        return None

    event_acc = EventAccumulator(event_files[0])
    event_acc.Reload()

    # Получение доступных тегов
    tags = event_acc.Tags()
    data = {}

    # Извлечение данных для разных метрик
    if 'scalars' in tags and tags['scalars']:
        for tag in tags['scalars']:
            events = event_acc.Scalars(tag)
            steps = [event.step for event in events]
            values = [event.value for event in events]
            data[tag] = {'steps': steps, 'values': values}

    return data


def plot_training_curves(algorithms_data, save_path='results/training_curves.png'):
    """
    Построение кривых обучения для нескольких алгоритмов

    Args:
        algorithms_data: Словарь {algorithm_name: tensorboard_data}
        save_path: Путь для сохранения графика
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Сравнение алгоритмов RL для BipedalWalker-v3', fontsize=16, fontweight='bold')

    colors = {'SAC': 'blue', 'TD3': 'red', 'PPO': 'green'}

    # Метрики для отображения
    metrics = {
        'rollout/ep_rew_mean': ('Средняя награда за эпизод', axes[0, 0]),
        'rollout/ep_len_mean': ('Средняя длина эпизода', axes[0, 1]),
        'train/learning_rate': ('Learning Rate', axes[1, 0]),
        'time/fps': ('FPS (скорость обучения)', axes[1, 1])
    }

    # Построение графиков
    for metric_tag, (title, ax) in metrics.items():
        for alg_name, data in algorithms_data.items():
            if data is not None and metric_tag in data:
                steps = data[metric_tag]['steps']
                values = data[metric_tag]['values']

                # Сглаживание с помощью скользящего среднего
                if len(values) > 50:
                    window = min(50, len(values) // 10)
                    smoothed = pd.Series(values).rolling(window=window, min_periods=1).mean()
                    ax.plot(steps, smoothed, label=alg_name,
                           color=colors.get(alg_name, 'black'), linewidth=2, alpha=0.8)
                    ax.plot(steps, values, alpha=0.2, color=colors.get(alg_name, 'black'))
                else:
                    ax.plot(steps, values, label=alg_name,
                           color=colors.get(alg_name, 'black'), linewidth=2)

        ax.set_xlabel('Шаги обучения', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"График сохранен: {save_path}")
    plt.close()


def plot_evaluation_comparison(eval_results, save_path='results/evaluation_comparison.png'):
    """
    Построение графика сравнения оценок для разных алгоритмов

    Args:
        eval_results: Словарь {algorithm_name: evaluation_stats}
        save_path: Путь для сохранения графика
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    algorithms = list(eval_results.keys())
    mean_rewards = [eval_results[alg]['mean_reward'] for alg in algorithms]
    std_rewards = [eval_results[alg]['std_reward'] for alg in algorithms]

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(algorithms))
    colors_map = {'SAC': 'blue', 'TD3': 'red', 'PPO': 'green'}
    colors = [colors_map.get(alg, 'gray') for alg in algorithms]

    bars = ax.bar(x_pos, mean_rewards, yerr=std_rewards, align='center',
                   alpha=0.8, ecolor='black', capsize=10, color=colors)

    ax.set_xlabel('Алгоритм', fontsize=12, fontweight='bold')
    ax.set_ylabel('Средняя награда', fontsize=12, fontweight='bold')
    ax.set_title('Сравнение производительности алгоритмов', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Добавление значений над столбцами
    for i, (bar, mean, std) in enumerate(zip(bars, mean_rewards, std_rewards)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"График сохранен: {save_path}")
    plt.close()


def plot_episode_rewards_distribution(eval_results, save_path='results/rewards_distribution.png'):
    """
    Построение распределения наград по эпизодам

    Args:
        eval_results: Словарь {algorithm_name: evaluation_stats}
        save_path: Путь для сохранения графика
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, len(eval_results), figsize=(15, 5))
    if len(eval_results) == 1:
        axes = [axes]

    colors_map = {'SAC': 'blue', 'TD3': 'red', 'PPO': 'green'}

    for i, (alg_name, results) in enumerate(eval_results.items()):
        episode_rewards = results.get('episode_rewards', [])

        if episode_rewards:
            axes[i].hist(episode_rewards, bins=20, alpha=0.7,
                        color=colors_map.get(alg_name, 'gray'), edgecolor='black')
            axes[i].axvline(np.mean(episode_rewards), color='red',
                          linestyle='--', linewidth=2, label=f'Среднее: {np.mean(episode_rewards):.1f}')
            axes[i].set_xlabel('Награда', fontsize=10)
            axes[i].set_ylabel('Частота', fontsize=10)
            axes[i].set_title(f'{alg_name} - Распределение наград', fontsize=12, fontweight='bold')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"График сохранен: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Построение графиков результатов')
    parser.add_argument('--tensorboard-logs', type=str, default='tensorboard_logs',
                       help='Директория с логами TensorBoard')

    args = parser.parse_args()

    print("\nЗагрузка данных из TensorBoard...")

    # Поиск всех логов
    log_dirs = glob.glob(os.path.join(args.tensorboard_logs, '*'))

    algorithms_data = {}
    for log_dir in log_dirs:
        alg_name = os.path.basename(log_dir).split('_')[0].upper()
        print(f"Загрузка данных для {alg_name} из {log_dir}")

        # Поиск вложенной директории с событиями
        event_dirs = glob.glob(os.path.join(log_dir, '*', '*'))
        if event_dirs:
            data = load_tensorboard_data(event_dirs[0])
        else:
            data = load_tensorboard_data(log_dir)

        if data:
            algorithms_data[alg_name] = data

    if algorithms_data:
        print(f"\nЗагружены данные для алгоритмов: {list(algorithms_data.keys())}")
        print("\nПостроение графиков...")
        plot_training_curves(algorithms_data)
        print("\nГрафики успешно созданы!")
    else:
        print("\nНе найдено данных для построения графиков.")
        print("Убедитесь, что обучение завершено и логи TensorBoard доступны.")


if __name__ == "__main__":
    main()
