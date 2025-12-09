"""
Скрипт для тренировки всех алгоритмов (SAC, TD3, PPO) последовательно
"""
import argparse
from train_bipedal import train_agent
import json
import os
from datetime import datetime


def train_all_algorithms(timesteps=500000, save_freq=50000):
    """
    Тренировка всех трех алгоритмов

    Args:
        timesteps: Количество шагов обучения для каждого алгоритма
        save_freq: Частота сохранения чекпоинтов
    """
    algorithms = ['SAC', 'TD3', 'PPO']
    results = {}

    print("\n" + "="*60)
    print("НАЧАЛО ТРЕНИРОВКИ ВСЕХ АЛГОРИТМОВ")
    print(f"Алгоритмы: {', '.join(algorithms)}")
    print(f"Шагов обучения на алгоритм: {timesteps}")
    print("="*60 + "\n")

    start_time = datetime.now()

    for i, algorithm in enumerate(algorithms, 1):
        print(f"\n{'#'*60}")
        print(f"# АЛГОРИТМ {i}/{len(algorithms)}: {algorithm}")
        print(f"{'#'*60}\n")

        try:
            model, model_dir = train_agent(
                algorithm=algorithm,
                timesteps=timesteps,
                save_freq=save_freq
            )

            results[algorithm] = {
                'status': 'success',
                'model_dir': model_dir,
                'final_model': f"{model_dir}/final_model"
            }

            print(f"\n✓ {algorithm} успешно обучен!")

        except Exception as e:
            print(f"\n✗ Ошибка при обучении {algorithm}: {str(e)}")
            results[algorithm] = {
                'status': 'failed',
                'error': str(e)
            }

    end_time = datetime.now()
    total_time = end_time - start_time

    # Сохранение результатов
    results_file = f"results/training_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("results", exist_ok=True)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_time_seconds': total_time.total_seconds(),
            'timesteps_per_algorithm': timesteps,
            'results': results
        }, f, indent=4, ensure_ascii=False)

    # Вывод итоговой информации
    print("\n" + "="*60)
    print("ТРЕНИРОВКА ЗАВЕРШЕНА")
    print("="*60)
    print(f"\nОбщее время: {total_time}")
    print(f"\nРезультаты:")

    for alg, result in results.items():
        status = "✓ Успех" if result['status'] == 'success' else "✗ Ошибка"
        print(f"  {alg}: {status}")
        if result['status'] == 'success':
            print(f"    Модель: {result['final_model']}")

    print(f"\nИтоговые результаты сохранены: {results_file}")
    print("="*60 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description='Тренировка всех алгоритмов RL')
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Количество шагов обучения для каждого алгоритма')
    parser.add_argument('--save-freq', type=int, default=50000,
                       help='Частота сохранения чекпоинтов')

    args = parser.parse_args()

    train_all_algorithms(
        timesteps=args.timesteps,
        save_freq=args.save_freq
    )


if __name__ == "__main__":
    main()
