"""
Evaluate all trained models and generate comparison report
"""
import os
import json
import numpy as np
from datetime import datetime
from eval import evaluate
from plot import plot_comparison


def find_latest_models(models_dir='models'):
    """Find the latest model for each algorithm"""
    models = {}

    for dirname in os.listdir(models_dir):
        dirpath = os.path.join(models_dir, dirname)
        if not os.path.isdir(dirpath):
            continue

        # Extract algorithm name from directory
        alg = dirname.split('_')[0].upper()

        # Check if this is a newer version
        if alg not in models or dirname > models[alg]['dirname']:
            final_model = os.path.join(dirpath, 'final_model')
            best_model = os.path.join(dirpath, 'best', 'best_model')

            models[alg] = {
                'dirname': dirname,
                'final_model': final_model,
                'best_model': best_model,
                'exists_final': os.path.exists(final_model + '.zip'),
                'exists_best': os.path.exists(best_model + '.zip')
            }

    return models


def evaluate_all(n_episodes=10, save_videos=False, use_best=False):
    """Evaluate all models"""
    print("\n" + "="*70)
    print("EVALUATING ALL MODELS")
    print("="*70 + "\n")

    models = find_latest_models()

    print(f"Found {len(models)} algorithms:")
    for alg, info in models.items():
        print(f"  {alg}: {info['dirname']}")
    print()

    results = {}
    all_rewards = {}

    start_time = datetime.now()

    for alg in sorted(models.keys()):
        info = models[alg]

        # Choose best or final model
        if use_best and info['exists_best']:
            model_path = info['best_model']
            model_type = 'best'
        else:
            model_path = info['final_model']
            model_type = 'final'

        print(f"\n{'='*70}")
        print(f"Evaluating {alg} ({model_type} model)")
        print(f"Path: {model_path}")
        print(f"{'='*70}")

        try:
            rewards = evaluate(
                model_path=model_path,
                algorithm=alg,
                n_episodes=n_episodes,
                save_video=save_videos
            )

            results[alg] = {
                'status': 'success',
                'model_path': model_path,
                'model_type': model_type,
                'n_episodes': n_episodes,
                'rewards': rewards,
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards))
            }
            all_rewards[alg] = rewards

        except Exception as e:
            print(f"\n✗ Failed to evaluate {alg}: {e}\n")
            results[alg] = {
                'status': 'failed',
                'error': str(e)
            }

    end_time = datetime.now()
    duration = end_time - start_time

    # Save results
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    results_file = f'newresults/evaluation_{timestamp}.json'

    summary = {
        'timestamp': start_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'n_episodes': n_episodes,
        'use_best_models': use_best,
        'results': results
    }

    os.makedirs('newresults', exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70 + "\n")

    for alg in sorted(results.keys()):
        res = results[alg]
        if res['status'] == 'success':
            print(f"{alg}:")
            print(f"  Mean: {res['mean']:.2f} ± {res['std']:.2f}")
            print(f"  Range: [{res['min']:.2f}, {res['max']:.2f}]")
        else:
            print(f"{alg}: FAILED - {res.get('error', 'Unknown error')}")

    print(f"\nResults saved: {results_file}")
    print(f"Total time: {duration}\n")

    # Generate comparison plot if we have successful results
    if all_rewards:
        print("Generating comparison plot...")
        plot_path = f'newresults/comparison_{timestamp}.png'
        plot_comparison(all_rewards, save_path=plot_path)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate all trained models')
    parser.add_argument('--n-episodes', type=int, default=10,
                        help='Number of episodes to evaluate')
    parser.add_argument('--save-videos', action='store_true',
                        help='Save videos of first episode')
    parser.add_argument('--use-best', action='store_true',
                        help='Use best models instead of final models')

    args = parser.parse_args()

    evaluate_all(
        n_episodes=args.n_episodes,
        save_videos=args.save_videos,
        use_best=args.use_best
    )
