"""
Train all algorithms sequentially
"""
from train import train
import json
from datetime import datetime


def train_all(timesteps=500000):
    algorithms = ['SAC', 'TD3', 'PPO']
    results = {}

    print("\n" + "="*60)
    print(f"Training all algorithms: {', '.join(algorithms)}")
    print(f"Timesteps per algorithm: {timesteps:,}")
    print("="*60 + "\n")

    start = datetime.now()

    for alg in algorithms:
        print(f"\n{'#'*60}\n# {alg}\n{'#'*60}\n")

        try:
            model, model_dir = train(alg, timesteps)
            results[alg] = {
                'status': 'success',
                'model_dir': model_dir,
                'final_model': f"{model_dir}/final_model"
            }
            print(f"\n✓ {alg} completed")

        except Exception as e:
            print(f"\n✗ {alg} failed: {e}")
            results[alg] = {'status': 'failed', 'error': str(e)}

    end = datetime.now()
    duration = end - start

    with open(f"results/training_summary_{start.strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump({
            'start': start.isoformat(),
            'end': end.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'timesteps': timesteps,
            'results': results
        }, f, indent=2)

    print("\n" + "="*60)
    print("Training completed")
    print(f"Total time: {duration}")
    print("="*60 + "\n")

    for alg, res in results.items():
        status = "✓" if res['status'] == 'success' else "✗"
        print(f"{status} {alg}: {res['status']}")
        if res['status'] == 'success':
            print(f"   Model: {res['final_model']}")

    return results


if __name__ == "__main__":
    train_all()
