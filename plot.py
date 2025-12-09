"""
Plot results and compare algorithms
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse


def plot_comparison(results_dict, save_path='results/comparison.png'):
    """
    results_dict: {'SAC': [rewards], 'TD3': [rewards], 'PPO': [rewards]}
    """
    algorithms = list(results_dict.keys())
    means = [np.mean(results_dict[alg]) for alg in algorithms]
    stds = [np.std(results_dict[alg]) for alg in algorithms]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(algorithms))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    bars = plt.bar(x, means, yerr=stds, align='center', alpha=0.8,
                   ecolor='black', capsize=10, color=colors)

    plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Reward', fontsize=12, fontweight='bold')
    plt.title('Algorithm Performance Comparison\nBipedalWalker-v3', fontsize=14, fontweight='bold')
    plt.xticks(x, algorithms, fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')

    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot comparison')
    parser.add_argument('--sac-rewards', type=float, nargs='+', required=True)
    parser.add_argument('--td3-rewards', type=float, nargs='+', required=True)
    parser.add_argument('--ppo-rewards', type=float, nargs='+', required=True)

    args = parser.parse_args()

    results = {
        'SAC': args.sac_rewards,
        'TD3': args.td3_rewards,
        'PPO': args.ppo_rewards
    }

    plot_comparison(results)
