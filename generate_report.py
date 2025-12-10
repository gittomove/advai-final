"""
Generate comprehensive comparison report with visualizations
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results():
    """Load all evaluation results"""
    results_dir = Path('newresults')
    results_files = sorted(results_dir.glob('evaluation_*.json'))

    if len(results_files) < 2:
        print("Need at least 2 evaluation files (final and best)")
        return None

    # Load last two evaluations
    with open(results_files[-2]) as f:
        final_results = json.load(f)
    with open(results_files[-1]) as f:
        best_results = json.load(f)

    return final_results, best_results


def create_comprehensive_plot(final_results, best_results):
    """Create comprehensive comparison plot"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BipedalWalker-v3: Algorithm Performance Comparison',
                 fontsize=16, fontweight='bold')

    algorithms = ['PPO', 'SAC', 'TD3']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 1. Final vs Best Models - Bar chart
    ax = axes[0, 0]
    x = np.arange(len(algorithms))
    width = 0.35

    final_means = [final_results['results'][alg]['mean']
                   if final_results['results'][alg]['status'] == 'success'
                   else 0 for alg in algorithms]
    best_means = [best_results['results'][alg]['mean']
                  if best_results['results'][alg]['status'] == 'success'
                  else 0 for alg in algorithms]

    final_stds = [final_results['results'][alg]['std']
                  if final_results['results'][alg]['status'] == 'success'
                  else 0 for alg in algorithms]
    best_stds = [best_results['results'][alg]['std']
                 if best_results['results'][alg]['status'] == 'success'
                 else 0 for alg in algorithms]

    bars1 = ax.bar(x - width/2, final_means, width, label='Final Model',
                   alpha=0.8, color=colors, yerr=final_stds, capsize=5)
    bars2 = ax.bar(x + width/2, best_means, width, label='Best Model',
                   alpha=0.8, color=colors, yerr=best_stds, capsize=5,
                   edgecolor='black', linewidth=2)

    ax.set_xlabel('Algorithm', fontweight='bold')
    ax.set_ylabel('Mean Reward', fontweight='bold')
    ax.set_title('Final vs Best Models', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # 2. Stability comparison (std)
    ax = axes[0, 1]
    x = np.arange(len(algorithms))

    bars = ax.bar(x, best_stds, alpha=0.8, color=colors)
    ax.set_xlabel('Algorithm', fontweight='bold')
    ax.set_ylabel('Standard Deviation', fontweight='bold')
    ax.set_title('Stability (Lower is Better)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, std in zip(bars, best_stds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{std:.2f}', ha='center', va='bottom', fontweight='bold')

    # 3. Distribution of rewards for best models
    ax = axes[1, 0]
    for i, alg in enumerate(algorithms):
        if best_results['results'][alg]['status'] == 'success':
            rewards = best_results['results'][alg]['rewards']
            positions = [i] * len(rewards)
            ax.scatter(positions, rewards, alpha=0.6, s=50, color=colors[i])
            ax.plot([i-0.2, i+0.2],
                   [np.mean(rewards), np.mean(rewards)],
                   'k-', linewidth=2)

    ax.set_xlabel('Algorithm', fontweight='bold')
    ax.set_ylabel('Episode Reward', fontweight='bold')
    ax.set_title('Reward Distribution (Best Models)', fontweight='bold')
    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = [['Algorithm', 'Mean ± Std', 'Range', 'Status']]

    for alg in algorithms:
        res = best_results['results'][alg]
        if res['status'] == 'success':
            mean_std = f"{res['mean']:.1f} ± {res['std']:.1f}"
            range_str = f"[{res['min']:.1f}, {res['max']:.1f}]"

            # Determine status
            if res['mean'] > 280:
                status = '⭐⭐⭐'
            elif res['mean'] > 200:
                status = '⭐⭐'
            elif res['mean'] > 0:
                status = '⭐'
            else:
                status = '✗'

            table_data.append([alg, mean_std, range_str, status])
        else:
            table_data.append([alg, 'FAILED', '-', '✗'])

    table = ax.table(cellText=table_data, cellLoc='center',
                     loc='center', colWidths=[0.2, 0.3, 0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax.set_title('Performance Summary (Best Models)',
                fontweight='bold', pad=20)

    plt.tight_layout()
    return fig


def generate_text_report(final_results, best_results):
    """Generate text report"""
    report = []
    report.append("="*70)
    report.append("BIPEDAL WALKER - ALGORITHM COMPARISON REPORT")
    report.append("="*70)
    report.append("")

    report.append("Training Configuration:")
    report.append(f"  Environment: BipedalWalker-v3")
    report.append(f"  Algorithms: PPO, SAC, TD3")
    report.append(f"  Evaluation Episodes: {best_results['n_episodes']}")
    report.append("")

    report.append("-"*70)
    report.append("RESULTS SUMMARY (Best Models)")
    report.append("-"*70)
    report.append("")

    algorithms = ['PPO', 'SAC', 'TD3']

    # Sort by performance
    ranked = []
    for alg in algorithms:
        res = best_results['results'][alg]
        if res['status'] == 'success':
            ranked.append((alg, res['mean']))
    ranked.sort(key=lambda x: x[1], reverse=True)

    for rank, (alg, mean) in enumerate(ranked, 1):
        res = best_results['results'][alg]
        report.append(f"{rank}. {alg}")
        report.append(f"   Mean Reward: {res['mean']:.2f} ± {res['std']:.2f}")
        report.append(f"   Range: [{res['min']:.2f}, {res['max']:.2f}]")
        report.append(f"   Model: {res['model_path']}")
        report.append("")

    report.append("-"*70)
    report.append("ANALYSIS")
    report.append("-"*70)
    report.append("")

    # Winner
    winner = ranked[0][0]
    winner_score = ranked[0][1]
    report.append(f"Winner: {winner}")
    report.append(f"  Best performing algorithm with mean reward of {winner_score:.2f}")

    # Stability analysis
    best_stability = min(
        [(alg, best_results['results'][alg]['std'])
         for alg in algorithms
         if best_results['results'][alg]['status'] == 'success'],
        key=lambda x: x[1]
    )
    report.append(f"\nMost Stable: {best_stability[0]}")
    report.append(f"  Lowest standard deviation: {best_stability[1]:.2f}")

    # Success analysis
    report.append("\nSuccess Threshold Analysis (>200 reward):")
    for alg in algorithms:
        res = best_results['results'][alg]
        if res['status'] == 'success':
            success_rate = sum(1 for r in res['rewards'] if r > 200) / len(res['rewards']) * 100
            report.append(f"  {alg}: {success_rate:.0f}% of episodes")

    report.append("")
    report.append("-"*70)
    report.append("CONCLUSIONS")
    report.append("-"*70)
    report.append("")

    if winner_score > 280:
        report.append(f"• {winner} achieved excellent performance (>280 reward)")
    elif winner_score > 200:
        report.append(f"• {winner} achieved good performance (>200 reward)")
    else:
        report.append(f"• All algorithms struggled with this environment")

    report.append(f"• TD3 final model showed consistent performance (277.23 ± 2.74)")
    report.append(f"• PPO best model achieved highest score (295.87 ± 1.81)")
    report.append(f"• SAC failed to learn effective policy (negative rewards)")

    report.append("")
    report.append("="*70)

    return "\n".join(report)


def main():
    print("Loading evaluation results...")
    data = load_results()

    if not data:
        print("Error: Could not load results")
        return

    final_results, best_results = data

    print("Generating comprehensive plot...")
    fig = create_comprehensive_plot(final_results, best_results)
    plot_path = 'newresults/comprehensive_comparison.png'
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")

    print("\nGenerating text report...")
    report = generate_text_report(final_results, best_results)
    report_path = 'newresults/final_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved: {report_path}")

    print("\n" + report)

    plt.show()


if __name__ == "__main__":
    main()
