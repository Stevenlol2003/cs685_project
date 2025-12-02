"""
Visualize LLM-as-Judge Evaluation Results

Generates summary statistics and visualizations for the scores in llm_judge_scores.json
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

INPUT_PATH = "results/evaluation/llm_judge_scores.json"
OUTPUT_DIR = "results/evaluation"

def load_scores(path: str) -> dict:
    """Load the LLM judge scores from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def calculate_statistics(scores: list) -> dict:
    """Calculate summary statistics for the scores."""
    scores_array = np.array(scores)
    return {
        "mean": float(np.mean(scores_array)),
        "median": float(np.median(scores_array)),
        "std": float(np.std(scores_array)),
        "min": int(np.min(scores_array)),
        "max": int(np.max(scores_array)),
        "q1": float(np.percentile(scores_array, 25)),
        "q3": float(np.percentile(scores_array, 75))
    }

def create_visualizations(scores: list, stats: dict, output_dir: str):
    """Create histogram and box plot visualizations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(scores, bins=10, range=(0.5, 10.5), edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]:.2f}')
    axes[0].axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f'Median: {stats["median"]:.2f}')
    axes[0].set_xlabel('Score (1-10)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of LLM Judge Scores', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Box plot
    bp = axes[1].boxplot(scores, vert=True, patch_artist=True, 
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5))
    axes[1].set_ylabel('Score (1-10)', fontsize=12)
    axes[1].set_title('Box Plot of LLM Judge Scores', fontsize=14, fontweight='bold')
    axes[1].set_xticklabels(['All Summaries'])
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add statistics text
    stats_text = f"n = {len(scores)}\nMean = {stats['mean']:.2f}\nMedian = {stats['median']:.2f}\nStd = {stats['std']:.2f}"
    axes[1].text(1.15, stats['median'], stats_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = Path(output_dir) / "llm_judge_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()

def print_summary(data: dict, stats: dict):
    """Print summary statistics to console."""
    print("\n" + "="*60)
    print("LLM-as-Judge Evaluation Summary")
    print("="*60)
    print(f"Model: {data['model']}")
    print(f"Timestamp: {data['timestamp']}")
    print(f"Total Evaluated: {data['num_evaluated']}")
    print("\n" + "-"*60)
    print("Score Statistics (Scale: 1-10)")
    print("-"*60)
    print(f"  Mean:        {stats['mean']:.2f}")
    print(f"  Median:      {stats['median']:.2f}")
    print(f"  Std Dev:     {stats['std']:.2f}")
    print(f"  Min:         {stats['min']}")
    print(f"  Max:         {stats['max']}")
    print(f"  Q1 (25%):    {stats['q1']:.2f}")
    print(f"  Q3 (75%):    {stats['q3']:.2f}")
    print("-"*60)
    
    # Score distribution
    score_bins = {i: 0 for i in range(1, 11)}
    for result in data['results']:
        score = result['scores']['total_score']
        if score in score_bins:
            score_bins[score] += 1
    
    print("\nScore Distribution:")
    print("-"*60)
    for score in range(1, 11):
        count = score_bins[score]
        pct = (count / data['num_evaluated']) * 100
        bar = '█' * int(pct / 2)
        print(f"  {score:2d}: {bar} {count:3d} ({pct:5.1f}%)")
    print("="*60 + "\n")

def main():
    # Load data
    data = load_scores(INPUT_PATH)
    
    # Extract scores
    scores = [result['scores']['total_score'] for result in data['results'] 
              if result['scores']['total_score'] > 0]
    
    # Calculate statistics
    stats = calculate_statistics(scores)
    
    # Print summary
    print_summary(data, stats)
    
    # Create visualizations
    create_visualizations(scores, stats, OUTPUT_DIR)
    
    # Save statistics to JSON
    stats_output = Path(OUTPUT_DIR) / "llm_judge_summary_stats.json"
    with open(stats_output, 'w') as f:
        json.dump({
            "model": data['model'],
            "timestamp": data['timestamp'],
            "num_evaluated": len(scores),
            "statistics": stats,
            "score_distribution": {str(i): sum(1 for s in scores if s == i) for i in range(1, 11)}
        }, f, indent=2)
    print(f"Summary statistics saved to: {stats_output}")

if __name__ == "__main__":
    main()
