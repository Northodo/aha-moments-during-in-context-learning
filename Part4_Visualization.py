"""
Part 4: ICL Emergence Visualization
Task 5 (Visualization Group) - Convert abstract data to intuitive charts
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('Agg')
import numpy as np

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Read data
print("Reading data...")
df = pd.read_csv(os.path.join(script_dir, 'icl_emergence_results.csv'))

# Group by task and k_shots, calculate accuracy and mean entropy
summary = df.groupby(['task', 'k_shots']).agg(
    accuracy=('is_correct', 'mean'),
    mean_entropy=('entropy', 'mean'),
    std_entropy=('entropy', 'std'),
    count=('is_correct', 'count')
).reset_index()

print(f"\nData Overview:")
print(f"Tasks: {df['task'].unique().tolist()}")
print(f"k_shots range: {df['k_shots'].min()} ~ {df['k_shots'].max()}")
print(f"Total samples: {len(df)}")

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

tasks = df['task'].unique().tolist()
colors = ['#2E86AB', '#A23B72', '#F18F01']

for idx, task in enumerate(tasks):
    if idx >= 3:
        break
    task_data = summary[summary['task'] == task].sort_values('k_shots')
    ax = axes[idx]
    
    # Plot accuracy line chart (left axis)
    color1 = colors[idx]
    ax.plot(task_data['k_shots'], task_data['accuracy'], 
            marker='o', linewidth=2.5, markersize=6, 
            color=color1, label='Accuracy')
    
    # Fill accuracy area
    ax.fill_between(task_data['k_shots'], task_data['accuracy'], 
                    alpha=0.3, color=color1)
    
    # Plot entropy (right axis)
    ax2 = ax.twinx()
    color2 = '#C73E1D'
    ax2.bar(task_data['k_shots'], task_data['mean_entropy'], 
            alpha=0.5, color=color2, width=0.6, label='Entropy')
    ax2.set_ylabel('Entropy', color=color2, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Set title and labels
    ax.set_title(f'{task}', fontsize=13, fontweight='bold')
    ax.set_xlabel('k (number of examples)', fontsize=11)
    ax.set_ylabel('Accuracy', color=color1, fontsize=11)
    ax.tick_params(axis='y', labelcolor=color1)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set range
    ax.set_ylim(0, 1.1)
    ax.set_xlim(-0.5, task_data['k_shots'].max() + 0.5)
    
    # Mark emergence point (accuracy >= 0.6)
    emergence_point = task_data[task_data['accuracy'] >= 0.6]
    if len(emergence_point) > 0:
        first_emergence = emergence_point.iloc[0]
        ax.axvline(x=first_emergence['k_shots'], 
                   color='green', linestyle='--', 
                   linewidth=1.5, alpha=0.7)
        ax.text(first_emergence['k_shots'] + 0.3, 0.95, 
                f'Emergence\nk={int(first_emergence["k_shots"])}',
                fontsize=9, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', 
                         edgecolor='green', alpha=0.8))

plt.tight_layout()
output_path = os.path.join(script_dir, 'icl_emergence_visualization.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n[OK] Visualization saved to: {output_path}")

# Print summary
print("\n" + "="*60)
print("Emergence Analysis Summary")
print("="*60)

for task in tasks:
    task_data = summary[summary['task'] == task].sort_values('k_shots')
    max_acc = task_data['accuracy'].max()
    emergence_k = None
    for _, row in task_data.iterrows():
        if row['accuracy'] >= 0.6:
            emergence_k = int(row['k_shots'])
            break
    
    print(f"\n[{task}]")
    print(f"  Max Accuracy: {max_acc:.2%}")
    if emergence_k is not None:
        print(f"  Emergence k: {emergence_k}")
    else:
        print(f"  Emergence k: Not observed")
    
    # Entropy change
    if len(task_data) > 1:
        first_entropy = task_data.iloc[0]['mean_entropy']
        last_entropy = task_data.iloc[-1]['mean_entropy']
        direction = "decrease" if last_entropy < first_entropy else "increase"
        print(f"  Entropy: {first_entropy:.4f} -> {last_entropy:.4f} ({direction})")

print("\n" + "="*60)
print("Visualization Guide:")
print("  - Line chart (left axis): Accuracy vs k")
print("  - Bar chart (right axis): Entropy (uncertainty)")
print("  - Green dashed line: Emergence point (accuracy >= 60%)")
print("="*60)
