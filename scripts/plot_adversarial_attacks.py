"""
Plot relative performance drops under adversarial query perturbations.
Creates separate plots for each metric.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from Table 7: Relative performance drops (%) under adversarial attacks
# Organized by dataset, metric, model, and attack
data = {
    'TREC-DL': {
        'nDCG@10': {
            'Hypencoder': {'Misspelling': 32.4, 'Naturality': 25.7, 'Ordering': 1.0, 'Paraphrase': 20.9, 'Synonym': 16.5},
            'BE-base': {'Misspelling': 28.6, 'Naturality': 25.8, 'Ordering': 1.7, 'Paraphrase': 19.1, 'Synonym': 16.0},
            'FT TAS-B + Hype': {'Misspelling': 37.6, 'Naturality': 29.4, 'Ordering': 1.2, 'Paraphrase': 19.8, 'Synonym': 18.7},
            'TAS-B': {'Misspelling': 33.6, 'Naturality': 28.3, 'Ordering': 2.3, 'Paraphrase': 20.0, 'Synonym': 18.5}
        },
        'MRR': {
            'Hypencoder': {'Misspelling': 25.5, 'Naturality': 21.6, 'Ordering': 0.3, 'Paraphrase': 15.1, 'Synonym': 12.7},
            'BE-base': {'Misspelling': 20.1, 'Naturality': 22.6, 'Ordering': 1.3, 'Paraphrase': 15.3, 'Synonym': 13.6},
            'FT TAS-B + Hype': {'Misspelling': 26.9, 'Naturality': 26.1, 'Ordering': 2.6, 'Paraphrase': 16.5, 'Synonym': 15.4},
            'TAS-B': {'Misspelling': 21.8, 'Naturality': 25.6, 'Ordering': 0.3, 'Paraphrase': 17.0, 'Synonym': 12.6}
        },
        'R@1000': {
            'Hypencoder': {'Misspelling': 18.6, 'Naturality': 15.8, 'Ordering': 0.4, 'Paraphrase': 11.7, 'Synonym': 7.0},
            'BE-base': {'Misspelling': 18.1, 'Naturality': 14.4, 'Ordering': 1.2, 'Paraphrase': 11.5, 'Synonym': 6.8},
            'FT TAS-B + Hype': {'Misspelling': 20.8, 'Naturality': 15.8, 'Ordering': 0.3, 'Paraphrase': 13.4, 'Synonym': 7.6},
            'TAS-B': {'Misspelling': 21.3, 'Naturality': 16.7, 'Ordering': 0.3, 'Paraphrase': 12.7, 'Synonym': 7.9}
        }
    },
    'MSMARCO-Dev': {
        'MRR@10': {
            'Hypencoder': {'Misspelling': 46.0, 'Naturality': 30.7, 'Ordering': 2.1, 'Paraphrase': 27.2, 'Synonym': 20.7},
            'BE-base': {'Misspelling': 46.4, 'Naturality': 34.8, 'Ordering': 4.9, 'Paraphrase': 29.4, 'Synonym': 20.8},
            'FT TAS-B + Hype': {'Misspelling': 51.0, 'Naturality': 35.1, 'Ordering': 2.6, 'Paraphrase': 28.9, 'Synonym': 23.1},
            'TAS-B': {'Misspelling': 48.7, 'Naturality': 33.4, 'Ordering': 1.7, 'Paraphrase': 28.2, 'Synonym': 21.6}
        },
        'R@1000': {
            'Hypencoder': {'Misspelling': 15.2, 'Naturality': 12.1, 'Ordering': 0.2, 'Paraphrase': 11.3, 'Synonym': 5.5},
            'BE-base': {'Misspelling': 15.4, 'Naturality': 11.8, 'Ordering': 0.2, 'Paraphrase': 10.5, 'Synonym': 4.8},
            'FT TAS-B + Hype': {'Misspelling': 18.0, 'Naturality': 13.2, 'Ordering': 0.2, 'Paraphrase': 12.8, 'Synonym': 6.7},
            'TAS-B': {'Misspelling': 17.7, 'Naturality': 12.3, 'Ordering': 0.3, 'Paraphrase': 11.3, 'Synonym': 5.8}
        }
    }
}

# Attack types
attacks = ['Misspelling', 'Naturality', 'Ordering', 'Paraphrase', 'Synonym']

# Models
models = ['Hypencoder', 'BE-base', 'FT TAS-B + Hype', 'TAS-B']

# Colorblind-safe, paper-friendly colors (Okabe–Ito inspired)
colors = {
'Misspelling': '#0072B2', # Blue
'Naturality': '#D55E00', # Vermillion
'Ordering': '#999999', # Grey (structural perturbation)
'Paraphrase': '#009E73', # Green
'Synonym': '#CC79A7' # Purple
}


# Light, distinct hatching patterns (print-friendly)
patterns = {
'Misspelling': '', # solid
'Naturality': '//', # diagonal
'Ordering': '..', # dotted
'Paraphrase': 'xx', # cross
'Synonym': '\\\\', # opposite diagonal
}

def create_plot(dataset_name, metric_name, metric_data, output_dir):
    """Create a single plot for a specific dataset and metric."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set up bar positions
    x = np.arange(len(models))
    width = 0.15  # Width of each bar
    offsets = np.arange(len(attacks)) * width - (len(attacks) - 1) * width / 2

    # Plot bars for each attack
    for i, attack in enumerate(attacks):
        values = [metric_data[model][attack] for model in models]
        bars = ax.bar(x + offsets[i], values, width, 
                       label=attack,
                       color=colors[attack],
                       edgecolor='#666666',
                       linewidth=0.5,
                       hatch=patterns[attack])

    # Customize the plot
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative Performance Drop (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(title='Attack Type', fontsize=10, title_fontsize=11, 
              loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Set y-axis limits
    max_value = max([max(metric_data[model].values()) for model in models])
    ax.set_ylim(0, max_value * 1.1)

    plt.tight_layout()

    # Save the figure
    output_subdir = output_dir / dataset_name.replace(' ', '_').replace('&', 'and')
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    metric_filename = metric_name.replace('@', '_at_').replace('/', '_')
    output_path_png = output_subdir / f'{metric_filename}.png'
    output_path_pdf = output_subdir / f'{metric_filename}.pdf'
    
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    
    print(f"✅ Saved {dataset_name} {metric_name} plot to {output_subdir}/")
    plt.close()

# Create output directory
output_dir = Path('imgs/adversarial_attacks')

# Generate plots for each dataset and metric
for dataset_name, metrics in data.items():
    for metric_name, metric_data in metrics.items():
        create_plot(dataset_name, metric_name, metric_data, output_dir)

print(f"\n✅ All plots saved to {output_dir}/")

