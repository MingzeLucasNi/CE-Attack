
import matplotlib.pyplot as plt
import numpy as np

# Data from the table
datasets = ['IMDB', 'SST2', 'WMT T1 (BD)', 'WMT T1 (SD)']
defenses = ['FGWS', 'RanMASK']

# Data values
data = {
    'IMDB': {
        'FGWS': {'CEA': 91, 'PSO': 80.0, 'RJA': 88.1},
        'RanMASK': {'CEA': 88.1, 'PSO': 81.0, 'RJA': 83.1},
    },
    'SST2': {
        'FGWS': {'CEA': 27.7, 'TextHoaxer': 19.9, 'HLBB': 21.2},
        'RanMASK': {'CEA': 20.7, 'TextHoaxer': 15.6, 'HLBB': 15.3},
    },
    'WMT T1 (BD)': {
        'FGWS': {'CEA': 17, 'Morph': 14, 'HAA': 10},
        'RanMASK': {'CEA': 14, 'Morph': 8, 'HAA': 6},
    },
    'WMT T1 (SD)': {
        'FGWS': {'CEA': 23, 'Morph': 18, 'HAA': 16},
        'RanMASK': {'CEA': 19, 'Morph': 16, 'HAA': 14},
    }
}

# Consistent colors for each method
colors = {
    'CEA': '#ff7f0e',
    'Morph': '#d62728',
    'HAA': '#ff9896',
    'TextHoaxer': '#fdae6b',
    'HLBB': '#ffbb78',
    'PSO': '#ff9896',
    'RJA': '#d62728'
}

# Plotting
fig, axes = plt.subplots(1, 4, figsize=(25, 6))
axes = axes.flatten()

for i, dataset in enumerate(datasets):
    ax = axes[i]
    methods = list(set([method for defense in defenses for method in data[dataset][defense].keys()]))
    methods.sort()
    
    # Calculate y-axis limits based on data
    all_values = [v for defense in defenses for v in data[dataset][defense].values()]
    y_max = max(all_values) + 10
    
    for method in methods:
        values = [data[dataset][defense].get(method, 0) for defense in defenses]
        ax.bar(np.arange(len(defenses)) + methods.index(method) * 0.2, values, 
               width=0.2, label=method, color=colors[method])
    
    ax.set_title(f'{dataset}', fontsize=25)
    ax.set_xticks(np.arange(len(defenses)) + 0.2)
    ax.set_xticklabels(defenses, fontsize=25)
    
    # Adjust y-axis label and limit
    if dataset == 'WMT T1 (BD)':
        ax.set_ylabel('BD', fontsize=25)
    elif dataset == 'WMT T1 (SD)':
        ax.set_ylabel('SD', fontsize=25)
    else:
        ax.set_ylabel('SAR', fontsize=25)
    ax.tick_params(axis='y', labelsize=17)  # Enlarge y-axis numbering
    ax.set_ylim(0, y_max)
    ax.legend(fontsize=25)
    ax.grid(True, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
