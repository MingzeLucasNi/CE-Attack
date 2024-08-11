import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family':'sans-serif'})

# Data
LimeAttack = [91.2, 92.2, 91.1, 91.31]
HLBB = [91.2, 93.4, 95.4, 88.05]
TextHoaxer = [91.2, 92.4, 89.01, 88.85]
CEA_S = [91.2, 94.1, 92.1, 89.91]
hard_label = [LimeAttack, HLBB, TextHoaxer, CEA_S]

PWWS = [83.5, 85.1,81.11,76.83]
TF = [83.5, 84.11, 85.01, 78.11]
PSO = [83.5, 85.1,  83.11, 78.31]
RJA = [83.5, 85.71, 86.61,78.91]
CEA_H = [83.5, 85.71, 86.61, 78.91]
soft_label = [PWWS, TF, PSO, RJA, CEA_H]
data = [hard_label, soft_label]

# Plot parameters
width = 0.08 
font_size = 18
labels = ['Hard-label', 'Soft-label']

# Corrected legend names
names_hard = ['LimeAttack', 'HLBB', 'TextHoaxer', 'CEA_S']
names_soft = ['PWWS', 'TF', 'PSO', 'RJA', 'CEA_H']

markers = ['o', 'v', '^', '<', '>']
colors_hard = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
colors_soft = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
lines = ['-', '--', '-.', ':', '-']
linewidth = 1.5
markersize = 5
x = ["0%", "10%", "50%", "100%"]

# Plotting
fig, axs = plt.subplots(2, sharex=True, figsize=(10, 8))

# Plot each subplot with its own legend
for i, (label, names) in enumerate(zip(labels, [names_hard, names_soft])):
    colors = colors_hard if i == 0 else colors_soft  # Different colors for each subplot
    for j in range(len(data[i])):
        axs[i].plot(x, data[i][j], label=names[j], color=colors[j], marker=markers[j], linestyle=lines[j], linewidth=linewidth, markersize=markersize)

    axs[i].set_ylabel(label + ' Accuracy (%)', fontsize=font_size)
    axs[i].grid(True, linestyle="--", alpha=0.5)
    axs[i].tick_params(axis='both', which='major', labelsize=font_size)

    # Move legend to the top of each subplot and align horizontally
    axs[i].legend(fontsize=font_size, loc='upper center', bbox_to_anchor=(0.5, 1.23), ncol=len(names))

# Common x-label
axs[1].set_xlabel('Number of adversarial examples', fontsize=font_size)

fig.tight_layout()
plt.savefig('adv_accu_fixed.png', dpi=300)
plt.show()
