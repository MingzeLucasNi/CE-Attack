import matplotlib.pyplot as plt
import numpy as np

# Data
x = ["BERT", "TextCNN"]
ori = [97, 93]
TextHoaxer = [51, 33]
HLBB = [49, 36]
LimeAttack = [24, 26]
CEA = [10, 11]
data = [ori, TextHoaxer, HLBB, LimeAttack, CEA]
names = ['No Attack', 'TextHoaxer', 'HLBB', 'LimeAttack', 'CEA']
# Use a popular color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
patterns = ['/', '\\', 'x', '-', '|']

# Plotting
fig, ax = plt.subplots(figsize=(14, 8))

bar_width = 0.15
x_indices = np.arange(len(x))

# Plot bars for each method
for i, (method, values) in enumerate(zip(names, data)):
    ax.bar(x_indices + i * bar_width, values, width=bar_width, label=method, color=colors[i], hatch=patterns[i % len(patterns)], edgecolor='black')

# Labels and title
# ax.set_xlabel("Models", fontsize=20)
ax.set_ylabel("Accuracy After Attack (%)", fontsize=28)
# ax.set_title("Attack Performance on BERT and TextCNN", fontsize=24)
ax.set_xticks(x_indices + bar_width * (len(data) - 1) / 2)
ax.set_xticklabels(x, fontsize=28)

# Adjust legend to be more compact and organized
ax.legend(fontsize=28, title='Attack Methods', title_fontsize=28, loc='best', ncol=1, bbox_to_anchor=(1, 1))

# Axis tick size
ax.tick_params(axis='y', labelsize=28)
ax.tick_params(axis='x', labelsize=28)

# Tight layout for better spacing
plt.tight_layout()

plt.show()
