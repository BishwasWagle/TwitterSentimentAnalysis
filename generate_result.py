import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create DataFrame from your experiment results
data = [
    {"trial": "trial_00000", "max_features": 1000, "min_df": 1, "max_df": 0.95, "n_estimators": 300, "max_depth": 30, "accuracy": 0.872433, "precision": 0.882478, "recall": 0.872433, "f1": 0.870003, "time": 57.5716},
    {"trial": "trial_00001", "max_features": 3000, "min_df": 3, "max_df": 0.99, "n_estimators": 200, "max_depth": 30, "accuracy": 0.858017, "precision": 0.87095, "recall": 0.858017, "f1": 0.853981, "time": 26.1409},
    {"trial": "trial_00002", "max_features": 3000, "min_df": 1, "max_df": 0.95, "n_estimators": 200, "max_depth": 30, "accuracy": 0.857849, "precision": 0.870584, "recall": 0.857849, "f1": 0.854094, "time": 26.6115},
    {"trial": "trial_00003", "max_features": 2000, "min_df": 3, "max_df": 0.95, "n_estimators": 300, "max_depth": 30, "accuracy": 0.865812, "precision": 0.877619, "recall": 0.865812, "f1": 0.862341, "time": 44.0651},
    {"trial": "trial_00004", "max_features": 1000, "min_df": 2, "max_df": 0.95, "n_estimators": 100, "max_depth": 20, "accuracy": 0.833962, "precision": 0.852847, "recall": 0.833962, "f1": 0.827225, "time": 14.3636},
    {"trial": "trial_00005", "max_features": 2000, "min_df": 3, "max_df": 0.9, "n_estimators": 300, "max_depth": 10, "accuracy": 0.719554, "precision": 0.765441, "recall": 0.719554, "f1": 0.696346, "time": 14.0156},
    {"trial": "trial_00006", "max_features": 2000, "min_df": 3, "max_df": 0.95, "n_estimators": 100, "max_depth": 30, "accuracy": 0.864722, "precision": 0.876367, "recall": 0.864722, "f1": 0.861355, "time": 17.0073},
    {"trial": "trial_00007", "max_features": 3000, "min_df": 3, "max_df": 0.99, "n_estimators": 100, "max_depth": None, "accuracy": 0.906378, "precision": 0.909541, "recall": 0.906378, "f1": 0.905977, "time": 42.9098},
    {"trial": "trial_00008", "max_features": 1000, "min_df": 1, "max_df": 0.9, "n_estimators": 300, "max_depth": 10, "accuracy": 0.7333, "precision": 0.77241, "recall": 0.7333, "f1": 0.714262, "time": 14.2902},
    {"trial": "trial_00009", "max_features": 3000, "min_df": 1, "max_df": 0.99, "n_estimators": 300, "max_depth": 10, "accuracy": 0.709915, "precision": 0.760913, "recall": 0.709915, "f1": 0.68468, "time": 9.34695}
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert None to a numerical value for plotting
df['max_depth_num'] = df['max_depth'].apply(lambda x: 40 if x is None else x)

# Set style
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Create a figure for all visualizations
fig = plt.figure(figsize=(18, 16))

# 1. Performance Metrics by Trial
ax1 = plt.subplot(2, 2, 1)
metrics_df = df[['trial', 'accuracy', 'precision', 'recall', 'f1']].melt(id_vars=['trial'], var_name='metric', value_name='value')
sns.barplot(x='trial', y='value', hue='metric', data=metrics_df, ax=ax1)
ax1.set_title('Performance Metrics by Trial')
ax1.set_xlabel('Trial')
ax1.set_ylabel('Score')
plt.setp(ax1.get_xticklabels(), rotation=45)
ax1.legend(title='Metric')

# 2. F1 Score vs Hyperparameters - max_features
ax2 = plt.subplot(2, 2, 2)
sns.scatterplot(x='max_features', y='f1', hue='max_depth_num', size='n_estimators', data=df, ax=ax2, palette='viridis', sizes=(100, 400))
ax2.set_title('F1 Score vs max_features')
ax2.set_xlabel('max_features')
ax2.set_ylabel('F1 Score')
handles, labels = ax2.get_legend_handles_labels()
# Replace 40 with "None" in the legend
labels = ["None" if l == "40" else l for l in labels]
ax2.legend(handles, labels, title='max_depth')

# 3. Top Models by F1 Score
ax3 = plt.subplot(2, 2, 3)
# Sort by F1 score
top_df = df.sort_values('f1', ascending=False).reset_index(drop=True)
# Plot top 5 models
sns.barplot(x='trial', y='f1', data=top_df.head(5), palette='viridis', ax=ax3)
ax3.set_title('Top 5 Models by F1 Score')
ax3.set_xlabel('Trial')
ax3.set_ylabel('F1 Score')
plt.setp(ax3.get_xticklabels(), rotation=45)
# Add value labels on bars
for i, v in enumerate(top_df.head(5)['f1']):
    ax3.text(i, v + 0.01, f'{v:.4f}', ha='center')

# 4. Best Model Performance
ax4 = plt.subplot(2, 2, 4)
best_model = df.loc[df['f1'].idxmax()]
metrics = ['accuracy', 'precision', 'recall', 'f1']
sns.barplot(x=metrics, y=[best_model[m] for m in metrics], palette='viridis', ax=ax4)
ax4.set_title(f'Best Model Performance (Trial {best_model["trial"]})')
ax4.set_xlabel('Metric')
ax4.set_ylabel('Score')
ax4.set_ylim([0.8, 1.0])  # Set the y-limit to focus on the high performance

# Add values on top of each bar
for i, v in enumerate([best_model[m] for m in metrics]):
    ax4.text(i, v+0.01, f'{v:.4f}', ha='center')

# Add best model parameters as text annotation
best_params = (
    f"Best Hyperparameters:\n"
    f"max_features: {best_model['max_features']}\n"
    f"min_df: {best_model['min_df']}\n"
    f"max_df: {best_model['max_df']}\n"
    f"n_estimators: {best_model['n_estimators']}\n"
    f"max_depth: {best_model['max_depth']}"
)
ax4.text(0.5, 0.5, best_params, transform=ax4.transAxes, 
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
         ha='center')

# Adjust layout
plt.tight_layout()
plt.savefig('output/ray_model_graphs/hyperparameter_tuning_results.png', dpi=300)
plt.show()

# Create additional visualization: Correlation between hyperparameters and F1 score
plt.figure(figsize=(10, 6))
correlation_data = df[['max_features', 'min_df', 'max_df', 'n_estimators', 'max_depth_num', 'f1']]
correlation_matrix = correlation_data.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation Between Hyperparameters and F1 Score')
plt.tight_layout()
plt.savefig('output/ray_model_graphs/correlation_heatmap.png', dpi=300)
plt.show()

# Create distribution of F1 scores
plt.figure(figsize=(10, 6))
sns.histplot(df['f1'], bins=10, kde=True)
plt.axvline(df['f1'].max(), color='red', linestyle='--', label=f'Best F1: {df["f1"].max():.4f}')
plt.title('Distribution of F1 Scores Across Trials')
plt.xlabel('F1 Score')
plt.ylabel('Count')
plt.legend()
plt.savefig('output/ray_model_graphs/f1_distribution.png', dpi=300)
plt.show()