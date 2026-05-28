import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast

# Define paths (update if needed)
base_path = r'D:\PHD WORK\research for thesis'
output_dir = os.path.join(base_path, 'analysis')
os.makedirs(output_dir, exist_ok=True)

# Model names and file mappings (lowercase for files)
models = ['xgboost', 'cnn', 'decisiontree', 'lstm', 'randomforest']
file_types = ['metrics_hybrid_', 'roc_curves_hybrid_', 'shap_summaries_hybrid_']

# Model name map
model_map = {
    'xgboost': 'Hybrid_XGBoost',
    'cnn': 'Hybrid_CNN',
    'decisiontree': 'Hybrid_DecisionTree',
    'lstm': 'Hybrid_LSTM',
    'randomforest': 'Hybrid_RandomForest'
}

# Colors and linestyles for distinguishability
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

# Feature methods
feature_methods = [
    'correlation', 'mutual_info', 'chi2', 'variance',
    'rfe', 'lasso', 'tree_importance', 'boruta'
]

# Load all CSVs
all_metrics = pd.DataFrame()
all_roc = pd.DataFrame()
all_shap = pd.DataFrame()

for model_lower in models:
    metrics_file = os.path.join(base_path, f'{file_types[0]}{model_lower}.csv')
    roc_file = os.path.join(base_path, f'{file_types[1]}{model_lower}.csv')
    shap_file = os.path.join(base_path, f'{file_types[2]}{model_lower}.csv')
    model_name = model_map[model_lower]
    
    if os.path.exists(metrics_file):
        df_metrics = pd.read_csv(metrics_file)
        df_metrics['Model'] = model_name
        all_metrics = pd.concat([all_metrics, df_metrics], ignore_index=True)
    
    if os.path.exists(roc_file):
        df_roc = pd.read_csv(roc_file)
        df_roc['Model'] = model_name
        all_roc = pd.concat([all_roc, df_roc], ignore_index=True)
    
    if os.path.exists(shap_file):
        df_shap = pd.read_csv(shap_file)
        df_shap['Model'] = model_name
        all_shap = pd.concat([all_shap, df_shap], ignore_index=True)

# 1. Process Metrics: Summary Table
summary = []
for model in all_metrics['Model'].unique():
    df_model = all_metrics[all_metrics['Model'] == model]
    if df_model.empty:
        continue
    best_idx = df_model['Multi_AUC_OVR'].idxmax()
    best_row = df_model.loc[best_idx]
    summary.append({
        'Model': model,
        'Best_FS_Method': best_row['Feature_Selection'],
        'Best_Binary_Accuracy': best_row['Binary_Accuracy'],
        'Best_Multi_F1_Macro': best_row['Multi_F1_Macro'],
        'Best_Multi_AUC_OVR': best_row['Multi_AUC_OVR'],
        'Best_FAR': best_row['FAR'],
        'Best_Unknown_Recall': best_row['Unknown_Recall'],
        'Avg_Time_min': df_model['Time_min'].mean()
    })
summary_df = pd.DataFrame(summary)
summary_df.sort_values('Best_Multi_AUC_OVR', ascending=False, inplace=True)
summary_path = os.path.join(output_dir, 'summary_metrics.csv')
summary_df.to_csv(summary_path, index=False)
print("Summary Metrics Table saved to:", summary_path)

# Full combined metrics table
combined_metrics_path = os.path.join(output_dir, 'combined_metrics.csv')
all_metrics.sort_values('Multi_AUC_OVR', ascending=False).to_csv(combined_metrics_path, index=False)
print("Combined Metrics Table saved to:", combined_metrics_path)

# Metrics Heatmap: Performance across Models and FS Methods
pivot_auc = all_metrics.pivot_table(index='Model', columns='Feature_Selection', values='Multi_AUC_OVR', aggfunc='mean')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_auc, annot=True, cmap='viridis', fmt='.3f')
plt.title('Heatmap of Multi_AUC_OVR Across Models and Feature Selection Methods')
plt.tight_layout()
heatmap_path = os.path.join(output_dir, 'metrics_heatmap_auc.png')
plt.savefig(heatmap_path, dpi=300)
plt.close()
print(f"Metrics Heatmap (AUC) saved to: {heatmap_path}")

# Similar heatmap for Unknown_Recall
pivot_recall = all_metrics.pivot_table(index='Model', columns='Feature_Selection', values='Unknown_Recall', aggfunc='mean')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_recall, annot=True, cmap='coolwarm', fmt='.3f')
plt.title('Heatmap of Unknown_Recall Across Models and Feature Selection Methods')
plt.tight_layout()
recall_heatmap_path = os.path.join(output_dir, 'metrics_heatmap_unknown_recall.png')
plt.savefig(recall_heatmap_path, dpi=300)
plt.close()
print(f"Metrics Heatmap (Unknown Recall) saved to: {recall_heatmap_path}")

# 2. Process ROC: Enhanced Plots
def plot_roc_curves(model_name, df, class_index=None):
    plt.figure(figsize=(12, 10))
    auc_texts = []
    
    for i, method in enumerate(feature_methods):
        method_df = df[(df['Feature_Selection'] == method)]
        if class_index is not None:
            method_df = method_df[method_df['Class_Index'] == class_index]
        method_df = method_df.sort_values('FPR')
        
        if not method_df.empty:
            fpr = method_df['FPR'].values
            tpr = method_df['TPR'].values
            auc_val = np.trapz(tpr, fpr)  # Compute AUC
            plt.plot(fpr, tpr,
                     color=colors[i % len(colors)],
                     linestyle=linestyles[i % len(linestyles)],
                     marker=markers[i % len(markers)] if i < 4 else None,
                     markersize=4, markevery=10,
                     linewidth=2.5, alpha=0.9,
                     label=f'{method} (AUC: {auc_val:.3f})')
            auc_texts.append(f'{method} AUC: {auc_val:.3f}')
    
    # Diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2.5, alpha=0.7, label='Random Classifier (AUC=0.5)')
    
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    class_title = f' (Class {class_index})' if class_index is not None else ''
    plt.title(f'ROC Curves for {model_name}\n(8 Feature Selection Methods){class_title}', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # Add AUC text box (bottom right)
    plt.text(0.95, 0.05, '\n'.join(auc_texts), fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    
    class_suffix = f'_class{class_index}' if class_index is not None else ''
    save_path = os.path.join(output_dir, f'roc_{model_name.lower()}{class_suffix}_enhanced.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" → Saved enhanced ROC: {save_path}")

# Generate ROC plots per model and class
for model in all_roc['Model'].unique():
    df_model_roc = all_roc[all_roc['Model'] == model]
    unique_classes = sorted(df_model_roc['Class_Index'].unique())
    for cls in unique_classes:
        plot_roc_curves(model, df_model_roc, class_index=cls)

# Combined ROC for best FS across models
if not summary_df.empty:
    best_fs = summary_df.iloc[0]['Best_FS_Method']
else:
    best_fs = 'boruta'  # Fallback
df_best_roc = all_roc[all_roc['Feature_Selection'] == best_fs]
if not df_best_roc.empty:
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_best_roc, x='FPR', y='TPR', hue='Model', style='Class_Index', palette='Set1')
    plt.title(f'Combined ROC Curves Across Models (FS: {best_fs})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    combined_roc_path = os.path.join(output_dir, f'combined_roc_{best_fs}.png')
    plt.savefig(combined_roc_path, dpi=300)
    plt.close()
    print(f"Combined ROC Plot ({best_fs}) saved to: {combined_roc_path}")

# 3. Process SHAP (XAI Results): Generate Plots and Tables
# Convert string lists to actual lists
def safe_literal_eval(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

all_shap['Mean_SHAP_Per_Feature'] = all_shap['Mean_SHAP_Per_Feature'].apply(safe_literal_eval)
all_shap['Features'] = all_shap['Features'].apply(safe_literal_eval)

# Summary SHAP table: Avg mean |SHAP| per class/model/FS
shap_summary = (
    all_shap
    .assign(Mean_SHAP=lambda df: df['Mean_SHAP_Per_Feature'].apply(
        lambda x: np.mean(np.abs(x)) if len(x) > 0 else np.nan
    ))
    .groupby(['Model', 'Feature_Selection', 'Class'])['Mean_SHAP']
    .mean()
    .reset_index()
)
shap_summary_path = os.path.join(output_dir, 'shap_summary.csv')
shap_summary.to_csv(shap_summary_path, index=False)
print("SHAP Summary Table saved to:", shap_summary_path)

# SHAP Bar Plots and Heatmaps: Per model and FS, using actual features
top_n_features = 20  # Limit to top N features per plot to avoid overcrowding

for model in all_shap['Model'].unique():
    df_model_shap = all_shap[all_shap['Model'] == model]
    if df_model_shap.empty:
        continue
    
    for fs in df_model_shap['Feature_Selection'].unique():
        df_fs = df_model_shap[df_model_shap['Feature_Selection'] == fs]
        if df_fs.empty:
            continue
        
        # Assume Features are the same for all classes in this FS/Model
        features = df_fs['Features'].iloc[0]  # Take from first row
        num_feats = len(features)
        
        # Validate all rows have same num_feats
        valid_mask = df_fs['Mean_SHAP_Per_Feature'].apply(len) == num_feats
        if not valid_mask.all():
            print(f"Warning: Inconsistent feature lengths for {model} {fs}; skipping.")
            continue
        
        # Expand SHAP values
        shap_expanded = pd.DataFrame(df_fs['Mean_SHAP_Per_Feature'].tolist(), columns=features)
        shap_expanded['Class'] = df_fs['Class'].values
        
        # Compute avg |SHAP| per feature (across classes) for sorting
        feature_avgs = shap_expanded.drop(columns='Class').abs().mean().sort_values(ascending=False)
        
        # For Bar Plot: Melt and sort by feature importance
        df_melt = shap_expanded.melt(id_vars=['Class'], var_name='Feature', value_name='SHAP_Value')
        df_melt['Abs_SHAP'] = df_melt['SHAP_Value'].abs()  # Use abs for importance
        
        # Group by Feature, avg over classes, sort descending
        feature_order = df_melt.groupby('Feature')['Abs_SHAP'].mean().sort_values(ascending=False).index
        
        # Limit to top N
        top_features = feature_order[:top_n_features]
        df_melt_top = df_melt[df_melt['Feature'].isin(top_features)]
        
        plt.figure(figsize=(12, max(8, len(top_features) * 0.5)))
        sns.barplot(data=df_melt_top, x='Abs_SHAP', y='Feature', hue='Class', orient='h', palette='viridis',
                    order=top_features)  # Order by importance
        plt.title(f'Mean |SHAP| Values for {model} ({fs}) - Top {top_n_features} Features')
        plt.xlabel('Mean Absolute SHAP Value')
        plt.ylabel('Features')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'shap_barplot_{model.lower()}_{fs}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"SHAP Bar Plot for {model} ({fs}) saved to: {plot_path}")
        
        # For Heatmap: Pivot on top N features
        shap_pivot = shap_expanded.set_index('Class')[top_features].T  # Features as rows, classes as cols
        
        plt.figure(figsize=(14, max(10, len(top_features) * 0.3)))
        sns.heatmap(shap_pivot.abs(), annot=True, cmap='YlGnBu', fmt='.3f', linewidths=0.5)  # Use abs for importance
        plt.title(f'SHAP Heatmap for {model} ({fs}) - Top {top_n_features} Features')
        plt.xlabel('Classes')
        plt.ylabel('Features')
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, f'shap_heatmap_{model.lower()}_{fs}.png')
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        print(f"SHAP Heatmap for {model} ({fs}) saved to: {heatmap_path}")

print("\nAnalysis Complete! All tables, plots, and heatmaps saved to:", output_dir)