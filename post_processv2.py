import pandas as pd
import numpy as np
import os

results_dir = 'results'

# Define the expected files based on the arguments we passed to train_hybrid.py
model_configs = [
    ("XGBOOST", "tree_importance"),
    ("RF", "tree_importance"),
    ("DT", "tree_importance"),
    ("CNN", "tree_importance"),
    ("LSTM", "variance")
]

# The new metrics we care about
metrics_to_agg = [
    "Binary_Acc", "Multi_F1", "Multi_AUC", "FAR", 
    "E2E_ZD_Recall", "E2E_ZD_Precision", "E2E_ZD_F1", "E2E_AUPRC"
]

summary_rows = []

print("="*80)
print("FINAL LEAKAGE-FREE OPERATIONAL RESULTS (Mean ± Std Dev across 5 Seeds)")
print("="*80)

for model, fs in model_configs:
    filename = f"metrics_{model}_{fs}.csv"
    filepath = os.path.join(results_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found! Did the run finish?")
        continue
        
    df = pd.read_csv(filepath)
    
    row = {
        "Model": model,
        "Feature_Selection": fs,
        "N_Features": df["N_Features"].iloc[0]
    }
    
    print(f"\n--- {model} ({fs}, {row['N_Features']} features) ---")
    
    for metric in metrics_to_agg:
        if metric in df.columns:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            row[f"{metric}_Mean"] = mean_val
            row[f"{metric}_Std"] = std_val
            formatted = f"{mean_val:.5f} ± {std_val:.5f}"
            print(f"{metric:15}: {formatted}")
        else:
            row[f"{metric}_Mean"] = np.nan
            row[f"{metric}_Std"] = np.nan
            
    summary_rows.append(row)

# Save Summary CSV
summary_df = pd.DataFrame(summary_rows)
out_path = os.path.join(results_dir, "FINAL_TABLE_SUMMARY.csv")
summary_df.to_csv(out_path, index=False)
print(f"\n✅ Saved final summary to {out_path}")

# --- ZERO-DAY EXPLAINABILITY ANALYSIS ---
print("\n" + "="*80)
print("ZERO-DAY EXPLAINABILITY (Top 5 Features driving the Autoencoder Flags)")
print("="*80)

for model, fs in model_configs:
    filename = f"zd_explain_{model}_{fs}.csv"
    filepath = os.path.join(results_dir, filename)
    
    if not os.path.exists(filepath):
        continue
        
    df_zd = pd.read_csv(filepath)
    
    # Group by feature and average the MSE contribution across all seeds and flagged samples
    top_features = df_zd.groupby("Feature")["MSE_Contribution"].mean().sort_values(ascending=False).head(5)
    
    print(f"\n{model} Top 5 Zero-Day Drivers:")
    for feat, mse in top_features.items():
        print(f"  - {feat}: {mse:.6f}")