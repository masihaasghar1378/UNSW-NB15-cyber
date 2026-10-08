import pandas as pd
import numpy as np

files = {
    "Hybrid XGBoost": r"D:\PHD WORK\research for thesis\metrics_hybrid_xgboost_multiseed.csv",
    "Hybrid Random Forest": r"D:\PHD WORK\research for thesis\metrics_hybrid_rf_multiseed.csv",
    "Hybrid Decision Tree": r"D:\PHD WORK\research for thesis\metrics_hybrid_dt_multiseed.csv",
    "Hybrid CNN": r"D:\PHD WORK\research for thesis\metrics_hybrid_cnn_multiseed.csv",
    "Hybrid LSTM": r"D:\PHD WORK\research for thesis\metrics_hybrid_lstm_multiseed.csv",
}

metrics = ["Binary_Accuracy", "Multi_F1_Macro", "Multi_AUC_OVR", "FAR", "Unknown_Recall"]
# Adjust column names if your CSVs use slightly different names
# e.g., "Binary_Acc" instead of "Binary_Accuracy"

results = []
for model_name, path in files.items():
    try:
        df = pd.read_csv(path)
        row = {"Model": model_name}
        # Get the feature selection method and num features from first row
        row["Feature_Selection"] = df["Feature_Selection"].iloc[0]
        row["Num_Features"] = df["Num_Features_Selected"].iloc[0] if "Num_Features_Selected" in df.columns else df["Num_Features"].iloc[0]
        
        for m in metrics:
            # Try alternate column names
            col = m
            if col not in df.columns:
                col = m.replace("_Macro", "").replace("_OVR", "").replace("Accuracy", "Acc")
            if col not in df.columns:
                # Fuzzy match
                matches = [c for c in df.columns if m.split("_")[0].lower() in c.lower()]
                col = matches[0] if matches else None
            
            if col and col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                row[m] = f"{mean_val:.5f} ± {std_val:.5f}"
            else:
                row[m] = "N/A"
        
        results.append(row)
    except Exception as e:
        print(f"Error reading {path}: {e}")

summary = pd.DataFrame(results)
print("\n" + "="*80)
print("LEAKAGE-FREE OPERATIONAL PERFORMANCE (Validation-Derived Threshold)")
print("="*80)
print(summary.to_string(index=False))
summary.to_csv(r"D:\PHD WORK\research for thesis\Table2_LeakageFree_Summary.csv", index=False)
print("\nSaved to Table2_LeakageFree_Summary.csv")