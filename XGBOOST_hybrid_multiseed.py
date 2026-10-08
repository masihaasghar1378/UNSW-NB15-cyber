import pandas as pd
import numpy as np
import time
import random
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve, accuracy_score, f1_score, roc_auc_score, confusion_matrix, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, chi2, VarianceThreshold,
    RFE, SelectFromModel
)
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
from boruta import BorutaPy
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import shap

# ========================= CONFIG =========================
train_path = r'D:\PHD WORK\research for thesis\CPS paper\New folder\archive1\UNSW_NB15_train.csv'
test_path = r'D:\PHD WORK\research for thesis\CPS paper\New folder\archive1\UNSW_NB15_test.csv'

# DOWNSCOPED: Only run the best method for XGBoost to save massive amounts of time!
methods = ['tree_importance'] 

metrics_results = []
roc_results = []
shap_results = []
zero_day_types = ['Worms', 'Shellcode', 'Backdoors']

feature_names = [f'feat_{i}' for i in range(47)]

# ========================= ROBUST PREPROCESSING =========================
train_freqs = {}
train_medians = {}

def preprocess_df(df, encoders=None, high_card_cols=None, categorical_cols=None, is_train=False):
    global train_freqs, train_medians
   
    if is_train:
        categorical_cols = []
        high_card_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_count = df[col].nunique()
                if unique_count > 100:
                    high_card_cols.append(col)
                else:
                    categorical_cols.append(col)
            elif df[col].nunique() < 20:
                categorical_cols.append(col)
    
        train_freqs = {}
        for col in high_card_cols:
            freq = df[col].value_counts(normalize=True)
            train_freqs[col] = freq
            df[col] = df[col].map(freq).fillna(0)
    
        encoders = {}
        for col in categorical_cols:
            col_values = df[col].astype(str).fillna('missing')
            unique_vals = np.unique(col_values)
            if 'missing' not in unique_vals:
                unique_vals = np.append(unique_vals, 'missing')
            le = LabelEncoder()
            le.fit(unique_vals)
            df[col] = le.transform(col_values)
            encoders[col] = le
        train_medians = df.select_dtypes(include=[np.number]).median()
        df = df.fillna(train_medians)
    else:
        for col in high_card_cols:
            df[col] = df[col].map(train_freqs.get(col, {})).fillna(0)
        for col in categorical_cols:
            le = encoders[col]
            col_values = df[col].astype(str).fillna('missing')
            unseen_mask = ~col_values.isin(le.classes_)
            col_values.loc[unseen_mask] = 'missing'
            df[col] = le.transform(col_values)
        df = df.fillna(train_medians)
    return df, encoders, high_card_cols, categorical_cols

# ========================= FEATURE SELECTION =========================
def apply_feature_selector(method, X_train_scaled, y_train, X_test_scaled, k=40):
    mask = None
    if method == 'tree_importance':
        xgb_clf = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        xgb_clf.fit(X_train_scaled, y_train)
        sel = SelectFromModel(xgb_clf, threshold='median', prefit=True)
        X_train_sel = sel.transform(X_train_scaled)
        X_test_sel = sel.transform(X_test_scaled)
        mask = sel.get_support()
    else:
        raise ValueError(f"Unknown feature selector: {method}")
    print(f" -> {method:15} -> {X_train_sel.shape[1]} features")
    return X_train_sel, X_test_sel, mask

# ========================= LOAD DATA =========================
print("Loading and preprocessing data...")
train_df = pd.read_csv(train_path, header=None, low_memory=False)
test_df = pd.read_csv(test_path, header=None, low_memory=False)

train_df = train_df[~train_df[47].isin(zero_day_types)]
y_test_multi_str = test_df[47].astype(str).values

train_df, encoders, high_card_cols, categorical_cols = preprocess_df(train_df, is_train=True)
test_df, _, _, _ = preprocess_df(test_df, encoders=encoders, high_card_cols=high_card_cols, categorical_cols=categorical_cols)

X_train = train_df.iloc[:, :-2].values.astype(np.float32)
y_train_binary = train_df.iloc[:, -1].values
y_train_multi_encoded = train_df.iloc[:, 47].values

X_test = test_df.iloc[:, :-2].values.astype(np.float32)
y_test_binary = test_df.iloc[:, -1].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = X_train_scaled.astype(np.float32)
X_test_scaled = X_test_scaled.astype(np.float32)
print(f"Data ready -> {X_train_scaled.shape[1]} features")

# === MULTI-CLASS LABEL HANDLING ===
attack_mask_train = (y_train_binary == 1)
y_train_attacks = y_train_multi_encoded[attack_mask_train]

le_multi = LabelEncoder()
le_multi.fit(np.unique(y_train_attacks))
y_train_attacks = le_multi.transform(y_train_attacks)
num_known_classes = len(le_multi.classes_)

is_zero_day_gt = np.isin(y_test_multi_str, zero_day_types)
known_attack_mask = (y_test_binary == 1) & ~is_zero_day_gt
y_test_multi = np.full(len(y_test_binary), -1)
known_codes = test_df.iloc[known_attack_mask, 47].values
y_test_multi[known_attack_mask] = le_multi.transform(known_codes)
y_test_multi[is_zero_day_gt] = num_known_classes

# ========================= MULTI-SEED HYBRID LOOP =========================
seeds = [42, 123, 456, 789, 1024]

for seed in seeds:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    
    for method in methods:
        start = time.time()
        print("="*80)
        print(f" SEED {seed} | STARTING Hybrid XGBoost + {method.upper()} at {time.strftime('%H:%M:%S')}")
        
        X_train_sel, X_test_sel, mask = apply_feature_selector(method, X_train_scaled, y_train_binary, X_test_scaled, k=40)
        n_selected = X_train_sel.shape[1]
        selected_cols = np.array(feature_names)[mask] if mask is not None else np.array(feature_names)[:n_selected]
        print(f" -> No PCA -> {n_selected} dimensions")
        
        # Stage 1: Binary Detection
        model_binary = xgb.XGBClassifier(
            n_estimators=600, max_depth=12, learning_rate=0.1, scale_pos_weight=1.8,
            random_state=seed, n_jobs=-1, tree_method='hist', subsample=0.8, colsample_bytree=0.8
        )
        model_binary.fit(X_train_sel, y_train_binary)
        probs_binary = model_binary.predict_proba(X_test_sel)[:, 1]
        preds_binary = (probs_binary > 0.5).astype(int)
        
        attack_mask_test = (preds_binary == 1)
        if np.any(attack_mask_test):
            X_train_attacks = X_train_sel[attack_mask_train]
            
            # Stage 2: Multi-Class Diagnosis
            model_multi = xgb.XGBClassifier(
                n_estimators=600, max_depth=12, learning_rate=0.1, objective='multi:softprob',
                num_class=num_known_classes, random_state=seed, n_jobs=-1, tree_method='hist',
                subsample=0.8, colsample_bytree=0.8
            )
            model_multi.fit(X_train_attacks, y_train_attacks)
            X_test_attacks = X_test_sel[attack_mask_test]
            probs_multi = model_multi.predict_proba(X_test_attacks)
            preds_multi = np.argmax(probs_multi, axis=1)
            
            # === LEAKAGE-FREE THRESHOLD CALIBRATION ===
            X_train_ae, X_val_ae = train_test_split(
                X_train_attacks, test_size=0.2, random_state=seed
            )

            ae = Sequential([
                Dense(64, activation='relu', input_shape=(n_selected,)),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dense(64, activation='relu'),
                Dense(n_selected, activation='linear')
            ])
            ae.compile(optimizer='adam', loss='mse')
            ae.fit(X_train_ae, X_train_ae, epochs=20, batch_size=2048, verbose=0)

            recon_val = ae.predict(X_val_ae)
            val_errors = np.mean((X_val_ae - recon_val)**2, axis=1)
            threshold = np.percentile(val_errors, 95)

            recon = ae.predict(X_test_attacks)
            errors = np.mean((X_test_attacks - recon)**2, axis=1)
            unknown_mask = errors > threshold
            
            # Override
            final_preds = np.full(len(y_test_binary), -1)
            final_preds_multi = preds_multi.copy()
            final_preds_multi[unknown_mask] = num_known_classes
            final_preds[attack_mask_test] = final_preds_multi
            
            # XAI: SHAP
            explainer = shap.TreeExplainer(model_multi)
            shap_values = explainer.shap_values(X_test_attacks[:1000])
            for class_idx in range(num_known_classes):
                mean_shap = np.mean(np.abs(shap_values[class_idx]), axis=0)
                shap_results.append({
                    "Seed": seed,
                    "Feature_Selection": method,
                    "Class": le_multi.classes_[class_idx],
                    "Mean_SHAP_Per_Feature": mean_shap.tolist(),
                    "Features": selected_cols.tolist()
                })
            
            # Metrics
            acc = accuracy_score(y_test_binary, preds_binary)
            cm = confusion_matrix(y_test_multi, final_preds)
            far = cm[0, 1:].sum() / (cm[0].sum() + 1e-9)
            unknown_recall = recall_score(is_zero_day_gt[attack_mask_test], unknown_mask) if np.any(is_zero_day_gt & attack_mask_test) else 0
            
            test_labels = y_test_multi[attack_mask_test]
            known_mask = test_labels < num_known_classes
            f1_multi = 0
            auc_multi = 0
            if np.any(known_mask):
                y_known = test_labels[known_mask]
                probs_known = probs_multi[known_mask]
                preds_known = final_preds_multi[known_mask]
                f1_multi = f1_score(y_known, preds_known, average='macro')
                present_classes = np.unique(y_known)
                if len(present_classes) > 1:
                    present_classes = np.sort(present_classes)
                    probs_subset = probs_known[:, present_classes]
                    probs_subset /= probs_subset.sum(axis=1, keepdims=True) + 1e-10
                    auc_multi = roc_auc_score(y_known, probs_subset, multi_class='ovr', labels=present_classes)
            
            # ROC
            for class_idx in range(num_known_classes + 1):
                if class_idx < num_known_classes:
                    class_probs = probs_multi[:, class_idx]
                else:
                    class_probs = errors / (errors.max() + 1e-9)
                fpr, tpr, _ = roc_curve((test_labels == class_idx).astype(int), class_probs)
                for fp, tp in zip(fpr, tpr):
                    roc_results.append({
                        "Seed": seed,
                        "Feature_Selection": method,
                        "Class_Index": class_idx,
                        "FPR": round(fp, 8),
                        "TPR": round(tp, 8)
                    })
        else:
            acc, f1_multi, auc_multi, far, unknown_recall = 0, 0, 0, 0, 0
        
        metrics_results.append({
            "Seed": seed,
            "Feature_Selection": method,
            "Model": "Hybrid_XGBoost",
            "Num_Features_Selected": n_selected,
            "Binary_Accuracy": round(acc, 5),
            "Multi_F1_Macro": round(f1_multi, 5),
            "Multi_AUC_OVR": round(auc_multi, 5),
            "FAR": round(far, 5),
            "Unknown_Recall": round(unknown_recall, 5),
            "Time_min": round((time.time() - start)/60, 2)
        })

# ========================= SAVE CLEAN FILES =========================
pd.DataFrame(metrics_results).sort_values("Multi_AUC_OVR", ascending=False).round(5).to_csv(
    r"D:\PHD WORK\research for thesis\metrics_hybrid_xgboost_multiseed.csv", index=False)
pd.DataFrame(roc_results).to_csv(
    r"D:\PHD WORK\research for thesis\roc_curves_hybrid_xgboost_multiseed.csv", index=False)
pd.DataFrame(shap_results).to_csv(
    r"D:\PHD WORK\research for thesis\shap_summaries_hybrid_xgboost_multiseed.csv", index=False)

print("\n" + "="*60)
print(" Hybrid XGBoost Multi-Seed Benchmark Completed!")
print("="*60)