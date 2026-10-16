import argparse
import os
import time
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, recall_score, precision_score, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM

# --- ARGPARSE ---
parser = argparse.ArgumentParser(description="Train Hybrid NIDS Pipeline")
parser.add_argument("--model", type=str, required=True, choices=["xgboost", "rf", "dt", "cnn", "lstm"], help="Model architecture")
parser.add_argument("--fs", type=str, default="tree_importance", choices=["tree_importance", "variance"], help="Feature selection method")
args = parser.parse_args()

MODEL_NAME = args.model.upper()
FS_METHOD = args.fs

# --- PATHS ---
TRAIN_PATH = 'data/UNSW_NB15_train.csv'
TEST_PATH = 'data/UNSW_NB15_test.csv'
OUT_DIR = 'results'
os.makedirs(OUT_DIR, exist_ok=True)

# --- CONFIG ---
SEEDS = [42, 123, 456, 789, 1024]
ZERO_DAY_TYPES = ['Worms', 'Shellcode', 'Backdoors']

# 39 Features remaining after dropping the 8 identifiers/timestamps
FEATURE_NAMES = ['proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']

# --- PREPROCESSING ---
def preprocess_df(df, encoders=None, high_card_cols=None, categorical_cols=None, is_train=False, train_freqs=None, train_medians=None):
    if is_train:
        categorical_cols, high_card_cols = [], []
        # FIX W11: Only process actual object/string columns. Do not cast low-cardinality numerics to strings.
        for col in df.columns:
            if df[col].dtype == 'object':
                if df[col].nunique() > 100: high_card_cols.append(col)
                else: categorical_cols.append(col)
        
        train_freqs = {}
        for col in high_card_cols:
            freq = df[col].value_counts(normalize=True)
            train_freqs[col] = freq
            df[col] = df[col].map(freq).fillna(0)
            
        encoders = {}
        for col in categorical_cols:
            col_values = df[col].astype(str).fillna('missing')
            unique_vals = np.unique(col_values)
            if 'missing' not in unique_vals: unique_vals = np.append(unique_vals, 'missing')
            le = LabelEncoder(); le.fit(unique_vals)
            df[col] = le.transform(col_values); encoders[col] = le
            
        train_medians = df.select_dtypes(include=[np.number]).median()
        df = df.fillna(train_medians)
        return df, encoders, high_card_cols, categorical_cols, train_freqs, train_medians
    else:
        for col in high_card_cols: df[col] = df[col].map(train_freqs.get(col, {})).fillna(0)
        for col in categorical_cols:
            le = encoders[col]
            col_values = df[col].astype(str).fillna('missing')
            col_values.loc[~col_values.isin(le.classes_)] = 'missing'
            df[col] = le.transform(col_values)
        df = df.fillna(train_medians)
        return df

print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH, header=None, low_memory=False)
test_df = pd.read_csv(TEST_PATH, header=None, low_memory=False)

# Safety check
train_df = train_df[~train_df.iloc[:, -2].isin(ZERO_DAY_TYPES)]
y_test_multi_str = test_df.iloc[:, -2].astype(str).values

train_df, encoders, hcc, cc, freqs, medians = preprocess_df(train_df, is_train=True)
test_df = preprocess_df(test_df, encoders=encoders, high_card_cols=hcc, categorical_cols=cc, train_freqs=freqs, train_medians=medians)

X_train = train_df.iloc[:, :-2].values.astype(np.float32)
y_train_binary = train_df.iloc[:, -1].values
y_train_multi_encoded = train_df.iloc[:, -2].values

X_test = test_df.iloc[:, :-2].values.astype(np.float32)
y_test_binary = test_df.iloc[:, -1].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

# Multi-class handling
attack_mask_train = (y_train_binary == 1)
y_train_attacks = y_train_multi_encoded[attack_mask_train]
le_multi = LabelEncoder()
le_multi.fit(np.unique(y_train_attacks))
y_train_attacks = le_multi.transform(y_train_attacks)
num_known_classes = len(le_multi.classes_)

is_zero_day_gt = np.isin(y_test_multi_str, ZERO_DAY_TYPES)
known_attack_mask = (y_test_binary == 1) & ~is_zero_day_gt
y_test_multi = np.full(len(y_test_binary), -1)
y_test_multi[known_attack_mask] = le_multi.transform(test_df.iloc[known_attack_mask, -2].values)
y_test_multi[is_zero_day_gt] = num_known_classes

# --- FEATURE SELECTION ---
def apply_fs(method, X_tr, y_tr, X_te):
    if method == 'tree_importance':
        clf = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        sel = SelectFromModel(clf, threshold='median', prefit=True)
        X_tr_sel = sel.transform(X_tr)
        X_te_sel = sel.transform(X_te)
        mask = sel.get_support()
    elif method == 'variance':
        sel = VarianceThreshold(threshold=0.01)
        X_tr_sel = sel.fit_transform(X_tr)
        X_te_sel = sel.transform(X_te)
        mask = sel.get_support()
    return X_tr_sel, X_te_sel, mask

# --- MODEL BUILDERS ---
def get_stage1_model(model_name, n_features, seed):
    if model_name == 'XGBOOST': return xgb.XGBClassifier(n_estimators=600, max_depth=12, learning_rate=0.1, scale_pos_weight=1.8, random_state=seed, n_jobs=-1, tree_method='hist', subsample=0.8, colsample_bytree=0.8)
    elif model_name == 'RF': return RandomForestClassifier(n_estimators=600, n_jobs=-1, random_state=seed, class_weight='balanced_subsample')
    elif model_name == 'DT': return DecisionTreeClassifier(max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=seed, class_weight='balanced')
    elif model_name == 'CNN':
        model = Sequential([Conv1D(128, 3, activation='relu', input_shape=(n_features, 1), padding='same'), MaxPooling1D(2), Conv1D(64, 3, activation='relu', padding='same'), MaxPooling1D(2), Conv1D(32, 3, activation='relu', padding='same'), Flatten(), Dense(64, activation='relu'), Dropout(0.4), Dense(1, activation='sigmoid')])
        model.compile(optimizer='adam', loss='binary_crossentropy'); return model
    elif model_name == 'LSTM':
        model = Sequential([LSTM(100, return_sequences=True, input_shape=(1, n_features)), Dropout(0.3), LSTM(50), Dropout(0.3), Dense(32, activation='relu'), Dense(1, activation='sigmoid')])
        model.compile(optimizer='adam', loss='binary_crossentropy'); return model
def get_stage2_model(model_name, n_features, n_classes, seed):
    if model_name == 'XGBOOST': return xgb.XGBClassifier(n_estimators=600, max_depth=12, learning_rate=0.1, objective='multi:softprob', num_class=n_classes, random_state=seed, n_jobs=-1, tree_method='hist', subsample=0.8, colsample_bytree=0.8)
    elif model_name == 'RF': return RandomForestClassifier(n_estimators=600, n_jobs=-1, random_state=seed, class_weight='balanced_subsample')
    elif model_name == 'DT': return DecisionTreeClassifier(max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=seed, class_weight='balanced')
    elif model_name == 'CNN':
        model = Sequential([Conv1D(128, 3, activation='relu', input_shape=(n_features, 1), padding='same'), MaxPooling1D(2), Conv1D(64, 3, activation='relu', padding='same'), MaxPooling1D(2), Conv1D(32, 3, activation='relu', padding='same'), Flatten(), Dense(64, activation='relu'), Dropout(0.4), Dense(n_classes, activation='softmax')])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy'); return model
    elif model_name == 'LSTM':
        model = Sequential([LSTM(100, return_sequences=True, input_shape=(1, n_features)), Dropout(0.3), LSTM(50), Dropout(0.3), Dense(32, activation='relu'), Dense(n_classes, activation='softmax')])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy'); return model

# --- MAIN LOOP ---
metrics_results = []
zd_explain_results = []

print(f"\nStarting {MODEL_NAME} with {FS_METHOD} across {len(SEEDS)} seeds...")

for seed in SEEDS:
    np.random.seed(seed); tf.random.set_seed(seed); random.seed(seed)
    start = time.time()
    
    X_tr_sel, X_te_sel, mask = apply_fs(FS_METHOD, X_train_scaled, y_train_binary, X_test_scaled)
    n_sel = X_tr_sel.shape[1]
    sel_cols = np.array(FEATURE_NAMES)[mask]
    
    # Reshape for DL
    if MODEL_NAME == 'CNN':
        X_tr_s1 = X_tr_sel.reshape((X_tr_sel.shape[0], n_sel, 1))
        X_te_s1 = X_te_sel.reshape((X_te_sel.shape[0], n_sel, 1))
    elif MODEL_NAME == 'LSTM':
        X_tr_s1 = X_tr_sel.reshape((X_tr_sel.shape[0], 1, n_sel))
        X_te_s1 = X_te_sel.reshape((X_te_sel.shape[0], 1, n_sel))
    else:
        X_tr_s1, X_te_s1 = X_tr_sel, X_te_sel

    # Stage 1
    m1 = get_stage1_model(MODEL_NAME, n_sel, seed)
    if MODEL_NAME in ['CNN', 'LSTM']:
        m1.fit(X_tr_s1, y_train_binary, epochs=20, batch_size=2048, verbose=0)
        probs_bin = m1.predict(X_te_s1, batch_size=2048, verbose=0).ravel()
    else:
        m1.fit(X_tr_s1, y_train_binary)
        probs_bin = m1.predict_proba(X_te_s1)[:, 1]
        
    preds_bin = (probs_bin > 0.5).astype(int)
    atk_mask_te = (preds_bin == 1)
    
    if not np.any(atk_mask_te): continue
        
    # Stage 2
    X_tr_atk = X_tr_sel[attack_mask_train]
    X_te_atk = X_te_sel[atk_mask_te]
    
    if MODEL_NAME == 'CNN':
        X_tr_atk_s2 = X_tr_atk.reshape((X_tr_atk.shape[0], n_sel, 1))
        X_te_atk_s2 = X_te_atk.reshape((X_te_atk.shape[0], n_sel, 1))
    elif MODEL_NAME == 'LSTM':
        X_tr_atk_s2 = X_tr_atk.reshape((X_tr_atk.shape[0], 1, n_sel))
        X_te_atk_s2 = X_te_atk.reshape((X_te_atk.shape[0], 1, n_sel))
    else:
        X_tr_atk_s2, X_te_atk_s2 = X_tr_atk, X_te_atk

    m2 = get_stage2_model(MODEL_NAME, n_sel, num_known_classes, seed)
    if MODEL_NAME in ['CNN', 'LSTM']:
        m2.fit(X_tr_atk_s2, y_train_attacks, epochs=20, batch_size=2048, verbose=0)
        probs_multi = m2.predict(X_te_atk_s2, batch_size=2048, verbose=0)
    else:
        m2.fit(X_tr_atk, y_train_attacks)
        probs_multi = m2.predict_proba(X_te_atk)
        
    preds_multi = np.argmax(probs_multi, axis=1)
    
    # Stage 3: AE with dynamic bottleneck (Fixes Reviewer W9)
    bottleneck = max(8, n_sel // 2) 
    ae = Sequential([
        Dense(n_sel, activation='relu', input_shape=(n_sel,)),
        Dense(bottleneck, activation='relu'),
        Dense(n_sel, activation='linear')
    ])
    ae.compile(optimizer='adam', loss='mse')
    
    X_tr_ae, X_val_ae = train_test_split(X_tr_atk, test_size=0.2, random_state=seed)
    ae.fit(X_tr_ae, X_tr_ae, epochs=20, batch_size=2048, verbose=0)
    
    threshold = np.percentile(np.mean((X_val_ae - ae.predict(X_val_ae))**2, axis=1), 95)
    recon = ae.predict(X_te_atk)
    errors = np.mean((X_te_atk - recon)**2, axis=1)
    unknown_mask = errors > threshold
    
    # Zero-Day Explainability (Fixes Reviewer W5)
    if np.any(unknown_mask):
        sq_errs = (X_te_atk[unknown_mask] - recon[unknown_mask])**2
        mean_feat_err = np.mean(sq_errs, axis=0)
        for i, fname in enumerate(sel_cols):
            zd_explain_results.append({"Seed": seed, "Model": MODEL_NAME, "Feature": fname, "MSE_Contribution": mean_feat_err[i]})

    # End-to-End Metrics (Fixes Reviewer W1, W3)
    final_is_unknown = np.zeros(len(y_test_binary), dtype=bool)
    final_is_unknown[atk_mask_te] = unknown_mask
    
    e2e_zd_recall = recall_score(is_zero_day_gt, final_is_unknown, zero_division=0)
    e2e_zd_precision = precision_score(is_zero_day_gt, final_is_unknown, zero_division=0)
    e2e_zd_f1 = f1_score(is_zero_day_gt, final_is_unknown, zero_division=0)
    
    full_anomaly_scores = np.zeros(len(y_test_binary))
    full_anomaly_scores[atk_mask_te] = errors
    if full_anomaly_scores.max() > 0: full_anomaly_scores /= full_anomaly_scores.max()
    e2e_auprc = average_precision_score(is_zero_day_gt, full_anomaly_scores)
    
    # Multi-class metrics (Fixes Reviewer W5 -1 Indexing Bug)
    gt_known_in_atk = ~is_zero_day_gt[atk_mask_te]
    f1_multi, auc_multi = 0, 0
    if np.any(gt_known_in_atk):
        y_known_sub = y_test_multi[atk_mask_te][gt_known_in_atk]
        preds_known_sub = preds_multi[gt_known_in_atk]
        probs_known_sub = probs_multi[gt_known_in_atk]
        
        f1_multi = f1_score(y_known_sub, preds_known_sub, average='macro', zero_division=0)
        present_classes = np.sort(np.unique(y_known_sub))
        if len(present_classes) > 1:
            probs_sub = probs_known_sub[:, present_classes]
            probs_sub /= probs_sub.sum(axis=1, keepdims=True) + 1e-10
            auc_multi = roc_auc_score(y_known_sub, probs_sub, multi_class='ovr', labels=present_classes)
            
    # Final Override for Confusion Matrix
    final_preds = np.full(len(y_test_binary), -1)
    final_preds_multi = preds_multi.copy()
    final_preds_multi[unknown_mask] = num_known_classes
    final_preds[atk_mask_te] = final_preds_multi
    
    cm = confusion_matrix(y_test_multi, final_preds)
    far = cm[0, 1:].sum() / (cm[0].sum() + 1e-9)
    acc = accuracy_score(y_test_binary, preds_bin)
    
    metrics_results.append({
        "Seed": seed, "Model": MODEL_NAME, "FS": FS_METHOD, "N_Features": n_sel,
        "Binary_Acc": round(acc, 5), "Multi_F1": round(f1_multi, 5), "Multi_AUC": round(auc_multi, 5),
        "FAR": round(far, 5), "E2E_ZD_Recall": round(e2e_zd_recall, 5), 
        "E2E_ZD_Precision": round(e2e_zd_precision, 5), "E2E_ZD_F1": round(e2e_zd_f1, 5),
        "E2E_AUPRC": round(e2e_auprc, 5), "Time_min": round((time.time()-start)/60, 2)
    })
    print(f"Seed {seed} done. E2E ZD Recall: {e2e_zd_recall:.4f} | AUPRC: {e2e_auprc:.4f}")

# Save
pd.DataFrame(metrics_results).to_csv(os.path.join(OUT_DIR, f'metrics_{MODEL_NAME}_{FS_METHOD}.csv'), index=False)
pd.DataFrame(zd_explain_results).to_csv(os.path.join(OUT_DIR, f'zd_explain_{MODEL_NAME}_{FS_METHOD}.csv'), index=False)
print(f"\n✅ Completed {MODEL_NAME}. Results saved to {OUT_DIR}/")
