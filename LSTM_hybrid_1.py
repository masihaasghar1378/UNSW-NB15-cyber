import pandas as pd
import numpy as np
import time
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
import xgboost as xgb
from boruta import BorutaPy
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import shap

# ========================= CONFIG =========================
#change the pathways to your current pathway
train_path = r'D:\PHD WORK\research for thesis\CPS paper\New folder\archive1\UNSW_NB15_train.csv'
test_path = r'D:\PHD WORK\research for thesis\CPS paper\New folder\archive1\UNSW_NB15_test.csv'
methods = ['correlation', 'mutual_info', 'chi2', 'variance', 'rfe', 'lasso', 'tree_importance', 'boruta']
metrics_results = []
roc_results = []
shap_results = []  # For XAI summaries
zero_day_types = ['Worms', 'Shellcode', 'Backdoors']  # For zero-day flagging

# Standard UNSW-NB15 feature names (47 features: columns 0-46)
feature_names = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl',
    'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin',
    'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime',
    'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd',
    'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm'
]

# ========================= ROBUST PREPROCESSING =========================
# (Unchanged, but add global vars)
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
    
        # Frequency encode high-cardinality
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
    mask = None  # For selected_cols
    if method == 'correlation':
        corr = np.corrcoef(X_train_scaled, rowvar=False)
        corr_abs = np.abs(corr)
        upper = np.triu(np.ones(corr_abs.shape), k=1).astype(bool)
        to_drop = [i for i in range(corr_abs.shape[0]) if any(corr_abs[i, upper[i]] > 0.9)]
        keep = [i for i in range(corr_abs.shape[0]) if i not in to_drop]
        X_train_sel = X_train_scaled[:, keep]
        X_test_sel = X_test_scaled[:, keep]
        mask = np.zeros(X_train_scaled.shape[1], dtype=bool)
        mask[keep] = True
    elif method == 'mutual_info':
        sel = SelectKBest(mutual_info_classif, k=k)
        X_train_sel = sel.fit_transform(X_train_scaled, y_train)
        X_test_sel = sel.transform(X_test_scaled)
        mask = sel.get_support()
    elif method == 'chi2':
        mm = MinMaxScaler()
        X_train_n = mm.fit_transform(X_train_scaled)
        X_test_n = mm.transform(X_test_scaled)
        sel = SelectKBest(chi2, k=k)
        X_train_sel = sel.fit_transform(X_train_n, y_train)
        X_test_sel = sel.transform(X_test_n)
        mask = sel.get_support()
    elif method == 'variance':
        sel = VarianceThreshold(threshold=0.01)
        X_train_sel = sel.fit_transform(X_train_scaled)
        X_test_sel = sel.transform(X_test_scaled)
        mask = sel.get_support()
    elif method == 'rfe':
        model = LogisticRegression(max_iter=200)
        sel = RFE(model, n_features_to_select=k)
        X_train_sel = sel.fit_transform(X_train_scaled, y_train)
        X_test_sel = sel.transform(X_test_scaled)
        mask = sel.get_support()
    elif method == 'lasso':
        lasso = LassoCV(cv=3, random_state=42)
        lasso.fit(X_train_scaled, y_train)
        mask = np.abs(lasso.coef_) > 1e-5
        X_train_sel = X_train_scaled[:, mask]
        X_test_sel = X_test_scaled[:, mask]
    elif method == 'tree_importance':
        xgb_clf = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        xgb_clf.fit(X_train_scaled, y_train)
        sel = SelectFromModel(xgb_clf, threshold='median', prefit=True)
        X_train_sel = sel.transform(X_train_scaled)
        X_test_sel = sel.transform(X_test_scaled)
        mask = sel.get_support()
    elif method == 'boruta':
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', random_state=42)
        boruta = BorutaPy(rf, n_estimators='auto', random_state=42, max_iter=60)
        boruta.fit(X_train_scaled, y_train)
        X_train_sel = boruta.transform(X_train_scaled)
        X_test_sel = boruta.transform(X_test_scaled)
        mask = boruta.support_
    else:
        raise ValueError(f"Unknown feature selector: {method}")
    print(f" → {method:15} → {X_train_sel.shape[1]} features")
    return X_train_sel, X_test_sel, mask

# ========================= LOAD DATA =========================
print("Loading and preprocessing data...")
# Memory-efficient read
train_df = pd.read_csv(train_path, header=None, low_memory=False)
test_df = pd.read_csv(test_path, header=None, low_memory=False)

# Filter rare/zero-day classes only from train
train_df = train_df[~train_df[47].isin(zero_day_types)]

# Capture original strings for test attack_cat (BEFORE preprocessing)
y_test_multi_str = test_df[47].astype(str).values

train_df, encoders, high_card_cols, categorical_cols = preprocess_df(train_df, is_train=True)
test_df, _, _, _ = preprocess_df(test_df, encoders=encoders, high_card_cols=high_card_cols, categorical_cols=categorical_cols)

X_train = train_df.iloc[:, :-2].values.astype(np.float32)
y_train_binary = train_df.iloc[:, -1].values
y_train_multi_encoded = train_df.iloc[:, 47].values  # encoded

X_test = test_df.iloc[:, :-2].values.astype(np.float32)
y_test_binary = test_df.iloc[:, -1].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = X_train_scaled.astype(np.float32)
X_test_scaled = X_test_scaled.astype(np.float32)
print(f"Data ready → {X_train_scaled.shape[1]} features")
print(f"Train memory: {X_train_scaled.nbytes / 1e9:.2f} GB")

# === MULTI-CLASS LABEL HANDLING ===
attack_mask_train = (y_train_binary == 1)
y_train_attacks = y_train_multi_encoded[attack_mask_train]

le_multi = LabelEncoder()
le_multi.fit(np.unique(y_train_attacks))
y_train_attacks = le_multi.transform(y_train_attacks)  # 0..k-1 for known attacks
num_known_classes = len(le_multi.classes_)

# Test labels
is_zero_day_gt = np.isin(y_test_multi_str, zero_day_types)
known_attack_mask = (y_test_binary == 1) & ~is_zero_day_gt
y_test_multi = np.full(len(y_test_binary), -1)  # -1 = benign
known_codes = test_df.iloc[known_attack_mask, 47].values
y_test_multi[known_attack_mask] = le_multi.transform(known_codes)
y_test_multi[is_zero_day_gt] = num_known_classes

# ========================= HYBRID LOOP =========================
for method in methods:
    start = time.time()
    print("="*80)
    print(f" STARTING Hybrid LSTM + {method.upper()} at {time.strftime('%H:%M:%S')}")
    X_train_sel, X_test_sel, mask = apply_feature_selector(method, X_train_scaled, y_train_binary, X_test_scaled, k=40)
    n_selected = X_train_sel.shape[1]
    selected_cols = np.array(feature_names)[mask] if mask is not None else np.array(feature_names)[:n_selected]  # Approx if no mask
    print(f" → No PCA → {n_selected} dimensions")
    
    # Reshape for LSTM
    X_train_lstm = X_train_sel.reshape((X_train_sel.shape[0], 1, n_selected))
    X_test_lstm = X_test_sel.reshape((X_test_sel.shape[0], 1, n_selected))
    
    # Stage 1: Binary Detection
    model_binary = Sequential([
        LSTM(100, return_sequences=True, input_shape=(1, n_selected)),
        Dropout(0.3),
        LSTM(50),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model_binary.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_binary.fit(X_train_lstm, y_train_binary, epochs=20, batch_size=2048, verbose=0, validation_split=0.1)
    probs_binary = model_binary.predict(X_test_lstm, batch_size=2048, verbose=0).ravel()
    preds_binary = (probs_binary > 0.5).astype(int)
    
    # Stage 2: Multi-Class Diagnosis on Detected Attacks
    attack_mask_test = (preds_binary == 1)
    if np.any(attack_mask_test):
        X_train_attacks_lstm = X_train_lstm[attack_mask_train]
        X_test_attacks_lstm = X_test_lstm[attack_mask_test]
        model_multi = Sequential([
            LSTM(100, return_sequences=True, input_shape=(1, n_selected)),
            Dropout(0.3),
            LSTM(50),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(num_known_classes, activation='softmax')
        ])
        model_multi.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model_multi.fit(X_train_attacks_lstm, y_train_attacks, epochs=20, batch_size=2048, verbose=0, validation_split=0.1)
        probs_multi = model_multi.predict(X_test_attacks_lstm, batch_size=2048, verbose=0)
        preds_multi = np.argmax(probs_multi, axis=1)
        
        # Unsupervised: Autoencoder for Zero-Day Flagging
        X_train_attacks = X_train_sel[attack_mask_train]
        X_test_attacks = X_test_sel[attack_mask_test]
        ae = Sequential([
            Dense(64, activation='relu', input_shape=(n_selected,)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(n_selected, activation='linear')
        ])
        ae.compile(optimizer='adam', loss='mse')
        ae.fit(X_train_attacks, X_train_attacks, epochs=20, batch_size=2048, verbose=0, validation_split=0.1)
        recon = ae.predict(X_test_attacks)
        errors = np.mean((X_test_attacks - recon)**2, axis=1)
        threshold = np.percentile(errors, 95)
        unknown_mask = errors > threshold
        
        # Override
        final_preds = np.full(len(y_test_binary), -1)
        final_preds_multi = preds_multi.copy()
        final_preds_multi[unknown_mask] = num_known_classes
        final_preds[attack_mask_test] = final_preds_multi
        
        # XAI: SHAP
        background = X_train_attacks_lstm[np.random.choice(X_train_attacks_lstm.shape[0], 100, replace=False)]
        explainer = shap.GradientExplainer(model_multi, background)
        shap_values = explainer.shap_values(X_test_attacks_lstm[:1000])
        for class_idx in range(num_known_classes):
            mean_shap = np.mean(np.abs(shap_values[class_idx]), axis=0).flatten()
            shap_results.append({
                "Feature_Selection": method,
                "Class": le_multi.classes_[class_idx],
                "Mean_SHAP_Per_Feature": mean_shap.tolist(),
                "Features": selected_cols.tolist()  # For plots
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
                class_probs = errors / errors.max()
            fpr, tpr, _ = roc_curve((test_labels == class_idx).astype(int), class_probs)
            for fp, tp in zip(fpr, tpr):
                roc_results.append({
                    "Feature_Selection": method,
                    "Class_Index": class_idx,
                    "FPR": round(fp, 8),
                    "TPR": round(tp, 8)
                })
    else:
        acc, f1_multi, auc_multi, far, unknown_recall = 0, 0, 0, 0, 0
    
    metrics_results.append({
        "Feature_Selection": method,
        "Model": "Hybrid_LSTM",
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
    r"D:\PHD WORK\research for thesis\metrics_hybrid_lstm.csv", index=False)
pd.DataFrame(roc_results).to_csv(
    r"D:\PHD WORK\research for thesis\roc_curves_hybrid_lstm.csv", index=False)
pd.DataFrame(shap_results).to_csv(
    r"D:\PHD WORK\research for thesis\shap_summaries_hybrid_lstm.csv", index=False)
print("\n" + "="*60)
print(" Hybrid LSTM Benchmark Completed!")
print("="*60)
