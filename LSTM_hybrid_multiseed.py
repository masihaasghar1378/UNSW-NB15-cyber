import pandas as pd
import numpy as np
import time
import random
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve, accuracy_score, f1_score, roc_auc_score, confusion_matrix, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import shap

# ========================= CONFIG =========================
train_path = r'D:\PHD WORK\research for thesis\CPS paper\New folder\archive1\UNSW_NB15_train.csv'
test_path = r'D:\PHD WORK\research for thesis\CPS paper\New folder\archive1\UNSW_NB15_test.csv'
methods = ['variance'] 
metrics_results, roc_results, shap_results = [], [], []
zero_day_types = ['Worms', 'Shellcode', 'Backdoors']
feature_names = [f'feat_{i}' for i in range(47)]

# ========================= PREPROCESSING =========================
train_freqs, train_medians = {}, {}
def preprocess_df(df, encoders=None, high_card_cols=None, categorical_cols=None, is_train=False):
    global train_freqs, train_medians
    if is_train:
        categorical_cols, high_card_cols = [], []
        for col in df.columns:
            if df[col].dtype == 'object':
                if df[col].nunique() > 100: high_card_cols.append(col)
                else: categorical_cols.append(col)
            elif df[col].nunique() < 20: categorical_cols.append(col)
        train_freqs = {col: df[col].value_counts(normalize=True) for col in high_card_cols}
        for col in high_card_cols: df[col] = df[col].map(train_freqs[col]).fillna(0)
        encoders = {}
        for col in categorical_cols:
            col_values = df[col].astype(str).fillna('missing')
            unique_vals = np.unique(col_values)
            if 'missing' not in unique_vals: unique_vals = np.append(unique_vals, 'missing')
            le = LabelEncoder(); le.fit(unique_vals)
            df[col] = le.transform(col_values); encoders[col] = le
        train_medians = df.select_dtypes(include=[np.number]).median()
        df = df.fillna(train_medians)
    else:
        for col in high_card_cols: df[col] = df[col].map(train_freqs.get(col, {})).fillna(0)
        for col in categorical_cols:
            le = encoders[col]
            col_values = df[col].astype(str).fillna('missing')
            col_values.loc[~col_values.isin(le.classes_)] = 'missing'
            df[col] = le.transform(col_values)
        df = df.fillna(train_medians)
    return df, encoders, high_card_cols, categorical_cols

def apply_feature_selector(method, X_train_scaled, y_train, X_test_scaled, k=40):
    sel = VarianceThreshold(threshold=0.01)
    return sel.fit_transform(X_train_scaled), sel.transform(X_test_scaled), sel.get_support()

# ========================= LOAD DATA =========================
print("Loading...")
train_df = pd.read_csv(train_path, header=None, low_memory=False)
test_df = pd.read_csv(test_path, header=None, low_memory=False)
train_df = train_df[~train_df[47].isin(zero_day_types)]
y_test_multi_str = test_df[47].astype(str).values

train_df, encoders, hcc, cc = preprocess_df(train_df, is_train=True)
test_df, _, _, _ = preprocess_df(test_df, encoders=encoders, high_card_cols=hcc, categorical_cols=cc)

X_train = train_df.iloc[:, :-2].values.astype(np.float32)
y_train_binary = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-2].values.astype(np.float32)
y_test_binary = test_df.iloc[:, -1].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

attack_mask_train = (y_train_binary == 1)
y_train_attacks = train_df.iloc[:, 47].values[attack_mask_train]
le_multi = LabelEncoder(); le_multi.fit(np.unique(y_train_attacks))
y_train_attacks = le_multi.transform(y_train_attacks)
num_known_classes = len(le_multi.classes_)

is_zero_day_gt = np.isin(y_test_multi_str, zero_day_types)
known_attack_mask = (y_test_binary == 1) & ~is_zero_day_gt
y_test_multi = np.full(len(y_test_binary), -1)
y_test_multi[known_attack_mask] = le_multi.transform(test_df.iloc[known_attack_mask, 47].values)
y_test_multi[is_zero_day_gt] = num_known_classes

# ========================= LOOP =========================
seeds = [42, 123, 456, 789, 1024]
for seed in seeds:
    np.random.seed(seed); tf.random.set_seed(seed); random.seed(seed)
    for method in methods:
        start = time.time()
        print(f"\n SEED {seed} | STARTING Hybrid LSTM + {method.upper()}")
        X_train_sel, X_test_sel, mask = apply_feature_selector(method, X_train_scaled, y_train_binary, X_test_scaled, k=40)
        n_selected = X_train_sel.shape[1]
        selected_cols = np.array(feature_names)[mask] if mask is not None else np.array(feature_names)[:n_selected]
        
        # Reshape for LSTM
        X_train_lstm = X_train_sel.reshape((X_train_sel.shape[0], 1, n_selected))
        X_test_lstm = X_test_sel.reshape((X_test_sel.shape[0], 1, n_selected))
        
        model_binary = Sequential([
            LSTM(100, return_sequences=True, input_shape=(1, n_selected)), Dropout(0.3),
            LSTM(50), Dropout(0.3),
            Dense(32, activation='relu'), Dense(1, activation='sigmoid')
        ])
        model_binary.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model_binary.fit(X_train_lstm, y_train_binary, epochs=20, batch_size=2048, verbose=0)
        preds_binary = (model_binary.predict(X_test_lstm, batch_size=2048, verbose=0).ravel() > 0.5).astype(int)
        
        attack_mask_test = (preds_binary == 1)
        if np.any(attack_mask_test):
            X_train_attacks_lstm = X_train_lstm[attack_mask_train]
            X_test_attacks_lstm = X_test_lstm[attack_mask_test]
            X_train_attacks = X_train_sel[attack_mask_train] # Flat for AE
            X_test_attacks = X_test_sel[attack_mask_test]     # Flat for AE
            
            model_multi = Sequential([
                LSTM(100, return_sequences=True, input_shape=(1, n_selected)), Dropout(0.3),
                LSTM(50), Dropout(0.3),
                Dense(32, activation='relu'), Dense(num_known_classes, activation='softmax')
            ])
            model_multi.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model_multi.fit(X_train_attacks_lstm, y_train_attacks, epochs=20, batch_size=2048, verbose=0)
            probs_multi = model_multi.predict(X_test_attacks_lstm, batch_size=2048, verbose=0)
            preds_multi = np.argmax(probs_multi, axis=1)
            
            X_train_ae, X_val_ae = train_test_split(X_train_attacks, test_size=0.2, random_state=seed)
            ae = Sequential([Dense(64, activation='relu', input_shape=(n_selected,)), Dropout(0.3), Dense(32, activation='relu'), Dense(64, activation='relu'), Dense(n_selected, activation='linear')])
            ae.compile(optimizer='adam', loss='mse')
            ae.fit(X_train_ae, X_train_ae, epochs=20, batch_size=2048, verbose=0)
            threshold = np.percentile(np.mean((X_val_ae - ae.predict(X_val_ae))**2, axis=1), 95)
            errors = np.mean((X_test_attacks - ae.predict(X_test_attacks))**2, axis=1)
            unknown_mask = errors > threshold
            
            final_preds = np.full(len(y_test_binary), -1)
            final_preds_multi = preds_multi.copy()
            final_preds_multi[unknown_mask] = num_known_classes
            final_preds[attack_mask_test] = final_preds_multi
            
            background = X_train_attacks_lstm[np.random.choice(X_train_attacks_lstm.shape[0], 100, replace=False)]
            explainer = shap.GradientExplainer(model_multi, background)
            shap_values = explainer.shap_values(X_test_attacks_lstm[:1000])
            for class_idx in range(num_known_classes):
                mean_shap = np.mean(np.abs(shap_values[class_idx]), axis=0).flatten()
                shap_results.append({"Seed": seed, "Feature_Selection": method, "Class": le_multi.classes_[class_idx], "Mean_SHAP_Per_Feature": mean_shap.tolist(), "Features": selected_cols.tolist()})
            
            cm = confusion_matrix(y_test_multi, final_preds)
            far = cm[0, 1:].sum() / (cm[0].sum() + 1e-9)
            unknown_recall = recall_score(is_zero_day_gt[attack_mask_test], unknown_mask) if np.any(is_zero_day_gt & attack_mask_test) else 0
            test_labels = y_test_multi[attack_mask_test]
            known_mask = test_labels < num_known_classes
            f1_multi = f1_score(test_labels[known_mask], final_preds_multi[known_mask], average='macro') if np.any(known_mask) else 0
            
            present_classes = np.sort(np.unique(test_labels[known_mask])) if np.any(known_mask) else []
            auc_multi = 0
            if len(present_classes) > 1:
                probs_subset = probs_multi[known_mask][:, present_classes]
                probs_subset /= probs_subset.sum(axis=1, keepdims=True) + 1e-10
                auc_multi = roc_auc_score(test_labels[known_mask], probs_subset, multi_class='ovr', labels=present_classes)
                
            for class_idx in range(num_known_classes + 1):
                class_probs = probs_multi[:, class_idx] if class_idx < num_known_classes else errors / (errors.max() + 1e-9)
                fpr, tpr, _ = roc_curve((test_labels == class_idx).astype(int), class_probs)
                for fp, tp in zip(fpr, tpr): roc_results.append({"Seed": seed, "Feature_Selection": method, "Class_Index": class_idx, "FPR": round(fp, 8), "TPR": round(tp, 8)})
        else:
            far, unknown_recall, f1_multi, auc_multi = 0, 0, 0, 0
            
        metrics_results.append({"Seed": seed, "Feature_Selection": method, "Model": "Hybrid_LSTM", "Num_Features": n_selected, "Binary_Acc": round(accuracy_score(y_test_binary, preds_binary), 5), "Multi_F1": round(f1_multi, 5), "Multi_AUC": round(auc_multi, 5), "FAR": round(far, 5), "Unknown_Recall": round(unknown_recall, 5), "Time_min": round((time.time() - start)/60, 2)})

# ========================= SAVE =========================
pd.DataFrame(metrics_results).to_csv(r"D:\PHD WORK\research for thesis\metrics_hybrid_lstm_multiseed.csv", index=False)
pd.DataFrame(roc_results).to_csv(r"D:\PHD WORK\research for thesis\roc_curves_hybrid_lstm_multiseed.csv", index=False)
pd.DataFrame(shap_results).to_csv(r"D:\PHD WORK\research for thesis\shap_summaries_hybrid_lstm_multiseed.csv", index=False)
print("\n Hybrid LSTM Multi-Seed Benchmark Completed!")