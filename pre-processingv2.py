import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

files = ['UNSW-NB15_1.csv', 'UNSW-NB15_2.csv', 'UNSW-NB15_3.csv', 'UNSW-NB15_4.csv']
df_list = []

print("Loading raw CSV files...")
for file in files:
    temp = pd.read_csv(file, header=None, low_memory=False)
    df_list.append(temp)

data = pd.concat(df_list, ignore_index=True)
print(f"Total raw records: {len(data)}")

# Column indices in the raw 49-column dataset
col_attack_cat = 47
col_label = 48

# Clean attack_cat
col48 = data[col_attack_cat].fillna('').astype(str).str.strip()
mask_empty = (col48 == '') | (col48.str.lower() == 'nan')
col48.loc[mask_empty] = 'Normal'
mask_backdoor = col48.str.lower().str.startswith('backdoor')
col48.loc[mask_backdoor] = 'Backdoors'
mask_normal = col48.str.lower() == 'normal'
col48.loc[mask_normal] = 'Normal'
data[col_attack_cat] = col48

# Withhold zero-day classes
withheld_attacks = ["Worms", "Shellcode", "Backdoors"]
is_withheld = data[col_attack_cat].isin(withheld_attacks)
withheld_data = data[is_withheld].copy()
remaining_data = data[~is_withheld].copy()

print(f"Withheld zero-day samples: {len(withheld_data)}")
print(f"Remaining samples for train/test split: {len(remaining_data)}")

# TRUE STRATIFIED SPLIT (Fixes Reviewer W12)
# Stratify on both attack_cat and binary label to ensure balanced splits
train_data, test_regular = train_test_split(
    remaining_data, test_size=0.2, random_state=42, 
    stratify=remaining_data[[col_attack_cat, col_label]]
)

# Combine regular test with withheld data
test_data = pd.concat([test_regular, withheld_data], ignore_index=True)

# DROP IDENTIFIERS AND TIMESTAMPS (Fixes Reviewer W6)
# Indices to drop: 0(srcip), 1(sport), 2(dstip), 3(dsport), 20(stcpb), 21(dtcpb), 28(Stime), 29(Ltime)
drop_indices = [0, 1, 2, 3, 20, 21, 28, 29]
train_data = train_data.drop(columns=drop_indices)
test_data = test_data.drop(columns=drop_indices)

print(f"Dropped {len(drop_indices)} identifier/timestamp columns to prevent data leakage.")
print(f"Final feature count: {train_data.shape[1] - 2}")

# Sanity check
if train_data[col_attack_cat].isin(withheld_attacks).any():
    raise AssertionError('Sanity check failed: withheld attacks in training set!')

# Save clean files
train_data.to_csv('data/UNSW_NB15_train.csv', header=False, index=False)
test_data.to_csv('data/UNSW_NB15_test.csv', header=False, index=False)

print("\n✅ Preprocessing complete. Files saved to data/ directory.")
print(f"Train size: {len(train_data)}")
print(f"Test size:  {len(test_data)} (includes {len(withheld_data)} withheld)")