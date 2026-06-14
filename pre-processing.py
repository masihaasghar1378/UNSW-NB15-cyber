import pandas as pd
import numpy as np

# List of CSV data files
files = ['UNSW-NB15_1.csv', 'UNSW-NB15_2.csv', 'UNSW-NB15_3.csv', 'UNSW-NB15_4.csv']
df_list = []
total_replaced = 0

for file in files:
    # Read CSV without headers
    temp = pd.read_csv(file, header=None, low_memory=False)
    
    if len(temp.columns) < 48:
        raise ValueError(f"File {file} has less than 48 columns.")
    
    # Python uses 0-based indexing, so column 48 is index 47
    col_idx = 47
    
    # --- Extract and normalize column 48 ---
    # Fill missing values with empty strings, convert to string, and trim whitespace
    col48 = temp[col_idx].fillna('').astype(str).str.strip()
    
    # Replace empty strings (or strings that became empty after stripping 'nan') with "Normal"
    mask_empty = (col48 == '') | (col48.str.lower() == 'nan')
    col48.loc[mask_empty] = 'Normal'
    total_replaced += mask_empty.sum()
    
    # --- Normalize common misspellings/variants of Backdoors (case-insensitive) ---
    lower_labels = col48.str.lower()
    
    # Any label that begins with 'backdoor'
    mask_backdoor_variants = lower_labels.str.startswith('backdoor')
    col48.loc[mask_backdoor_variants] = 'Backdoors'
    
    # Normalize 'normal' casing
    mask_normal_lower = (lower_labels == 'normal')
    col48.loc[mask_normal_lower] = 'Normal'
    
    # Assign corrected column back to the dataframe
    temp[col_idx] = col48
    
    # Append to list for merging
    df_list.append(temp)

# Merge into master table
data = pd.concat(df_list, ignore_index=True)

print("Finished loading and cleaning column 48.")

## -----------------------------------------------------------
#  WITHHOLD specific attack types (forced into test set)
## -----------------------------------------------------------

# Canonical withheld attack labels (must match the normalized labels above)
withheld_attacks = ["worms", "shellcode", "backdoors"] # Using lowercase for easier comparison

# Convert to string and use case-insensitive matching
attack_col_lower = data[47].astype(str).str.lower()
is_withheld = attack_col_lower.isin(withheld_attacks)

withheld_data = data[is_withheld]
remaining_data = data[~is_withheld]

## Split the remaining data 80/20 randomly
# Sample 80% for training (random_state ensures reproducibility, remove it if you want different splits each run)
train_data = remaining_data.sample(frac=0.8, random_state=42)
test_regular = remaining_data.drop(train_data.index)

## Final test set = regular test + all withheld attacks
test_data = pd.concat([test_regular, withheld_data], ignore_index=True)

# Reset index for clean training dataframe
train_data.reset_index(drop=True, inplace=True)

## Sanity check: assert no withheld attacks in train
train_attack_col_lower = train_data[47].astype(str).str.lower()
if train_attack_col_lower.isin(withheld_attacks).any():
    raise AssertionError('Sanity check failed: some withheld attack types ended up in the training set.')

## Save results (no headers)
train_data.to_csv('UNSW_NB15_train.csv', header=False, index=False)
test_data.to_csv('UNSW_NB15_test.csv', header=False, index=False)

print('\n🎉 Done.')
print(f'Replaced {total_replaced} empty cells in column 48.')
print(f'Train size: {len(train_data)} rows')
print(f'Test size:  {len(test_data)} rows (includes {len(withheld_data)} withheld attack rows)')
