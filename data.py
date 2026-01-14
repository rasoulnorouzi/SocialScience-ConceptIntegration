# %%
import pandas as pd 
import numpy as np
# %%

df_train_p = pd.read_csv('datasets/processed_datasets/train_positive_pairs.csv')
df_train_p.head(20)
# %%
df_train_n = pd.read_csv('datasets/processed_datasets/train_negative_pairs.csv')
# %%
df_train_n.head(20)
# %%
df_train_n.tail(20)
# %%

# ============================================================================
# CHECK: No shared concepts between train and test
# ============================================================================
# %%
df_test_p = pd.read_csv('datasets/processed_datasets/test_positive_pairs.csv')
df_test_n = pd.read_csv('datasets/processed_datasets/test_negative_pairs.csv')

# Get all TERMS from train
train_terms = set(df_train_p['term1'].unique())
train_terms.update(df_train_p['term2'].unique())
train_terms.update(df_train_n['term1'].unique())
train_terms.update(df_train_n['term2'].unique())

# Get all TERMS from test
test_terms = set(df_test_p['term1'].unique())
test_terms.update(df_test_p['term2'].unique())
test_terms.update(df_test_n['term1'].unique())
test_terms.update(df_test_n['term2'].unique())

# Check overlap
overlap = train_terms & test_terms

print(f"Train terms: {len(train_terms)}")
print(f"Test terms:  {len(test_terms)}")
print(f"Overlap:     {len(overlap)}")

if overlap:
    print(f"\n⚠️ Shared terms found:")
    for term in list(overlap)[:10]:
        print(f"  - {term}")
else:
    print("\n✅ No overlap - train and test terms are completely separate!")
# %%

# ============================================================================
# SAVE AS JSONL FORMAT
# ============================================================================
# %%
import json

def save_as_jsonl(df, filename):
    filepath = f'datasets/processed_datasets/{filename}'
    with open(filepath, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')
    print(f"Saved: {filepath} ({len(df)} rows)")

save_as_jsonl(df_train_p, 'train_positive_pairs.json')
save_as_jsonl(df_train_n, 'train_negative_pairs.json')
save_as_jsonl(df_test_p, 'test_positive_pairs.json')
save_as_jsonl(df_test_n, 'test_negative_pairs.json')
# %%
