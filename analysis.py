import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import torch

# Reproducibility Setup
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. Load Data
print("Loading datasets...")
try:
    pos_df = pd.read_csv('datasets/processed_datasets/train_positive_pairs.csv', on_bad_lines='skip')
    neg_df = pd.read_csv('datasets/processed_datasets/train_negative_pairs.csv', on_bad_lines='skip')
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

# 2. Random Sampling
print(f"Positive samples: {len(pos_df)}")
print(f"Negative samples (total): {len(neg_df)}")

n_pos = len(pos_df)
if len(neg_df) > n_pos:
    neg_df_sampled = neg_df.sample(n=n_pos, random_state=SEED)
else:
    neg_df_sampled = neg_df
print(f"Negative samples (sampled): {len(neg_df_sampled)}")

# Combine and cleaning
df = pd.concat([pos_df, neg_df_sampled])
# Shuffle entire dataset
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

df['term1'] = df['term1'].astype(str)
df['term2'] = df['term2'].astype(str)
labels = df['label'].values

# Models to evaluate
models_to_test = ['all-mpnet-base-v2', 'dwulff/mpnet-personality', 'allenai/scibert_scivocab_uncased', 'bert-base-uncased']
all_summary_data = []

for model_name in models_to_test:
    print(f"\n{'='*40}")
    print(f" Processing Model: {model_name}")
    print(f"{'='*40}")

    # 3. Model & Encoding
    print(f"Loading Model ({model_name})...")
    try:
        if model_name in ['allenai/scibert_scivocab_uncased', 'bert-base-uncased']:
            word_embedding_model = models.Transformer(model_name)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        else:
            model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        continue

    print("Encoding sentences...")
    embeddings1 = model.encode(df['term1'].tolist(), convert_to_tensor=True, show_progress_bar=False, batch_size=16)
    embeddings2 = model.encode(df['term2'].tolist(), convert_to_tensor=True, show_progress_bar=False, batch_size=16)

    # 4. Compute Cosine Similarities
    print("Computing Cosine Similarities...")
    similarities = torch.nn.functional.cosine_similarity(embeddings1, embeddings2).cpu().numpy()

    # 5. Threshold Optimization
    print("Starting Threshold Optimization...")
    thresholds = np.arange(0.10, 1.00, 0.01)

    # 6. Aggregating
    print(f"\n--- Results per Threshold ({model_name}) ---")
    print(f"{'Threshold':<10} | {'F1':<10} | {'Precision':<10} | {'Recall':<10} | {'PosAcc':<10} | {'NegAcc':<10}")
    print("-" * 80)

    current_model_results = []
    
    # Track best for this model
    best_f1_model = -1
    best_threshold_model = -1
    
    for t in thresholds:
        preds = (similarities > t).astype(int)
        
        # Calculate metrics
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        
        # Class-specific accuracy
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        
        pos_acc = np.mean(preds[pos_mask] == 1) if np.any(pos_mask) else 0.0
        neg_acc = np.mean(preds[neg_mask] == 0) if np.any(neg_mask) else 0.0

        # Update specific model best
        if f1 > best_f1_model:
            best_f1_model = f1
            best_threshold_model = t
        
        result_entry = {
            'model': model_name,
            'threshold': t,
            'precision_mean': p, 'precision_var': 0.0,
            'recall_mean': r, 'recall_var': 0.0,
            'f1_mean': f1, 'f1_var': 0.0,
            'pos_acc_mean': pos_acc,
            'neg_acc_mean': neg_acc
        }
        all_summary_data.append(result_entry)
        current_model_results.append(result_entry)
        
        print(f"{t:.2f}       | {f1:.4f}     | {p:.4f}     | {r:.4f}     | {pos_acc:.4f}     | {neg_acc:.4f}")

    # Generate Individual Plot
    print(f"Generating plot for {model_name}...")
    current_df = pd.DataFrame(current_model_results)
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=current_df, x='threshold', y='f1_mean', label='F1 Score', linewidth=2.5)
    sns.lineplot(data=current_df, x='threshold', y='precision_mean', label='Precision', linestyle='--')
    sns.lineplot(data=current_df, x='threshold', y='recall_mean', label='Recall', linestyle='--')
    
    plt.axvline(x=best_threshold_model, color='r', linestyle=':', label=f'Best Threshold ({best_threshold_model:.2f})')
    
    plt.title(f'Model Performance: {model_name}', fontsize=16)
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    safe_name = model_name.replace('/', '_').replace('-', '_')
    plt.savefig(f'performance_graph_{safe_name}.png')
    print(f"Graph saved to 'performance_graph_{safe_name}.png'")
    # Close plot to avoid interference
    plt.close()


# Save results
summary_df = pd.DataFrame(all_summary_data)
summary_df.to_csv('cv_results.csv', index=False)
print(f"\nDetailed results saved to 'cv_results.csv'")

# 7. Plotting
print("Generating plot...")
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 8))

# Plot F1 Score Comparison
sns.lineplot(data=summary_df, x='threshold', y='f1_mean', hue='model', linewidth=2.5)

# Find best points for annotation
for model_name in summary_df['model'].unique():
    model_data = summary_df[summary_df['model'] == model_name]
    if not model_data.empty:
        best_idx = model_data['f1_mean'].idxmax()
        best_row = model_data.loc[best_idx]
        
        plt.plot(best_row['threshold'], best_row['f1_mean'], 'o', markersize=10, 
                 label=f"Best {model_name} (T={best_row['threshold']:.2f}, F1={best_row['f1_mean']:.3f})")

plt.title('F1 Score Comparison: All-MPNet-Base-v2 vs. MPNet-Personality', fontsize=16)
plt.xlabel('Cosine Similarity Threshold', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.legend()
plt.tight_layout()

plt.savefig('performance_comparison.png')
print(f"Graph saved to 'performance_comparison.png'")

# 8. Best Settings Report
print("\n" + "="*40)
print(" BEST SETTINGS PER MODEL ")
print("="*40)
for model_name in summary_df['model'].unique():
    model_data = summary_df[summary_df['model'] == model_name]
    if not model_data.empty:
        best_idx = model_data['f1_mean'].idxmax()
        best_row = model_data.loc[best_idx]
        print(f"Model: {model_name}")
        print(f"  Optimal Threshold: {best_row['threshold']:.2f}")
        print(f"  Max F1 Score:      {best_row['f1_mean']:.4f}")
        print(f"  Precision:         {best_row['precision_mean']:.4f}")
        print(f"  Recall:            {best_row['recall_mean']:.4f}")
        print("-" * 40)
