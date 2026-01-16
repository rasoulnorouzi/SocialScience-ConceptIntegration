import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import torch

def setup_reproducibility(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

def load_and_prepare_data(pos_path, neg_path, seed=42):
    print("Loading datasets...")
    try:
        pos_df = pd.read_csv(pos_path, on_bad_lines='skip')
        neg_df = pd.read_csv(neg_path, on_bad_lines='skip')
    except Exception as e:
        print(f"Error loading files: {e}")
        raise e

    print(f"Positive samples: {len(pos_df)}")
    print(f"Negative samples (total): {len(neg_df)}")

    n_pos = len(pos_df)
    if len(neg_df) > n_pos:
        neg_df_sampled = neg_df.sample(n=n_pos, random_state=seed)
    else:
        neg_df_sampled = neg_df
    print(f"Negative samples (sampled): {len(neg_df_sampled)}")

    df = pd.concat([pos_df, neg_df_sampled])
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df['term1'] = df['term1'].astype(str)
    df['term2'] = df['term2'].astype(str)
    
    return df

def load_model(model_name, model_type='sentence_transformer'):
    """
    Load a Sentence Transformer model based on its type.
    
    Args:
        model_name (str): The name/path of the model to load.
        model_type (str): The type of model ('sentence_transformer' or 'token_embedding_mean_pool').
    """
    print(f"Loading Model ({model_name})...")
    try:
        if model_type == 'token_embedding_mean_pool':
            word_embedding_model = models.Transformer(model_name, model_args={'local_files_only': True})
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        else:
            model = SentenceTransformer(model_name, model_kwargs={'local_files_only': True})
        return model
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None

def compute_embeddings_and_similarity(model, df, batch_size=16):
    print("Encoding sentences...")
    embeddings1 = model.encode(df['term1'].tolist(), convert_to_tensor=True, show_progress_bar=False, batch_size=batch_size)
    embeddings2 = model.encode(df['term2'].tolist(), convert_to_tensor=True, show_progress_bar=False, batch_size=batch_size)

    print("Computing Cosine Similarities...")
    similarities = torch.nn.functional.cosine_similarity(embeddings1, embeddings2).cpu().numpy()
    return similarities

def evaluate_model(model_name, similarities, labels, thresholds=np.arange(0.10, 1.00, 0.01)):
    print(f"\n--- Results per Threshold ({model_name}) ---")
    print(f"{'Threshold':<10} | {'F1':<10} | {'Precision':<10} | {'Recall':<10} | {'PosAcc':<10} | {'NegAcc':<10}")
    print("-" * 80)

    current_model_results = []
    best_f1_model = -1
    best_threshold_model = -1
    
    for t in thresholds:
        preds = (similarities > t).astype(int)
        
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        
        pos_acc = np.mean(preds[pos_mask] == 1) if np.any(pos_mask) else 0.0
        neg_acc = np.mean(preds[neg_mask] == 0) if np.any(neg_mask) else 0.0

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
        current_model_results.append(result_entry)
        
        print(f"{t:.2f}       | {f1:.4f}     | {p:.4f}     | {r:.4f}     | {pos_acc:.4f}     | {neg_acc:.4f}")
        
    return current_model_results, best_threshold_model

def plot_individual_performance(model_results, model_name, best_threshold_model, display_name=None):
    if display_name is None:
        display_name = model_name

    print(f"Generating plot for {display_name}...")
    current_df = pd.DataFrame(model_results)
    
    # Set a nicer theme
    sns.set_theme(style="whitegrid", context="notebook", palette="deep")
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=current_df, x='threshold', y='f1_mean', label='F1 Score', linewidth=2.5, color='#e74c3c')
    sns.lineplot(data=current_df, x='threshold', y='precision_mean', label='Precision', linestyle='--', color='#2980b9')
    sns.lineplot(data=current_df, x='threshold', y='recall_mean', label='Recall', linestyle='--', color='#2ecc71')
    
    # Annotate best threshold without crowding the legend
    plt.axvline(x=best_threshold_model, color='#34495e', linestyle=':', alpha=0.9)
    # Find max F1 for annotation position
    max_f1 = current_df.loc[current_df['threshold'] == best_threshold_model, 'f1_mean'].values[0]
    
    plt.annotate(f'Best F1: {max_f1:.3f}\nThresh: {best_threshold_model:.2f}', 
                 xy=(best_threshold_model, max_f1), 
                 xytext=(10, 20), textcoords='offset points',
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    plt.title(f'Performance Analysis: {display_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Cosine Similarity Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    
    # Adjusted legend to be smaller and outside if needed, or just smaller inside
    plt.legend(loc='upper right', frameon=True, fontsize='small')
    plt.tight_layout()
    
    # Save results
    plots_dir = 'results/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    safe_name = model_name.replace('/', '_').replace('-', '_')
    save_path = os.path.join(plots_dir, f'performance_graph_{safe_name}.svg')
    
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    print(f"Graph saved to '{save_path}'")
    plt.show() # Inline display

def plot_comparison(summary_df, model_display_names=None):
    print("Generating comparison plot...")
    
    # Enhance visualization style
    sns.set_theme(style="whitegrid", context="talk", palette="turbo")
    
    plt.figure(figsize=(14, 9))
    
    ax = sns.lineplot(data=summary_df, x='threshold', y='f1_mean', hue='model', linewidth=2.5,  style='model', dashes=False)

    # Annotate best points directly on the graph
    colors = sns.color_palette("turbo", n_colors=len(summary_df['model'].unique()))
    
    for i, model_name in enumerate(summary_df['model'].unique()):
        model_data = summary_df[summary_df['model'] == model_name]
        
        # Use friendly name if available
        display_label = model_display_names.get(model_name, model_name) if model_display_names else model_name
        
        if not model_data.empty:
            best_idx = model_data['f1_mean'].idxmax()
            best_row = model_data.loc[best_idx]
            
            # Simple marker for the max point
            plt.plot(best_row['threshold'], best_row['f1_mean'], 'o', markersize=8, color='black', alpha=0.5)
            
            # Smart annotation placement
            y_offset = 10 if i % 2 == 0 else -25  # Stagger labels up/down
            plt.annotate(
                f"{display_label}\n(F1={best_row['f1_mean']:.2f} @ {best_row['threshold']:.2f})",
                xy=(best_row['threshold'], best_row['f1_mean']),
                xytext=(0, y_offset),
                textcoords='offset points',
                ha='center',
                fontsize=9,
                color='black',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6)
            )

    plt.title('F1 Score Comparison Across Models', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Cosine Similarity Threshold', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    
    # Improve legend - Move to bottom to avoid squeezing width
    handles, labels = ax.get_legend_handles_labels()
    # Map raw model names to display names for legend
    if model_display_names:
        labels = [model_display_names.get(l, l) for l in labels]
        
    plt.legend(handles, labels, title="Models", loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
    
    plt.tight_layout()
    
    plots_dir = 'results/plots'
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, 'performance_comparison.svg')
    
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    print(f"Graph saved to '{save_path}'")
    plt.show() # Inline display

def print_best_settings(summary_df):
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
