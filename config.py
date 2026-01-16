import numpy as np

# Random Seed
SEED = 42

# Data Paths
DATA_PATHS = {
    'pos_pairs': 'datasets/processed_datasets/train_positive_pairs.csv',
    'neg_pairs': 'datasets/processed_datasets/train_negative_pairs.csv',
    'results_dir': 'results',
    'plots_dir': 'results/plots',
    'results_csv': 'results/cv_results.csv'
}

# Model Types Enum-like constants
TYPE_SENTENCE = 'sentence_transformer'
TYPE_TOKEN = 'token_embedding_mean_pool'

# Models Configuration
# You can add more models here.
# 'type': Defines how the model is loaded/pooled.
# 'display_name': Used for plotting legends instead of the raw huggingface ID.
MODELS = {
    'all-mpnet-base-v2': {
        'type': TYPE_SENTENCE,
        'display_name': 'All-MPNet-Base-v2'
    },
    'dwulff/mpnet-personality': {
        'type': TYPE_SENTENCE,
        'display_name': 'MPNet-Personality'
    },
    'allenai/scibert_scivocab_uncased': {
        'type': TYPE_TOKEN,
        'display_name': 'SciBERT (SciVocab)'
    },
    'bert-base-uncased': {
        'type': TYPE_TOKEN,
        'display_name': 'BERT Base'
    }
}

# Analysis Parameters
# Thresholds: Range of cosine similarity thresholds to test
THRESHOLDS = np.arange(0.10, 1.00, 0.01)

# Batch size for encoding
BATCH_SIZE = 16
