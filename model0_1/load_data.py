import numpy as np
import pandas as pd

def load_data():
    """
    Load the data needed for model training
    """
    # Presence absence features
    train_pa_genes = pd.read_csv('data/train_test_data/train_pa_genes.csv').set_index('genome_id')
    test_pa_genes = pd.read_csv('data/train_test_data/test_pa_genes.csv').set_index('genome_id')
    
    # Load Kmer data
    train_kmers = np.load('data/train_test_data/train_kmers.npy', allow_pickle=True)
    test_kmers = np.load('data/train_test_data/test_kmers.npy', allow_pickle=True)

    # Load target data & IDs
    y_train = np.load('data/train_test_data/y_train.npy', allow_pickle=True)
    y_train_ids = np.load('data/train_test_data/train_ids.npy', allow_pickle=True).astype(str)
    y_test_ids = np.load('data/train_test_data/test_ids.npy', allow_pickle=True).astype(str)

    # Load raw gene data for optional neural network section
    train_gene_alignment = pd.read_csv('data/train_test_data/train_genes.csv')
    
    return train_pa_genes, test_pa_genes, train_kmers, test_kmers, y_train, y_train_ids, y_test_ids, train_gene_alignment