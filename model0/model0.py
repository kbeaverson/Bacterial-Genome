import os
from collections import defaultdict

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
import sklearn.metrics
import sklearn.model_selection
from tensorflow import keras
from scipy import stats

import datetime

# Define seed for consistency checks
seed = 73

# Define kfold split variables here
K = 5
n_iter = 10
cv = 10


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

train_pa_genes, test_pa_genes, train_kmers, test_kmers, y_train, y_train_ids, y_test_ids, train_gene_alignment = load_data()
X_train_kmers, X_test_kmers = np.array(train_kmers), np.array(test_kmers)
X_train_pa, X_test_pa = np.array(train_pa_genes), np.array(test_pa_genes)
y_train = y_train.reshape(-1)

print(y_train[0:10])

### 
# Build manual K-fold loop for cross-validation 
###
kfold = sklearn.model_selection.KFold(
    n_splits = K,
    shuffle = True,
    random_state = seed,
)

### Variable to track performance of each model type
kmer_model_performance = {}
pa_model_performance = {}

### 
# Iterate through each fold, training models for each on presence/absence
###
pa_start = datetime.datetime.now()
for i, (train_index, val_index) in enumerate(kfold.split(X_train_pa)):
    print(f"Starting outer fold {i}")

    # Grab the data for this fold
    X_train_outer, X_val_outer, y_train_outer, y_val_outer = (X_train_pa[train_index], X_train_pa[val_index], y_train[train_index], y_train[val_index])

    print("Creating pres/abs random forest model")
    # Random forest model creation
    pa_rf_random_cv = sklearn.model_selection.RandomizedSearchCV(
        estimator = ensemble.RandomForestClassifier(random_state=seed),
        param_distributions = {
            "n_estimators": stats.randint(low = 1, high = 20),
            "max_depth": stats.randint(low = 1, high = 10),
        },
        n_iter = n_iter,
        cv = cv,
        scoring = sklearn.metrics.make_scorer(sklearn.metrics.balanced_accuracy_score)
    )

    # Random forest model fit
    print("Fitting random forest")
    pa_rf_random_cv.fit(X_train_outer, y_train_outer)

    # Assess model performance
    print("Assessing random forest model performance")
    y_pred_outer_rf = pa_rf_random_cv.predict(X_val_outer)
    pa_model_performance[i] = sklearn.metrics.confusion_matrix(y_val_outer, y_pred_outer_rf, labels = ["S", "R"])
pa_end = datetime.datetime.now()

### 
# Iterate through each fold, training models for each on kmers
###
kmer_start = datetime.datetime.now()
for i, (train_index, val_index) in enumerate(kfold.split(X_train_kmers)):

    print(f"Starting outer fold {i}")

    # Grab the data for this fold
    X_train_outer, X_val_outer, y_train_outer, y_val_outer = (X_train_kmers[train_index], X_train_kmers[val_index], y_train[train_index], y_train[val_index])

    # kmer random forest model creation
    print("Creating kmer model")
    kmer_rf_random_cv = sklearn.model_selection.RandomizedSearchCV(
        estimator = ensemble.RandomForestClassifier(random_state=seed),
        param_distributions = {
            "n_estimators": stats.randint(low = 1, high = 20),
            "max_depth": stats.randint(low = 1, high = 10),
        },
        n_iter = n_iter,
        cv = cv,
        scoring = sklearn.metrics.make_scorer(sklearn.metrics.balanced_accuracy_score)
    )

    # Random forest model fit
    print("Fitting kmer random forest")
    kmer_rf_random_cv.fit(X_train_outer, y_train_outer)

    # Assess model performance
    print("Assessing kmer random forest model performance")
    y_pred_outer_rf = kmer_rf_random_cv.predict(X_val_outer)
    kmer_model_performance[i] = sklearn.metrics.confusion_matrix(y_val_outer, y_pred_outer_rf, labels = ["S", "R"]) 
kmer_end = datetime.datetime.now()

### 
# Combine all data across folds into one matrix
###
combined_matrix_kmer = np.mean(list(kmer_model_performance.values()), axis = 0)
pd.DataFrame(data = combined_matrix_kmer, index = ["S", "R"], columns = ["S", "R"])
print("Kmer performance:")
print(combined_matrix_kmer)


combined_matrix_pa = np.mean(list(pa_model_performance.values()), axis = 0)
pd.DataFrame(data = combined_matrix_pa, index = ["S", "R"], columns = ["S", "R"])
print("Pres/Abs performance:")
print(combined_matrix_pa)

### 
# Calculate confidence intervals for models
###
def calculate_normal_confidence_intervals(confusion_matrices_list):
    accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    bal_acc_list = []

    for confusion_matrix in confusion_matrices_list:
        sensitivity = confusion_matrix[1,1] / confusion_matrix[1,:].sum()
        specificity = confusion_matrix[0,0] / confusion_matrix[0,:].sum()

        accuracy_list.append((confusion_matrix[0,0] + confusion_matrix[1,1]) / confusion_matrix.sum())
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        bal_acc_list.append(np.mean([sensitivity, specificity]))

    location, std = np.mean(accuracy_list), np.std(accuracy_list)
    accuracy_ci = stats.norm.interval(0.95, loc = location, scale = std)

    location, std = np.mean(sensitivity_list), np.std(sensitivity_list)
    sensitivity_ci = stats.norm.interval(0.95, loc = location, scale = std)

    location, std = np.mean(specificity_list), np.std(specificity_list)
    specificity_ci = stats.norm.interval(0.95, loc = location, scale = std)

    location, std = np.mean(bal_acc_list), np.std(bal_acc_list)
    bal_acc_ci = stats.norm.interval(0.95, loc = location, scale = std)

    return accuracy_ci, sensitivity_ci, specificity_ci, bal_acc_ci

# Kmer
kmer_acc_ci, kmer_sens_ci, kmer_spec_ci, kmer_ba_ci = calculate_normal_confidence_intervals(list(kmer_model_performance.values()))
print(f"Kmer balanced accuracy confidence interval: {kmer_ba_ci}")
# Pres/Abs
pa_acc_ci, pa_sens_ci, pa_spec_ci, pa_ba_ci = calculate_normal_confidence_intervals(list(pa_model_performance.values()))
print(f"Pres/Abs balanced accuracy confidence interval: {pa_ba_ci}")

### 
# Write results of test to file for future reference? 
###
file_path = "model0/m0_results.txt"
with open(file_path, "a") as file: 
    file.write(f"Number of outer folds: {K}\n")
    file.write(f"Number of inner folds: {cv}\n")
    file.write(f"Number of iterations: {n_iter}\n")
    file.write(f"Pres/Abs balanced accuracy confidence interval: {pa_ba_ci}\n")
    file.write(f"Pres/Abs runtime: {pa_end - pa_start}\n")
    file.write(f"Kmer balanced accuracy confidence interval: {kmer_ba_ci}\n")
    file.write(f"Pres/Abs runtime: {kmer_end - kmer_start}\n")
    file.write("========\n\n")

print("Text appended to output file.")