# Define seed for consistency checks
import datetime
import numpy as np
import pandas as pd
from scipy import stats
import sklearn
from sklearn import ensemble
from sklearn.feature_selection import SelectKBest, chi2
import sklearn.model_selection

from model0_1.calculate_normal_confidence_intervals import calculate_normal_confidence_intervals
from model0_1.load_data import load_data
from model0_2.train_gb_model import train_gb_model
from model0_2.write_gb_results_to_file import write_gb_results_to_file


seed = 73

# Define kfold split variables here
K = 5
n_iter = 50
cv = 10

# Define gb variables here
n_estimators_low = 1
n_estimators_high = 50
k_best = 2000

### 
# Load data
###
train_pa_genes, test_pa_genes, train_kmers, test_kmers, y_train, y_train_ids, y_test_ids, train_gene_alignment = load_data()
X_train_kmers, X_test_kmers = np.array(train_kmers), np.array(test_kmers)
X_train_pa, X_test_pa = np.array(train_pa_genes), np.array(test_pa_genes)
X_train_combined = np.hstack([X_train_pa,X_train_kmers])
y_train = y_train.reshape(-1)

### 
# Build manual K-fold loop for cross-validation 
###
kfold = sklearn.model_selection.StratifiedKFold(
    n_splits = K,
    shuffle = True,
    random_state = seed,
)

### Variable to track performance of each model type
kmer_model_performance = {}
pa_model_performance = {}
combined_model_performance = {}

# ### 
# # Iterate through each fold, training models for each on presence/absence
# ###
# pa_start = datetime.datetime.now()
# train_gb_model(
#     X_train = X_train_pa,
#     y_train = y_train,
#     performance_dict = pa_model_performance,
#     kfold = kfold,
#     n_estimators_low = n_estimators_low,
#     n_estimators_high = n_estimators_high,
#     n_iter = n_iter,
#     cv = cv,
#     seed = seed,
# )
# pa_end = datetime.datetime.now()

# combined_matrix_pa = np.mean(list(pa_model_performance.values()), axis = 0)
# pd.DataFrame(data = combined_matrix_pa, index = ["S", "R"], columns = ["S", "R"])
# print("Pres/Abs performance:")
# print(combined_matrix_pa)

# # Pres/Abs
# pa_acc_ci, pa_sens_ci, pa_spec_ci, pa_ba_ci = calculate_normal_confidence_intervals(list(pa_model_performance.values()))
# print(f"Pres/Abs balanced accuracy confidence interval: {pa_ba_ci}")

# write_gb_results_to_file(
#     file_path = "model0_2/results/gb_pa_results_strat_Hist.txt",
#     K = K,
#     cv = cv,
#     n_iter = n_iter,
#     n_estimators_low = n_estimators_low,
#     n_estimators_high = n_estimators_high,
#     ba_ci = pa_ba_ci,
#     runtime = pa_end - pa_start,
# )

# ### 
# # Iterate through each fold, training models for each on kmers
# ###
# kmer_start = datetime.datetime.now()
# train_gb_model(
#     X_train = X_train_kmers,
#     y_train = y_train,
#     performance_dict = kmer_model_performance,
#     kfold = kfold,
#     n_estimators_low = n_estimators_low,
#     n_estimators_high = n_estimators_high,
#     n_iter = n_iter,
#     cv = cv,
#     seed = seed,
#     k_best = k_best,
# )
# kmer_end = datetime.datetime.now()

### 
# Combine kmer and pres/abs into one model
###

# combined_start = datetime.datetime.now()
# train_gb_model(
#     X_train = X_train_combined,
#     y_train = y_train,
#     performance_dict = combined_model_performance,
#     kfold = kfold,
#     n_estimators_low = n_estimators_low,
#     n_estimators_high = n_estimators_high,
#     n_iter = n_iter,
#     cv = cv,
#     seed = seed,
#     k_best = k_best,
# )
# combined_end = datetime.datetime.now()

### 
# Combine all data across folds into one matrix
###
# combined_matrix_kmer = np.mean(list(kmer_model_performance.values()), axis = 0)
# pd.DataFrame(data = combined_matrix_kmer, index = ["S", "R"], columns = ["S", "R"])
# print("Kmer performance:")
# print(combined_matrix_kmer)

# combined_matrix_combined = np.mean(list(combined_model_performance.values()), axis = 0)
# pd.DataFrame(data = combined_matrix_combined, index = ["S", "R"], columns = ["S", "R"])
# print("Combined model performance:")
# print(combined_matrix_combined)

### 
# Calculate confidence intervals for models
###

# Kmer
# kmer_acc_ci, kmer_sens_ci, kmer_spec_ci, kmer_ba_ci = calculate_normal_confidence_intervals(list(kmer_model_performance.values()))
# print(f"Kmer balanced accuracy confidence interval: {kmer_ba_ci}")
# # Combined
# combined_acc_ci, combined_sens_ci, combined_spec_ci, combined_ba_ci = calculate_normal_confidence_intervals(list(combined_model_performance.values()))
# print(f"Combined model balanced accuracy confidence interval: {combined_ba_ci}")

### 
# Write results of test to file for future reference? 
###

# write_gb_results_to_file(
#     file_path = "model0_2/results/gb_kmer_results_strat_Hist.txt",
#     K = K,
#     cv = cv,
#     n_iter = n_iter,
#     n_estimators_low = n_estimators_low,
#     n_estimators_high = n_estimators_high,
#     ba_ci = kmer_ba_ci,
#     runtime = kmer_end - kmer_start,
# )

# write_gb_results_to_file(
#     file_path = "model0_2/results/gb_combined_results_stratified_weighted_Hist.txt",
#     K = K,
#     cv = cv,
#     n_iter = n_iter,
#     n_estimators_low = n_estimators_low,
#     n_estimators_high = n_estimators_high,
#     ba_ci = combined_ba_ci,
#     runtime = combined_end - combined_start,
# )

### 
# Train the final model
###

selector = SelectKBest(score_func = chi2, k = k_best)
X_train = selector.fit_transform(X_train_kmers, y_train)
X_test = selector.transform(X_test_kmers)

print("Training final model")
gb_final_cv = sklearn.model_selection.RandomizedSearchCV(
    estimator = ensemble.HistGradientBoostingClassifier(random_state=seed),
    param_distributions = {
        "learning_rate": stats.loguniform(0.001, 1),
        "max_iter": stats.randint(n_estimators_low, n_estimators_high),
        "l2_regularization": stats.loguniform(1e-5, 10)
    },
    cv = cv,
    scoring = sklearn.metrics.make_scorer(sklearn.metrics.balanced_accuracy_score),
    random_state = seed,
    n_jobs = 1,
    n_iter = 5,
    verbose = 1,
)

print("Fitting final cv model")
# Fit the final model
gb_final_cv.fit(X_train, y_train)

print("Getting optimal parameters")
# Get optimal parameters
final_lr = gb_final_cv.best_estimator_.get_params()["learning_rate"]
final_max_iter = gb_final_cv.best_estimator_.get_params()["max_iter"]
final_l2_regularization = gb_final_cv.best_estimator_.get_params()["l2_regularization"]

print(f"Final learning rate: {final_lr}")
print(f"Final max iterations: {final_max_iter}")
print(f"Final l2 regularization: {final_l2_regularization}")

print("Building final model")
# Build final model
final_model = ensemble.HistGradientBoostingClassifier(
    learning_rate = final_lr,
    max_iter = final_max_iter,
    l2_regularization = final_l2_regularization,
)

print("Fitting final model")
final_model.fit(X_train, y_train)
print("Predicting")
y_pred_test = final_model.predict(X_test)
final_predictions = pd.DataFrame(data = {"genome_id":y_test_ids, "y_pred":y_pred_test})
final_predictions.to_csv('kmer_gb_hist_stratified.csv', index = False)