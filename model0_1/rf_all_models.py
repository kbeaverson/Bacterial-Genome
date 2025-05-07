# Define seed for consistency checks
import datetime
import numpy as np
import pandas as pd
from scipy import stats
import sklearn
from sklearn import ensemble

from model0_1.calculate_normal_confidence_intervals import calculate_normal_confidence_intervals
from model0_1.load_data import load_data
from model0_1.train_rf_model import train_rf_model
from model0_1.write_rf_results_to_file import write_rf_results_to_file


seed = 73

# Define kfold split variables here
K = 5
n_iter = 20
cv = 10

# Define random forest variables here
n_estimators_low = 1
n_estimators_high = 50
max_depth_low = 1
max_depth_high = 10

### 
# Load data
###
train_pa_genes, test_pa_genes, train_kmers, test_kmers, y_train, y_train_ids, y_test_ids, train_gene_alignment = load_data()
X_train_kmers, X_test_kmers = np.array(train_kmers), np.array(test_kmers)
X_train_pa, X_test_pa = np.array(train_pa_genes), np.array(test_pa_genes)
X_train_combined = np.hstack([X_train_pa,X_train_kmers])
X_test_combined = np.hstack([X_test_pa, X_test_kmers])
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
# train_rf_model(
#     X_train = X_train_pa,
#     y_train = y_train,
#     performance_dict = pa_model_performance,
#     kfold = kfold,
#     seed = seed,
#     n_estimators_low = n_estimators_low,
#     n_estimators_high = n_estimators_high,
#     max_depth_low = max_depth_low,
#     max_depth_high = max_depth_high,
#     n_iter = n_iter,
#     cv = cv,
# )
# pa_end = datetime.datetime.now()

# # ### 
# # # Iterate through each fold, training models for each on kmers
# # ###
# kmer_start = datetime.datetime.now()
# train_rf_model(
#     X_train = X_train_kmers,
#     y_train = y_train,
#     performance_dict = kmer_model_performance,
#     kfold = kfold,
#     seed = seed,
#     n_estimators_low = n_estimators_low,
#     n_estimators_high = n_estimators_high,
#     max_depth_low = max_depth_low,
#     max_depth_high = max_depth_high,
#     n_iter = n_iter,
#     cv = cv,
# )
# kmer_end = datetime.datetime.now()

### 
# Combine kmer and pres/abs into one model
###



### 
# Combine all data across folds into one matrix
###
# combined_matrix_kmer = np.mean(list(kmer_model_performance.values()), axis = 0)
# pd.DataFrame(data = combined_matrix_kmer, index = ["S", "R"], columns = ["S", "R"])
# print("Kmer performance:")
# print(combined_matrix_kmer)


# combined_matrix_pa = np.mean(list(pa_model_performance.values()), axis = 0)
# pd.DataFrame(data = combined_matrix_pa, index = ["S", "R"], columns = ["S", "R"])
# print("Pres/Abs performance:")
# print(combined_matrix_pa)

### 
# Calculate confidence intervals for models
###

# # Pres/Abs
# pa_acc_ci, pa_sens_ci, pa_spec_ci, pa_ba_ci = calculate_normal_confidence_intervals(list(pa_model_performance.values()))
# print(f"Pres/Abs balanced accuracy confidence interval: {pa_ba_ci}")
# # Kmer
# kmer_acc_ci, kmer_sens_ci, kmer_spec_ci, kmer_ba_ci = calculate_normal_confidence_intervals(list(kmer_model_performance.values()))
# print(f"Kmer balanced accuracy confidence interval: {kmer_ba_ci}")

# combined_start = datetime.datetime.now()
# train_rf_model(
#     X_train = X_train_combined,
#     y_train = y_train,
#     performance_dict = combined_model_performance,
#     kfold = kfold,
#     seed = seed,
#     n_estimators_low = n_estimators_low,
#     n_estimators_high = n_estimators_high,
#     max_depth_low = max_depth_low,
#     max_depth_high = max_depth_high,
#     n_iter = n_iter,
#     cv = cv,
# )
# combined_end = datetime.datetime.now()

# combined_matrix_combined = np.mean(list(combined_model_performance.values()), axis = 0)
# pd.DataFrame(data = combined_matrix_combined, index = ["S", "R"], columns = ["S", "R"])
# print("Combined model performance:")
# print(combined_matrix_combined)

# # Combined
# combined_acc_ci, combined_sens_ci, combined_spec_ci, combined_ba_ci = calculate_normal_confidence_intervals(list(combined_model_performance.values()))
# print(f"Combined model balanced accuracy confidence interval: {combined_ba_ci}")

# write_rf_results_to_file(
#     file_path = "model0_1/results/rf_combined_results_stratified.txt",
#     K = K,
#     cv = cv,
#     n_iter = n_iter,
#     max_depth_low = max_depth_low,
#     max_depth_high = max_depth_high,
#     n_estimators_low = n_estimators_low,
#     n_estimators_high = n_estimators_high,
#     ba_ci = combined_ba_ci,
#     runtime = combined_end - combined_start,
# )

### 
# Write results of test to file for future reference? 
###

# write_rf_results_to_file(
#     file_path = "model0_1/results/rf_pa_results.txt",
#     K = K,
#     cv = cv,
#     n_iter = n_iter,
#     max_depth_low = max_depth_low,
#     max_depth_high = max_depth_high,
#     n_estimators_low = n_estimators_low,
#     n_estimators_high = n_estimators_high,
#     ba_ci = pa_ba_ci,
#     runtime = pa_end - pa_start,
# )

# write_rf_results_to_file(
#     file_path = "model0_1/results/rf_kmer_results.txt",
#     K = K,
#     cv = cv,
#     n_iter = n_iter,
#     max_depth_low = max_depth_low,
#     max_depth_high = max_depth_high,
#     n_estimators_low = n_estimators_low,
#     n_estimators_high = n_estimators_high,
#     ba_ci = kmer_ba_ci,
#     runtime = kmer_end - kmer_start,
# )

### 
# Train the final model
###
X_train = X_train_combined
X_test = X_test_combined

print("Training final model")
rf_final_cv = sklearn.model_selection.RandomizedSearchCV(
    estimator = ensemble.RandomForestClassifier(random_state=seed, class_weight='balanced'),
    param_distributions = {
        "n_estimators": stats.randint(low = n_estimators_low, high = n_estimators_high),
        "max_depth": stats.randint(low = max_depth_low, high = max_depth_high),
    },
    n_iter = n_iter,
    cv = cv,
    scoring = sklearn.metrics.make_scorer(sklearn.metrics.balanced_accuracy_score)
)

print("Fitting final cv model")
# Fit the final model
rf_final_cv.fit(X_train, y_train)

print("Getting optimal parameters")
# Get optimal parameters
final_n_estimators = rf_final_cv.best_estimator_.get_params()["n_estimators"]
final_max_depth = rf_final_cv.best_estimator_.get_params()["max_depth"]

print(f"Final n estimators: {final_n_estimators}")
print(f"Final max depth: {final_max_depth}")

print("Building final model")
# Build final model
final_model = ensemble.RandomForestClassifier(
    random_state = seed,
    class_weight = 'balanced',
    n_estimators = final_n_estimators,
    max_depth = final_max_depth,
)
print("Fitting final model")
final_model.fit(X_train, y_train)
print("Predicting")
y_pred_test = final_model.predict(X_test)
final_predictions = pd.DataFrame(data = {"genome_id":y_test_ids, "y_pred":y_pred_test})
final_predictions.to_csv('combined_rf_stratified.csv', index = False)