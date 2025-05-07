import numpy as np
from scipy import stats
import sklearn
from sklearn import ensemble
from sklearn.model_selection import KFold

def train_rf_model(X_train: np.array, y_train: np.array, performance_dict: dict, kfold: KFold, seed: int, n_estimators_low: int, n_estimators_high: int, max_depth_low: int, max_depth_high: int, n_iter: int, cv: int):
    for i, (train_index, val_index) in enumerate(kfold.split(X_train, y_train)):
        print(f"Starting outer fold {i}")

        # Grab the data for this fold
        X_train_outer, X_val_outer, y_train_outer, y_val_outer = (X_train[train_index], X_train[val_index], y_train[train_index], y_train[val_index])

        # Random forest model creation
        pa_rf_random_cv = sklearn.model_selection.RandomizedSearchCV(
            estimator = ensemble.RandomForestClassifier(random_state=seed, class_weight='balanced'),
            param_distributions = {
                "n_estimators": stats.randint(low = n_estimators_low, high = n_estimators_high),
                "max_depth": stats.randint(low = max_depth_low, high = max_depth_high),
            },
            n_iter = n_iter,
            cv = cv,
            scoring = sklearn.metrics.make_scorer(sklearn.metrics.balanced_accuracy_score)
        )

        # Random forest model fit
        pa_rf_random_cv.fit(X_train_outer, y_train_outer)

        # Assess model performance
        y_pred_outer_rf = pa_rf_random_cv.predict(X_val_outer)
        performance_dict[i] = sklearn.metrics.confusion_matrix(y_val_outer, y_pred_outer_rf, labels = ["S", "R"])