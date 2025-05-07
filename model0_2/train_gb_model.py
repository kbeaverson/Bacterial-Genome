import numpy as np
from scipy import stats
import sklearn
from sklearn import ensemble
from sklearn import tree
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import KFold

def train_gb_model(X_train: np.array, y_train: np.array, performance_dict: dict, kfold: KFold, n_estimators_low: int, n_estimators_high: int, n_iter: int, cv: int, seed: int, k_best: int = 200):
    for i, (train_index, val_index) in enumerate(kfold.split(X_train, y_train)):

        print(f"Starting Outer fold {i}")
        X_train_outer, X_val_outer, y_train_outer, y_val_outer  = (
            X_train[train_index], X_train[val_index], y_train[train_index], y_train[val_index]
        )

        # Select k best features to reduce cost
        selector = SelectKBest(score_func=chi2, k=k_best)
        print(X_train_outer.shape)
        X_train_outer = selector.fit_transform(X_train_outer, y_train_outer)
        X_val_outer = selector.transform(X_val_outer)
        print("K Best selected.")
        print(X_train_outer.shape)

        gbc_random_cv = sklearn.model_selection.RandomizedSearchCV(
            estimator = ensemble.HistGradientBoostingClassifier(random_state=seed),
            param_distributions = {
                "learning_rate": stats.loguniform(0.001, 1),
                "max_iter": stats.randint(n_estimators_low, n_estimators_high),
                "l2_regularization": stats.loguniform(1e-5, 10)
            },
            cv = cv,
            scoring = sklearn.metrics.make_scorer(sklearn.metrics.balanced_accuracy_score),
            random_state = seed,
            n_jobs = -1,
        )

        # Fit the model
        print(f"Fitting model for fold {i}")
        gbc_random_cv.fit(X_train_outer, y_train_outer)

        # Assess the best model using the outer validation data
        print(f"Assessing model performance for fold {i}")
        y_pred_outer_gbc = gbc_random_cv.predict(X_val_outer)
        performance_dict[i] = sklearn.metrics.confusion_matrix(y_val_outer, y_pred_outer_gbc, labels=["S","R"])