# ============================================================
# Models with GridSearchCV compatible with sklearn Pipeline
# take in mind that this gridserach is temporal! not iid.
# ============================================================

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from sklearn.metrics import (
    make_scorer,
    confusion_matrix,
    f1_score,
    accuracy_score,
    recall_score
)


def help():
    print("Models with grid search and stratified K-folds...")


# ============================================================
# Support Vector Machine
# ============================================================

def grid_SVC(
    X_train,
    y_train,
    performance_metric="f1",
    resultsGrid=False,
    return_grid=False
):
    model = SVC(probability=True)

    C = np.linspace(0.000001, 1000, 5)
    kernels = ["poly", "rbf"]
    gamma = ["scale"]

    grid = {
        "C": C,
        "kernel": kernels,
        "gamma": gamma
    }

    cv = TimeSeriesSplit(
    n_splits=5)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        n_jobs=-1,
        cv=cv,
        scoring=performance_metric,
        error_score="raise"
    )

    grid_result = grid_search.fit(X_train, y_train)

    if resultsGrid:
        return grid_result.cv_results_

    if return_grid:
        return grid_result

    return grid_result.best_estimator_


class GridSearchSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, performance_metric="f1"):
        self.performance_metric = performance_metric

    def fit(self, X, y):
        self.grid_ = grid_SVC(
            X,
            y,
            performance_metric=self.performance_metric,
            return_grid=True
        )
        self.model_ = self.grid_.best_estimator_
        self.classes_ = self.model_.classes_
        return self

    def hypers(self):
        check_is_fitted(self, "grid_")
        return self.grid_.best_params_

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)


# ============================================================
# K-Nearest Neighbors
# ============================================================

def grid_KNN(
    X_train,
    y_train,
    performance_metric="f1",
    resultsGrid=False,
    return_grid=False
):
    model = KNeighborsClassifier()

    n_neighbors = np.arange(1, 10, 1)
    weights = ["uniform", "distance"]

    grid = {
        "n_neighbors": n_neighbors,
        "weights": weights
    }

    cv = TimeSeriesSplit(
    n_splits=5)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        n_jobs=-1,
        cv=cv,
        scoring=performance_metric,
        error_score="raise"
    )

    grid_result = grid_search.fit(X_train, y_train)

    if resultsGrid:
        return grid_result.cv_results_

    if return_grid:
        return grid_result

    return grid_result.best_estimator_


class GridSearchKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, performance_metric="f1"):
        self.performance_metric = performance_metric

    def fit(self, X, y):
        self.grid_ = grid_KNN(
            X,
            y,
            performance_metric=self.performance_metric,
            return_grid=True
        )
        self.model_ = self.grid_.best_estimator_
        self.classes_ = self.model_.classes_
        return self

    def hypers(self):
        check_is_fitted(self, "grid_")
        return self.grid_.best_params_

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)


# ============================================================
# Logistic Regression
# ============================================================

def grid_lr(
    X_train,
    y_train,
    performance_metric="roc_auc",
    resultsGrid=False,
    return_grid=False
):
    model = LogisticRegression(
        random_state=666,
        max_iter=10000
    )

    solvers = ["liblinear"]
    penalty = ["l2", "l1"]
    c_values = [100, 10, 1.0, 0.1, 0.01, 0.001, 0.0001]

    grid = {
        "solver": solvers,
        "penalty": penalty,
        "C": c_values
    }

    cv = TimeSeriesSplit(
    n_splits=5)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        n_jobs=-1,
        cv=cv,
        scoring=performance_metric,
        error_score="raise"
    )

    grid_result = grid_search.fit(X_train, y_train)

    if resultsGrid:
        return grid_result.cv_results_

    if return_grid:
        return grid_result

    return grid_result.best_estimator_


class GridSearchLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, performance_metric="roc_auc"):
        self.performance_metric = performance_metric

    def fit(self, X, y):
        self.grid_ = grid_lr(
            X,
            y,
            performance_metric=self.performance_metric,
            return_grid=True
        )
        self.model_ = self.grid_.best_estimator_
        self.classes_ = self.model_.classes_
        return self

    def hypers(self):
        check_is_fitted(self, "grid_")
        return self.grid_.best_params_

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)


# ============================================================
# Random Forest
# ============================================================

def grid_RandomForest(
    X_train,
    y_train,
    performance_metric="roc_auc",
    resultsGrid=False,
    return_grid=False
):
    model = RandomForestClassifier(random_state=0)

    grid = dict(
        n_estimators=[1000],
        max_depth=[7, 12, None],
        min_samples_leaf=[15, 20, 30, ],
        min_samples_split=[10, 20],
        max_features=["sqrt", "log2", 0.2, 0.3,
        bootstrap=[True],
        class_weight=[None, "balanced", "balanced_subsample"],
        criterion=["gini", "entropy"]
    )

    cv = TimeSeriesSplit(
    n_splits=5)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        n_jobs=-1,
        cv=cv,
        scoring=performance_metric,
        error_score="raise"
    )

    grid_result = grid_search.fit(X_train, y_train)

    if resultsGrid:
        return grid_result.cv_results_

    if return_grid:
        return grid_result

    return grid_result.best_estimator_


class GridSearchRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, performance_metric="roc_auc"):
        self.performance_metric = performance_metric

    def fit(self, X, y):
        self.grid_ = grid_RandomForest(
            X,
            y,
            performance_metric=self.performance_metric,
            return_grid=True
        )
        self.model_ = self.grid_.best_estimator_
        self.classes_ = self.model_.classes_
        return self

    def hypers(self):
        check_is_fitted(self, "grid_")
        return self.grid_.best_params_

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)


# ============================================================
# AdaBoost
# ============================================================

def grid_Adaboost(
    X_train,
    y_train,
    performance_metric="roc_auc",
    resultsGrid=False,
    return_grid=False
):
    model = AdaBoostClassifier(random_state=1)

    n_estimators = [2, 15, 35, 50, 70, 100]
    learning_rate = np.linspace(0.01, 1, 10)

    grid = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate
    }

    cv = TimeSeriesSplit(
    n_splits=5)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        n_jobs=-1,
        cv=cv,
        scoring=performance_metric,
        error_score="raise"
    )

    grid_result = grid_search.fit(X_train, y_train)

    if resultsGrid:
        return grid_result.cv_results_

    if return_grid:
        return grid_result

    return grid_result.best_estimator_



# ============================================================
# Specificity
# ============================================================

def especificityF(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


especificity = make_scorer(
    especificityF,
    greater_is_better=True
)


# ============================================================
# Plot tools for GridSearchCV results
# ============================================================

def meet(fix_parameters, array):
    counter = 0

    for param in fix_parameters:
        if array[param] != fix_parameters[param]:
            counter += 1

    return not bool(counter)


def meet_indexes(results_, fix_parameters):
    indexes = []

    for ith, array in enumerate(results_["params"]):
        if meet(fix_parameters, array):
            indexes.append(ith)

    return indexes


def Plot_parameter_score(results_, fix_parameters, x_axis):
    indices = meet_indexes(results_, fix_parameters)

    X = [
        i[x_axis]
        for i in np.array(results_["params"])[indices]
    ]

    y = results_["mean_test_score"][indices]

    return X, y


# ============================================================
# Manual repeated stratified K-fold scores
# ============================================================

def mean_scores_kfold(classifier, X_, y_):
    X = X_.reset_index(drop=True)
    y = y_.reset_index(drop=True)

    X = X.to_numpy()

    rskf = RepeatedStratifiedKFold(
        n_splits=10,
        n_repeats=3,
        random_state=1
    )

    accuracys = []
    recalls = []
    f1s = []

    for itrain, itest in rskf.split(X, y):
        classifier.fit(X[itrain], y[itrain])
        preds = classifier.predict(X[itest])

        accuracys.append(accuracy_score(y[itest], preds))
        recalls.append(recall_score(y[itest], preds))
        f1s.append(f1_score(y[itest], preds))

    return (
        np.array(accuracys).mean(),
        np.array(recalls).mean(),
        np.array(f1s).mean()
    )
