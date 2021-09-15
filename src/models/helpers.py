"""Helper functions, not project specific."""
from typing import Any, Union
import logging
from time import time

import pandas as pd

from sklearn.base import is_classifier, ClassifierMixin
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# Hide warnings
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)


def find_best_params_classifier(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
    estimator: ClassifierMixin,
    params: dict[str, list[Union[str, float, int, bool]]] = {},
) -> dict[str, Any]:

    if not is_classifier(estimator):
        logging.error(f"{estimator} is not a classifier.")
        raise ValueError(f"{estimator} is not a classifier.")

    clf = HalvingRandomSearchCV(
        estimator=estimator,
        param_distributions=params,

        ## StratifiedKFold Cross Validator
        # StratifiedKFold permet de séparer les données en nombre de folds de manière stratifiée.
        # Les proportions des classes sont conservées.
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),

        ## F1 Score
        # F1 Score permet de mesurer la qualité d'un modèle en évaluant la précision et le
        # recall.
        scoring='f1',

        verbose=1,
        n_jobs=-1,
        random_state=42,
    ).fit(
        X=X_train,
        y=y_train,
    )

    start_time = time()
    y_pred = clf.predict(X_test)
    predict_time = time() - start_time

    if hasattr(clf, 'predict_proba'):
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, 'decision_function'):
        y_pred_proba = clf.decision_function(X_test)
    else:
        y_pred_proba = y_pred

    return {
        'model': clf.best_estimator_,
        'params': clf.best_params_,
        'score': clf.best_score_,
        'predict_time': predict_time,
        'cv_results_': clf.cv_results_,
        'best_index_': clf.best_index_,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'average_precision': average_precision_score(y_test, y_pred_proba),
        'precision_recall_curve': precision_recall_curve(y_test, y_pred_proba),
        'roc_auc_score': roc_auc_score(y_test, y_pred_proba),
        'roc_curve': roc_curve(y_test, y_pred_proba),
    }
