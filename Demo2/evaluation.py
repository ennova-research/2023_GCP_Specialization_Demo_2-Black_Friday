from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score
from sklearn.model_selection import cross_val_predict, cross_val_score


def evaluate(model, X, y, model_type='classification', metric=None, cross_val=False, classification_threshold=None):
    """
    Evaluate the performance of a machine learning model.

    Parameters:
    - model: The trained machine learning model.
    - X: Input features for evaluation.
    - y: True labels for evaluation.
    - model_type: Type of the model, either 'regression' or 'classification'.
    - metric: The evaluation metric to be used.
    - cross_val: Whether to use cross-validation for evaluation.
    - classification_threshold: Threshold for classification models.

    Returns:
    - The evaluation score based on the specified metric.
    """
    assert(model_type in ["regression", "classification"])

    if model_type == "regression":
        if not metric:
            metric = r2_score
        score = evaluate_regression(model, X, y, metric, cross_val)
    else:
        if not metric:
            metric = f1_score
        score = evaluate_classification(model, X, y, metric, classification_threshold, cross_val)
        
    return score


def evaluate_classification(model, X, y, metric, threshold, cross_val=False):
    """
    Evaluate classification model performance.

    Parameters:
    - model: The trained classification model.
    - X: Input features for evaluation.
    - y: True labels for evaluation.
    - metric: The evaluation metric to be used.
    - threshold: Threshold for classification.
    - cross_val: Whether to use cross-validation for evaluation.

    Returns:
    - The evaluation score based on the specified metric.
    """
    if cross_val:
        if threshold is not None:
            # Use cross_val_predict for evaluation with the specified threshold
            y_probs = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]
            y_preds = (y_probs >= threshold).astype(int)
            score = metric(y, y_preds)
        else:
            score = cross_val_score(model, X, y, cv=5, scoring=metric.__name__.split('_')[0])
    else:
        if threshold is None:
            y_pred = model.predict(X)
        else:
            y_prob = model.predict_proba(X)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)
        score = metric(y, y_pred)
        
    return score


def evaluate_regression(model, X, y, metric, cross_val=False):
    """
    Evaluate regression model performance.

    Parameters:
    - model: The trained regression model.
    - X: Input features for evaluation.
    - y: True labels for evaluation.
    - metric: The evaluation metric to be used.
    - cross_val: Whether to use cross-validation for evaluation.

    Returns:
    - The evaluation score based on the specified metric.
    """
    if cross_val:
        score = cross_val_score(model, X, y, cv=5, scoring=metric.__name__.split('_')[0])
    else:
        y_pred = model.predict(X)
        score = metric(y, y_pred)
        
    return score