from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score
from sklearn.model_selection import cross_val_score


def evaluate(model, X_test, y_test, model_type='classification'):
    if model_type == "regression":
        metric = r2_score
    else:
        metric = recall_score
        
    y_pred = model.predict(X_test)
    score = metric(y_test, y_pred)
    return score