from lightgbm import LGBMClassifier, LGBMRegressor
from optuna import create_study
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score
from sklearn.model_selection import cross_val_score



def create_model(study=None, model_type='classification', seed=42):
    assert(model_type in ["regression", "classification"])
    
    if study == None:
        params = {}
    else:
        params = study.best_trial.params
    
    if model_type == "regression":
        model = LGBMRegressor(**params, n_jobs=-1, random_state=seed, verbosity=0)
    else:
        model = LGBMClassifier(**params, n_jobs=-1, random_state=seed, verbosity=0)
    return model


def evaluate(model, X, y, model_type='classification', cross_val=False):
    assert(model_type in ["regression", "classification"])

    if model_type == "regression":
        metric = r2_score
    else:
        metric = recall_score
        
    if cross_val:
        score = cross_val_score(model, X, y, cv=5, scoring=metric.__name__.split('_')[0])
    else:
        y_pred = model.predict(X)
        score = metric(y, y_pred)
    return score