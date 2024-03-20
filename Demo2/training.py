import numpy as np
import optuna
import random
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import precision_recall_curve

from .evaluation import evaluate


def create_model(params={}, model_type='classification', seed=42):
    """
    Create a LightGBM model based on the given parameters.

    Parameters:
    - params: Dictionary of hyperparameters for the model.
    - model_type: Type of the model, either 'regression' or 'classification'.
    - seed: Random seed for reproducibility.

    Returns:
    - LGBMRegressor or LGBMClassifier: LightGBM model.
    """
    assert(model_type in ["regression", "classification"])
        
    if model_type == "regression":
        model = LGBMRegressor(**params, n_jobs=-1, random_state=seed, verbosity=0)
    else:
        model = LGBMClassifier(**params, n_jobs=-1, random_state=seed, verbosity=0)
    
    return model


def select_threshold(model, X, y):
    """
    Select the best threshold for classification based on F1 score.

    Parameters:
    - model: Classifier model with predict_proba method.
    - X: Input features for prediction.
    - y: True labels.

    Returns:
    - float: Best threshold.
    """
    y_prob = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    return best_threshold


def tune(X, y, n_trials=100, direction='maximize'):
    """
    Optimize hyperparameters using Optuna for a LightGBM model.

    Parameters:
    - X: Input features for training.
    - y: True labels for training.
    - n_trials: Number of optimization trials.
    - direction: Direction of optimization, 'maximize' or 'minimize'.

    Returns:
    - optuna.study.Study: Optuna study object.
    """
    objective = Objective(X_train=X, y_train=y)

    # Creating a study object and optimize the objective function
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Printing the best hyperparameters and corresponding accuracy
    print('Best Trial:')
    print('  Value: ', study.best_trial.value)
    print('  Params: ')
    study_best = {}
    for key, value in study.best_trial.params.items():
        print(f'    {key}: {value}')
        study_best[key] = value

    return study_best


class Objective:
    """
    Objective function for Optuna optimization.

    Parameters:
    - X_train: Input features for training.
    - y_train: True labels for training.
    - model_type: Type of the model, either 'regression' or 'classification'.
    - seed: Random seed for reproducibility.
    """
       
    def __init__(self, X_train, y_train, model_type='classification', seed=42):
        self.X_train = X_train
        self.y_train = y_train
        assert(model_type in ["regression", "classification"])
        self.model_type = model_type
        self.seed = seed

    def __call__(self, trial):
        random.seed(self.seed)
        X_train = self.X_train
        y_train = self.y_train
        model_type = self.model_type
        seed = self.seed
        
        # Defining the hyperparameters to search over
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 5,1000),
            'num_leaves': trial.suggest_int('num_leaves', 10, 50),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
        }

        # Evaluating the model with the current set of hyperparameters
        model = create_model(params, model_type, seed)

        random.seed(seed)
        train_idx = random.sample(list(X_train.index), round(X_train.shape[0] * 4/5))
        valid_idx = list(set(X_train.index) - set(train_idx))

        model.fit(X_train.loc[train_idx], y_train.loc[train_idx])
        best_threshold = select_threshold(model, X_train.loc[train_idx], y_train.loc[train_idx])
        score_valid = evaluate(model, X_train.loc[valid_idx], y_train.loc[valid_idx], model_type=model_type,
                               classification_threshold=best_threshold)
          
        return score_valid