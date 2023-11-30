import random
from lightgbm import LGBMClassifier, LGBMRegressor
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score
from sklearn.model_selection import cross_val_score



class Objective:
    
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
        if model_type == "regression":
            model = LGBMRegressor(**params, n_jobs=-1, random_state=seed, verbosity=-1)
            metric = r2_score
        else:
            model = LGBMClassifier(**params, n_jobs=-1, random_state=seed, verbosity=-1)
            metric = recall_score

        random.seed(seed)
        train_idx = random.sample(list(X_train.index), round(X_train.shape[0] * 4/5))
        valid_idx = list(set(X_train.index) - set(train_idx))

        model.fit(X_train.loc[train_idx], y_train.loc[train_idx], )
        y_pred = model.predict(X_train.loc[valid_idx])
        score_valid = metric(y_train.loc[valid_idx], y_pred)
        return score_valid
    