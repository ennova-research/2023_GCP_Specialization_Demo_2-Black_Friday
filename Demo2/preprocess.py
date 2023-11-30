import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_categoricals(data):
    label_encoder = LabelEncoder()
    object_cols = data.columns[data.dtypes == 'object']
    data.loc[:, object_cols] = data.loc[:, object_cols].apply(lambda x: label_encoder.fit_transform(x))
    return data


def preprocess_per_regression(train_set, test_set, X_cols, y_col):
    X_train = train_set.loc[:, X_cols].copy()
    y_train = train_set.loc[:, y_col].copy()
    X_test = test_set.loc[:, X_cols].copy()
    y_test = test_set.loc[:, y_col].copy()
    
    X_train = encode_categoricals(X_train)
    X_test = encode_categoricals(X_test)
    
    return X_train, y_train, X_test, y_test



def preprocess_per_classification(train_set, test_set, X_cols, old_y_col, new_y_col='y', threshold_quantile=.7):
    threshold = np.quantile(train_set[old_y_col], threshold_quantile, axis=0)
    train_set[new_y_col] = (train_set[old_y_col] >= threshold).astype(int)
    test_set[new_y_col] = (test_set[old_y_col] >= threshold).astype(int)
    
    X_train = train_set.loc[:, X_cols].copy()
    y_train = train_set.loc[:, new_y_col].copy()
    X_test = test_set.loc[:, X_cols].copy()
    y_test = test_set.loc[:, new_y_col].copy()
    
    X_train = encode_categoricals(X_train)
    X_test = encode_categoricals(X_test)
    
    return X_train, y_train, X_test, y_test
