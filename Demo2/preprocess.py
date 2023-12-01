import numpy as np
import pandas as pd


def encode_categoricals(data):
    data['Gender'] = (data['Gender'] == 'F').astype(int)
    
    data.loc[data['Age'] == '55+', 'Age'] = '55-95'
    data['Age'] = data['Age'].apply(lambda x: np.mean(list(map(int, x.split('-')))))
    
    one_hot_encoded_city = pd.get_dummies(data['Occupation'], prefix='Occupation').astype(int)
    data = pd.concat([data, one_hot_encoded_city], axis=1)
    data = data.drop('Occupation', axis=1)
    
    one_hot_encoded_city = pd.get_dummies(data['City_Category'], prefix='City_Category').astype(int)
    data = pd.concat([data, one_hot_encoded_city], axis=1)
    data = data.drop('City_Category', axis=1)
    
    data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].str[:1].astype(int)
    
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
