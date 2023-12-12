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


def preprocess_per_regression(df, X_cols, y_col):
    X = df.loc[:, X_cols].copy()
    y = df.loc[:, y_col].copy()
    
    X = encode_categoricals(X)
    
    return X, y



def preprocess_per_classification(df, X_cols, old_y_col, new_y_col='y', threshold=None, threshold_quantile=.7):
    if not threshold:
        threshold = np.quantile(df[old_y_col], threshold_quantile, axis=0)
    df[new_y_col] = (df[old_y_col] >= threshold).astype(int)
    
    X = df.loc[:, X_cols].copy()
    y = df.loc[:, new_y_col].copy()
    
    X = encode_categoricals(X)
    
    return X, y, threshold
