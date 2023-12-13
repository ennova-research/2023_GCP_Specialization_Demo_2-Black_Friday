import numpy as np
import pandas as pd


def encode_categoricals(data):
    """
    Encode categorical variables in the given DataFrame.

    Parameters:
    - data: DataFrame containing the data to be encoded.

    Returns:
    - DataFrame with encoded categorical variables.
    """
    # Convert 'Gender' to binary (0 for 'M', 1 for 'F')
    data['Gender'] = (data['Gender'] == 'F').astype(int)
    
    # Convert 'Age' to the average of the age range
    data.loc[data['Age'] == '55+', 'Age'] = '55-95'
    data['Age'] = data['Age'].apply(lambda x: np.mean(list(map(int, x.split('-')))))
    
    # One-hot encode 'Occupation' and append to the DataFrame
    one_hot_encoded_city = pd.get_dummies(data['Occupation'], prefix='Occupation').astype(int)
    data = pd.concat([data, one_hot_encoded_city], axis=1)
    data = data.drop('Occupation', axis=1)
    
    # One-hot encode 'City_Category' and append to the DataFrame
    one_hot_encoded_city = pd.get_dummies(data['City_Category'], prefix='City_Category').astype(int)
    data = pd.concat([data, one_hot_encoded_city], axis=1)
    data = data.drop('City_Category', axis=1)
    
     # Extract the first character of 'Stay_In_Current_City_Years' (possible values are '0', '1', '2', '3', and '4+') and convert to int
    data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].str[:1].astype(int)
    
    return data


def preprocess_per_regression(df, X_cols, y_col):
    """
    Preprocess data for regression tasks.

    Parameters:
    - df: DataFrame containing the original data.
    - X_cols: List of column names for features.
    - y_col: Column name for the target variable.

    Returns:
    - X: Features DataFrame after preprocessing.
    - y: Target variable DataFrame.
    """
    X = df.loc[:, X_cols].copy()
    y = df.loc[:, y_col].copy()
    
    # Encode categorical variables
    X = encode_categoricals(X)
    
    return X, y



def preprocess_per_classification(df, X_cols, old_y_col, new_y_col='y', threshold=None, threshold_quantile=.7):
    """
    Preprocess data for classification tasks.

    Parameters:
    - df: DataFrame containing the original data.
    - X_cols: List of column names for features.
    - old_y_col: Column name for the original target variable.
    - new_y_col: Column name for the new target variable (default is 'y').
    - threshold: Threshold for binary classification (default is calculated using quantile).
    - threshold_quantile: Quantile for threshold calculation (default is 0.7).

    Returns:
    - X: Features DataFrame after preprocessing.
    - y: New target variable DataFrame.
    - threshold: Calculated threshold for binary classification.
    """
    if not threshold:
        # Calculate threshold using quantile
        threshold = np.quantile(df[old_y_col], threshold_quantile, axis=0)
    
    # Create new binary target variable based on the threshold
    df[new_y_col] = (df[old_y_col] >= threshold).astype(int)
    
    X = df.loc[:, X_cols].copy()
    y = df.loc[:, new_y_col].copy()
    
    # Encode categorical variables
    X = encode_categoricals(X)
    
    return X, y, threshold
