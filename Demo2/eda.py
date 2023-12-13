import numpy as np
import pandas as pd


def group_by_user(data):
    """
    Group data by user and aggregate relevant information.

    Args:
        data (pd.DataFrame): Input data containing user information.

    Returns:
        pd.DataFrame: User-level aggregated information.
    """
    users=pd.DataFrame()
    
    # Extract user-level information
    user_info_cols = ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']
    for col in user_info_cols:
        users[col] = data.groupby('User_ID').agg({col: 'first'})

    # Calculate additional user-level metrics
    users['Number of Purchases'] = data.groupby('User_ID').size()
    users['Sum spent'] = data.groupby('User_ID')['Purchase'].sum()
    users['Average Purchase'] = data.groupby('User_ID')['Purchase'].mean()
    
    # Sort users by total spending in descending order
    users=users.sort_values(by='Sum spent', ascending=False)
    
    return users


def group_by_product(data):
    """
    Group data by product and aggregate relevant information.

    Args:
        data (pd.DataFrame): Input data containing product information.

    Returns:
        pd.DataFrame: Product-level aggregated information.
    """
    prods=pd.DataFrame()
    
    # Extract product-level information
    prods['Product_Category_1'] = data.groupby('Product_ID').agg({'Product_Category_1': 'first'})
    
    # Calculate additional product-level metrics
    prods['Number of Purchases'] = data.groupby('Product_ID').size()
    prods['Sum spent'] = data.groupby('Product_ID')['Purchase'].sum()
    prods['Unitary Price'] = prods['Sum spent'] / prods['Number of Purchases']
    
    # Sort products by total spending in descending order
    prods=prods.sort_values(by='Sum spent', ascending=False)
    
    return prods


def group_by_product_category(data):
    """
    Group data by product category and aggregate relevant information.

    Args:
        data (pd.DataFrame): Input data containing product category information.

    Returns:
        pd.DataFrame: Product category-level aggregated information.
    """
    prods_cat=pd.DataFrame()
    
    # Calculate product category-level metrics
    prods_cat['Number of Purchases'] = data.groupby('Product_Category_1').size()
    prods_cat['Sum spent'] = data.groupby('Product_Category_1')['Purchase'].sum()
    prods_cat['Average Price'] = prods_cat['Sum spent'] / prods_cat['Number of Purchases']
    
    # Sort product categories by total spending in descending order
    prods_cat=prods_cat.sort_values(by='Sum spent', ascending=False)
    
    return prods_cat


def group_by_user_category(data):
    """
    Group data by user categories and aggregate relevant information.

    Args:
        data (pd.DataFrame): Input data containing user and purchase information.

    Returns:
        pd.DataFrame: User category-level aggregated information.
    """
    user_cat=pd.DataFrame()
    
    # Specify columns for grouping
    grouping_cols = ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']
    
    # Calculate user category-level metrics
    user_cat['Numerosity'] = data.groupby(grouping_cols).size()
    user_cat['log_Numerosity'] = np.log(user_cat['Numerosity'])
    user_cat['Sum spent'] = data.groupby(grouping_cols)['Sum spent'].sum()
    user_cat['log_SumSpent'] = np.log(user_cat['Sum spent'])
    user_cat['Average Purchase'] = data.groupby(grouping_cols)['Average Purchase'].mean()
    
    # Sort user categories by total spending in descending order
    user_cat=user_cat.sort_values(by='Sum spent', ascending=False)
    
    return user_cat