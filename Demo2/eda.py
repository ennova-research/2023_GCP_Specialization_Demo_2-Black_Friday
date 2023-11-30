import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from sklearn.model_selection import train_test_split


def group_by_user(data):
    users=pd.DataFrame()
    users['Gender'] = data.groupby('User_ID').agg({'Gender': 'first'})
    users['Age'] = data.groupby('User_ID').agg({'Age': 'first'})
    users['Occupation'] = data.groupby('User_ID').agg({'Occupation': 'first'})
    users['City_Category'] = data.groupby('User_ID').agg({'City_Category': 'first'})
    users['Stay_In_Current_City_Years'] = data.groupby('User_ID').agg({'Stay_In_Current_City_Years': 'first'})
    users['Marital_Status'] = data.groupby('User_ID').agg({'Marital_Status': 'first'})
    users['Number of Purchases'] = data.groupby('User_ID').size()
    users['Sum spent'] = data.groupby('User_ID')['Purchase'].sum()
    users['Average Purchase'] = data.groupby('User_ID')['Purchase'].mean()
    return users


def group_by_product(data):
    prods=pd.DataFrame()
    prods['Product_Category_1'] = data.groupby('Product_ID').agg({'Product_Category_1': 'first'})
    prods['Number of Purchases'] = data.groupby('Product_ID').size()
    prods['Sum spent'] = data.groupby('Product_ID')['Purchase'].sum()
    prods['Unitary Price'] = prods['Sum spent'] / prods['Number of Purchases']
    prods=prods.sort_values(by='Sum spent', ascending=False)
    return prods


def group_by_product_category(data):
    prods_cat=pd.DataFrame()
    prods_cat['Number of Purchases'] = data.groupby('Product_Category_1').size()
    prods_cat['Sum spent'] = data.groupby('Product_Category_1')['Purchase'].sum()
    prods_cat['Average Price'] = prods_cat['Sum spent'] / prods_cat['Number of Purchases']
    prods_cat=prods_cat.sort_values(by='Sum spent', ascending=False)
    return prods_cat


def group_by_user_category(data):
    user_cat=pd.DataFrame()
    grouping_cols = ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']
    user_cat['Numerosity'] = data.groupby(grouping_cols).size()
    user_cat['log_Numerosity'] = np.log(user_cat['Numerosity'])
    user_cat['Sum spent'] = data.groupby(grouping_cols)['Sum spent'].sum()
    user_cat['log_SumSpent'] = np.log(user_cat['Sum spent'])
    user_cat['Average Purchase'] = data.groupby(grouping_cols)['Average Purchase'].mean()
    user_cat=user_cat.sort_values(by='Sum spent', ascending=False)
    return user_cat