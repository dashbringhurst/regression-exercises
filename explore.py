import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wrangle
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

def plot_variable_hist(df):
    '''This function takes in a dataframe and returns a histogram for each variable.'''
    plt.figure(figsize=(24, 20))

    # List of columns
    cols = ['bedrooms', 'bathrooms', 'sqft', 'tax_value', 'tax_amount', 'year', 'decades']
    for i, col in enumerate(cols):
        # i starts at 0, but plot nos should start at 1
        subplot_num = i+1
        # Create subplot.
        plt.subplot(5,2,subplot_num)
        # Title with column name.
        plt.title(col)
        # Display histogram for column.
        df[col].hist()
        # Hide gridlines.
        plt.grid(False)

def plot_variable_pairs(df):
    '''This function takes in a dataframe and returns a pairplot of variables with a red regression line.'''
    return sns.pairplot(df, kind='reg', corner=True, plot_kws={'line_kws':{'color':'red'}})

def plot_categorical_and_continuous_vars(df):
    '''This function takes in a dataframe and returns visualizations for each discrete/continuous variable combination.'''
    # adjust figure size to make the charts easier to see
    plt.figure(figsize=[20,16])   
    # first subplot of 9, displayed as 3 rows and 3 columns
    plt.subplot(3,3,1)
    # enhanced boxplot with scatterpoints of outliers removed
    sns.boxenplot(x='bedrooms', y='tax_value', data=df, showfliers=False)
    # red regression line overlayed on boxplot, extended to fit the entire x-axis
    sns.regplot(x='bedrooms', y='tax_value', data=df, truncate=False, scatter=False, color='red')
    # second subplot of 9, displayed as 3 rows and 3 columns
    plt.subplot(3,3,2)
    # enhanced boxplot with scatterpoints of outliers removed
    sns.boxenplot(x='bathrooms', y='tax_value', data=df, showfliers=False)
    # red regression line overlayed on boxplot, extended to fit the entire x-axis
    sns.regplot(x='bathrooms', y='tax_value', data=df, truncate=False, scatter=False, color='red')
    # third subplot of 9, displayed as 3 rows and 3 columns
    plt.subplot(3,3,3)
    # lineplot is used for 'year' because it was the best visualization for this variable
    sns.lineplot(x='year', y='tax_value', data=df)
    # red regression line overlayed on boxplot, extended to fit the entire x-axis
    sns.regplot(x='year', y='tax_value', data=df, truncate=False, scatter=False, color='red')
    # the rest of the code is the same as above but the y-variables are changed
    plt.subplot(3,3,4)
    sns.boxenplot(x='bedrooms', y='tax_amount', data=df, showfliers=False)
    sns.regplot(x='bedrooms', y='tax_amount', data=df, truncate=False, scatter=False, color='red')
    plt.subplot(3,3,5)
    sns.boxenplot(x='bathrooms', y='tax_amount', data=df, showfliers=False)
    sns.regplot(x='bathrooms', y='tax_amount', data=df, truncate=False, scatter=False, color='red')
    plt.subplot(3,3,6)
    sns.lineplot(x='year', y='tax_amount', data=df)
    sns.regplot(x='year', y='tax_amount', data=df, truncate=False, scatter=False, color='red')
    plt.subplot(3,3,7)
    sns.boxenplot(x='bedrooms', y='sqft', data=df, showfliers=False)
    sns.regplot(x='bedrooms', y='sqft', data=df, truncate=False, scatter=False, color='red')
    plt.subplot(3,3,8)
    sns.boxenplot(x='bathrooms', y='sqft', data=df, showfliers=False)
    sns.regplot(x='bathrooms', y='sqft', data=df, truncate=False, scatter=False, color='red')
    plt.subplot(3,3,9)
    sns.lineplot(x='year', y='sqft', data=df)
    sns.regplot(x='year', y='sqft', data=df, truncate=False, scatter=False, color='red')

def select_kbest(X, y, k):
    '''This function takes in three arguments, X (selected features), y (target variable), and k (number of 
    features to select) and calculates the top features using selectKbest. The function returns a list of the 
    top features.'''
    f_selector = SelectKBest(f_regression, k=k)
    # find the top 2 X's correlated with y
    f_selector.fit(X, y)
    # boolean mask of whether the column was selected or not. 
    feature_mask = f_selector.get_support()
    # get list of top K features. 
    f_feature = X.iloc[:,feature_mask].columns.tolist()
    return f_feature

def rfe(X, y, k):
    '''This function takes in three arguments, X (selected features), y (target variable), and k (number of 
    features to select) and calculates the top features using recursive feature elimination. The function
    returns a list of the top features.'''
    # initialize the ML algorithm
    lm = LinearRegression()
    # create the rfe object, indicating the ML object (lm) and the number of features I want to end up with. 
    rfe = RFE(lm, n_features_to_select=k)
    # fit the data using RFE
    rfe.fit(X,y)  
    # get the mask of the columns selected
    feature_mask = rfe.support_
    # get list of the column names. 
    rfe_feature = X.iloc[:,feature_mask].columns.tolist()
    return rfe_feature