import pandas as pd
import numpy as np
import env
import acquire
import os
from sklearn.model_selection import train_test_split

def get_connection(db, user=env.user, host=env.host, password=env.password):
    '''This function uses credentials from an env file to log into a database'''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_zillow_db():
    '''The function uses the get_connection function to connect to a database and retrieve the zillow dataset'''
    return pd.read_sql('''SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, 
    yearbuilt, taxamount, fips from properties_2017;''', get_connection('zillow'))

def get_zillow_data():
    '''
    This function reads in telco data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_db()
        
        # Cache data
        df.to_csv('zillow.csv')
        
    return df

def wrangle_zillow():
    '''This function acquires the zillow dataset from the Codeup database using a SQL query and returns a cleaned
    dataframe from a csv file.'''
    # use the get_zillow_data function to acquire the dataset and save it to a csv
    df = get_zillow_data()
    # drop rows with null values
    df = df.dropna()
    # change bedroom count to an integer
    df.bedroomcnt = df.bedroomcnt.astype(int)
    # change year built to an integer
    df.yearbuilt = df.yearbuilt.astype(int)
    # change fips to an integer
    df.fips = df.fips.astype(int)
    # rename columns for readability
    df = df.rename(columns={'bedroomcnt': 'bedrooms', 'bathroomcnt': 'bathrooms', 'calculatedfinishedsquarefeet': 'sqft', 
                        'taxvaluedollarcnt': 'tax_value', 'yearbuilt': 'year', 'taxamount': 'tax_amount'})
    return df

def split_data(df, column):
    '''This function takes in two arguments, a dataframe and a string. The string argument is the name of the
        column that will be used to stratify the train_test_split. The function returns three dataframes, a 
        training dataframe with 60 percent of the data, a validate dataframe with 20 percent of the data and test
        dataframe with 20 percent of the data.'''
    train, test = train_test_split(df, test_size=.2, random_state=217, stratify=df[column])
    train, validate = train_test_split(train, test_size=.25, random_state=217, stratify=train[column])
    return train, validate, test

def quantile_scaler():
    '''This function returns scaled dataframes for train, validate, and test using the QuantileTransformer method
    from sklearn.preprocessing with a normal output distribution.'''
    scaler = sklearn.preprocessing.QuantileTransformer(output_distribution='normal')
    # scale the train set and save into a variable as a dataframe
    X_train = pd.DataFrame(scaler.fit_transform(train))
    # scale the validate set and save into a variable as a dataframe
    X_validate = pd.DataFrame(scaler.fit_transform(validate))
    # scale the test set and save into a variable as a dataframe
    X_test = pd.DataFrame(scaler.fit_transform(test))
    # return the three scaled dataframes for data exploration and modeling
    return X_train, X_validate, X_test

