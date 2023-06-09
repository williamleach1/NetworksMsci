# Here have functions that are used in the analysis of the data
# From ML models to correlatuion analysis

import pandas as pd
from copy import deepcopy
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics 
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import shapiro
"""
These first functions are primarily data cleaning functions

"""




# Function to get intersection of index of two dataframes
def get_intersection_index(df1,df2):
    # get the index of the two dataframes
    df1_index = df1.index.values.tolist()
    df2_index = df2.index.values.tolist()
    # get the intersection of the two dataframes
    intersection_index = [x for x in df1_index if x in df2_index]
    return intersection_index

# Function to filter the dataframe based on the index
def filter_dataframe_index(df,index):
    # filter the dataframe
    df_filtered = df.loc[index]
    return df_filtered

# Function to filter the dataframe based on the columns
def filter_dataframe_columns(df,columns):
    # filter the dataframe
    df_filtered = df[columns]
    return df_filtered

# Function to join two dataframes on the index 
# and return the joined dataframe
def join_dataframes(df1,df2):
    # join the two dataframes
    df_joined = pd.concat((df1,df2),axis=1)
    return df_joined

# Function to clean the dataframe
def clean_dataframe(df):
    # drop the rows with any NaN values
    df_clean = df.dropna()
    # keep with rchi > 1
    df_clean = df_clean[df_clean['rchi'] > 1]
    # Set all the values of the dataframe to be floats
    df_clean = df_clean.astype(float)
    return df_clean

# Function to save to excel
def save_to_excel(df,excel_file, folder='ExcelFiles'):
    # save to excel
    # make directory if it does not exist
    os.makedirs(folder,exist_ok=True)
    # save to excel
    df.to_excel('ExcelFiles/'+excel_file+'.xlsx')
    return None

# Function to get the index of the dataframe
def get_index(df):
    # get the index of the dataframe
    index = df.index.values.tolist()
    return index

# Function to get the columns of the dataframe
def get_columns(df):
    # get the columns of the dataframe
    columns = df.columns.values.tolist()
    return columns

# Function to go through columns of a dataframe
# and take user input to decide whether to keep the column

def filter_columns_user(df):
    # get the columns of the dataframe
    columns = get_columns(df)
    # make a copy of the dataframe
    df_filtered = deepcopy(df)
    # go through the columns and ask the user if they want to keep the column
    kept_columns = []
    for column in columns:
        # ask the user if they want to keep the column
        keep = input('Keep column '+column+'? (y/n) ')
        # if the user does not want to keep the column then drop it
        if keep == 'n':
            df_filtered = df_filtered.drop(column,axis=1)
            print('Dropped column '+column)
        else:
            kept_columns.append(column)
    print('-----------------------')
    print('Kept columns: ')
    print(kept_columns)
    print('-----------------------')
    print('Dropped columns: ')
    print([x for x in columns if x not in kept_columns])
    return df_filtered, kept_columns

# Function to plot the correlation matrix of a dataframe,
# and return the correlation matrix
# Repeat for pearson, spearman, kendall
# plot on seperate plots
def plot_correlation_matrix(df,method='pearson'):
    # plot the correlation matrix of the dataframe
    # get the correlation matrix
    corr_matrix = df.corr(method=method)
    # plot the correlation matrix
    plt.figure(figsize=(20,20))
    sns.heatmap(corr_matrix,annot=True,cmap='coolwarm')
    plt.title('Correlation matrix of '+method)
    plt.show()
    return corr_matrix


# function to return lambda function to classify
# a value based on if above or below a certain value
def get_lambda_function(value):
    # return lambda function
    return lambda x: 0 if x > value else 1

# Function to apply a lambda function to a column
# and return the new df
def apply_lambda_function(df,column,lambda_function):
    # apply the lambda function to the column
    df[column] = df[column].apply(lambda_function)
    return df

# find median of a column
def find_median(df,column):
    # find the median of the column
    median = df[column].median()
    return median

# Split in to X and Y
def split_X_Y(df, Y_column):
    # split the dataframe in to X and Y
    # drop the Y column from the dataframe
    X = df.drop(Y_column,axis=1)
    Y = df[Y_column]
    return X, Y

# Function to plot the distribution of a column
def plot_distribution(df,column):
    # plot the distribution of a column
    # get the values of the column
    values = df[column].values
    # plot the distribution
    plt.hist(values)
    plt.title('Distribution of '+column)
    plt.show()
    return None

# Function to find normallity of columns
def find_normallity(df):
    alpha = 0.05
    # get the columns of the dataframe
    columns = get_columns(df)
    # go through the columns and find the normallity
    passed = []
    failed = []
    for column in columns:
        # get the values of the column
        values = df[column].values
        # find the normallity
        stat, p = shapiro(values)
        print('Column: '+column)
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
            passed.append(column)
        else:
            print('Sample does not look Gaussian (reject H0)')
            failed.append(column)
        print('-----------------------')
    return passed, failed

# Function to split and scale the data
# Avoid data leakage by splitting before scaling
# Have an option to scale the data
def split_scale_data(X, Y, random, scale_data=True, test_size=0.25):
    # split the data in to train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=test_size,
                                                       random_state=random)
    # scale the data
    if scale_data:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, Y_train, Y_test



# Function to return the confusion matrix
# as well as the accuracy, precision and recall
def get_confusion_matrix_and_stats(Y_test, Y_pred):
    # get the confusion matrix
    cm = metrics.confusion_matrix(Y_test, Y_pred)
    # get the accuracy, precision and recall
    accuracy = metrics.accuracy_score(Y_test, Y_pred)
    precision = metrics.precision_score(Y_test, Y_pred)
    recall = metrics.recall_score(Y_test, Y_pred)
    return cm, accuracy, precision, recall

# Function to plot the confusion matrix
# if multiple confusion matrices are passed in
# then plot them all
def plot_confusion_matrix(cm_list, accuracy_list, precision_list, recall_list):
    # plot the confusion matrix
    # get the number of confusion matrices
    n_cm = len(cm_list)
    # plot the confusion matrices
    fig, ax = plt.subplots(1,n_cm,figsize=(20,10))
    for i in range(n_cm):
        # get the confusion matrix
        cm = cm_list[i]
        # plot the confusion matrix
        # get the labels
        labels = ['True Negative','False Positive',
                  'False Negative','True Positive']
        labels = np.asarray(labels).reshape(2,2)
        # plot the confusion matrix
        sns.heatmap(cm,ax=ax[i],annot=labels,fmt='',cmap='Blues')
        # set the title
        ax[i].set_title('Accuracy: '+str(round(accuracy_list[i],3))+
                        ', Precision: '+str(round(precision_list[i],3))+
                        ', Recall: '+str(round(recall_list[i],3)))
    plt.show()
    return None

# Plot all columns of a dataframe against chosen column
def plot_all_columns_against_chosen(df, chosen_column):
    # get the columns of the dataframe
    columns = get_columns(df)
    # plot all columns against chosen column
    for column in columns:
        # plot the column against the chosen column
        plt.scatter(df[column],df[chosen_column])
        plt.title(column+' vs '+chosen_column)
        plt.xlabel(column)
        plt.ylabel(chosen_column)
        plt.show()
    return None

# go through predicted vs actual and print
# Option for classification or regression
# If classification was either predicted above or below
# a certain value
def print_predicted_vs_actual(Y_test_actual, Y_test, Y_pred, classification=True, value=0.5):
    for i in range(len(Y_pred)):
        print('---------------------------------')
        print('prediction = ',Y_pred[i],', expected = ',Y_test[i])
        if classification:
            if Y_pred[i] == 0:
                print('rchi > 1.5 predicted, rchi = ',Y_test_actual.iloc[i]) 
            else:
                print('rchi < 1.5 predicted, rchi = ',Y_test_actual.iloc[i])    
        else:
            if Y_pred[i] == Y_test[i]:
                print('correct')
            else:
                print('incorrect')
    return None











