import pandas as pd
from copy import deepcopy
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import ProcessBase as pb
import seaborn as sns
import shap
from sklearn.svm import SVR

Unipartite_df = pd.read_pickle('Data/unipartite.pkl')

# transpose the dataframes so that the index is the network name
Unipartite_df = Unipartite_df.transpose()

# load completed dataframes
df_uni_processed = pd.read_pickle('Output/RealUniNets/RealUniNets.pkl')

# display unipartite dataframes

# Now we want to join the two uniprtite dataframes together 
# on the index (which is the network name)
# need to trasnspose the unipartite dataframe to make the index the column

uni_names_processed = df_uni_processed.index.values.tolist()
uni_names_all = Unipartite_df.index.values.tolist()

# get the names of the networks that have been processed
# this is not really needed as we can just use the index of the processed dataframe 
uni_names = [x for x in uni_names_all if x in uni_names_processed]

print(len(uni_names))

# take the subset of the unipartite dataframe that has not processed
Unipartite_df_features_processed = deepcopy(Unipartite_df.loc[uni_names])

# print two dataframes
print('Unipartite_df_features_processed')
print(Unipartite_df_features_processed)
print('df_uni_processed')
print(df_uni_processed)

# join the two dataframes together
df_uni_processed = pd.concat((df_uni_processed,Unipartite_df_features_processed),axis=1)

# display the new unipartite dataframe
print('df_uni_processed')
print(df_uni_processed)

# save to excel
# make directory if it does not exist
os.makedirs('ExcelFiles',exist_ok=True)
df_uni_processed.to_excel('ExcelFiles/Processed_With_Features_Uni.xlsx')

# Now to clean up dataframe
# remove the columns that are not needed
# iterate through and take user input to remove columns
# this is a bit of a pain but it is the only way to do it

# get the columns of the dataframe
uni_columns = df_uni_processed.columns.values.tolist()

# get the columns that are not needed by taking user input
'''
not_needed = []
needed = []
i=0


while i < len(uni_columns):
    print('-----------------------------------')
    print('Column: ',i)
    c = uni_columns[i]
    print(c)
    user_input = input('Do you want to remove this column? (y/n)')
    if user_input == 'y':
        not_needed.append(c)
        i += 1
    elif user_input == 'n':
        needed.append(c)   
        i += 1
    else: # if invalid input then ask again
        print('Invalid input')

'''
print('-----------------------------------')
print(uni_columns)




keep4 = ['rchi', 'pearson r', 'spearmans r', 'num_edges', 'num_vertices', 
        'average_degree', 'degree_std_dev', 'global_clustering', 'degree_assortativity', 
        'largest_component_fraction', 'transition_gap', 'mixing_time', 'hashimoto_radius', 
        'diameter', 'knn_proj_1', 'knn_proj_2', "density", "av_degree", "clustering", 
        "L", "SWI", "asortivity","std_degree"]

keep = ['rchi', 'hashimoto_radius', 'diameter', "density", 
        "av_degree", "clustering", "L", "SWI", "asortivity","std_degree"]


keep1 = ["rchi","global_clustering", "degree_assortativity", "hashimoto_radius",
        "diameter","density", "av_degree", "clustering", "L", "SWI", "asortivity","std_degree"]

keep2 = ['rchi','average_degree', 'degree_std_dev', 'global_clustering', 'degree_assortativity', 
         'hashimoto_radius']

# take only the columns that are needed
df_uni_processed = df_uni_processed[keep]

# only keep rows with rchi > 1
df_uni_processed_class = df_uni_processed[df_uni_processed['rchi'] > 1]
# remove for rchi >200
df_uni_processed_class = df_uni_processed_class[df_uni_processed_class['rchi'] < 200]

# set all columns data types to float
df_uni_processed_class = df_uni_processed_class.astype(float)


svr = SVR(kernel='rbf', C=1e3, gamma=0.1, epsilon=0.1)
X = df_uni_processed_class.drop('rchi',axis=1) 
y = df_uni_processed_class['rchi']

# scale the data
scaler = StandardScaler()
scaler.fit(X)
X_scaled = pd.DataFrame(scaler.transform(X),columns = X.columns)



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

svr.fit(X_train,y_train)

predictions = svr.predict(X_test)

for i in range(len(predictions)):
    print("Predicted=%s, Actual=%s" % (predictions[i], y_test.iloc[i]))
