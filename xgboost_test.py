import pandas as pd
from copy import deepcopy
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt
import ProcessBase as pb
import seaborn as sns
import shap




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

# plot all features against rchi
pearsons = []

for c in df_uni_processed_class.columns.values.tolist():
    if c != 'rchi':
        plt.scatter(df_uni_processed_class[c],df_uni_processed_class['rchi'])
        plt.xlabel(c)
        plt.ylabel('rchi')
        plt.xscale('log')
        plt.yscale('log')
        plt.title(c)    
        
        pearson = pb.pearson(df_uni_processed_class[c],df_uni_processed_class['rchi'])
        spearman = pb.spearman(df_uni_processed_class[c],df_uni_processed_class['rchi'])
        pearsons.append(pearson) # append the pearson correlation coefficient
        plt.suptitle('Pearson correlation coefficient: '+str(pearson))
        plt.show()
        print('-----------------------------------')
        print('Column: ',c)
        print('Pearson correlation coefficient: ',pearson)
        print('Spearman correlation coefficient: ',spearman)



# Split to classes based on rchi

# rchi > 2 is class 1
# rchi < 2 is class 0
func = lambda x: 1 if x > 1.5 else 0

df_uni_processed_class['rchi'] = df_uni_processed_class['rchi'].apply(func)

# split into X and Y. X is the features and Y is the target (rchi)
X = df_uni_processed_class.drop('rchi',axis=1)
Y = df_uni_processed_class['rchi']


# split the dataframe into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=9)

# Define model parameters
n_est = 500
learning = 0.01
early_stop = 20

# train the model
model = XGBClassifier(max_depth=12, subsample=0.33, objective='binary:logistic',
                        n_estimators=n_est, 
                        learning_rate = learning)

eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=early_stop, eval_metric=["error", "logloss"],
            eval_set=eval_set, verbose=False)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]




# print the predictions and the actual values
for i in range(len(predictions)):
    print("Predicted=%s, Actual=%s" % (predictions[i], y_test.iloc[i]))

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

results = model.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
# plot log loss
# Create a figure with 2 subplots (1 row, 2 columns)
fig, axs = plt.subplots(1, 2)

# Plot log loss in the first subplot
axs[0].plot(x_axis, results['validation_0']['logloss'], label='Train')
axs[0].plot(x_axis, results['validation_1']['logloss'], label='Test')
axs[0].set_ylabel('Log Loss')
axs[0].set_title('XGBoost Log Loss')
axs[0].legend()

# Plot classification error in the second subplot
axs[1].plot(x_axis, results['validation_0']['error'], label='Train')
axs[1].plot(x_axis, results['validation_1']['error'], label='Test')
axs[1].set_ylabel('Classification Error')
axs[1].set_title('XGBoost Classification Error')
axs[1].legend()

plt.show()


# get the feature importance and display it and feature name
importance = model.feature_importances_
features = X.columns.values.tolist()


for i,v in enumerate(importance):
    print('-----------------------------------')
    print('Feature: %0d, Score: %.5f' % (i,v))
    print('Feature Name: ',features[i])

feature_importance = abs(model.feature_importances_)
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

featfig = plt.figure()
featax = featfig.add_subplot(1, 1, 1)
featax.barh(pos, feature_importance[sorted_idx], align='center')
featax.set_yticks(pos)
featax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=8)
featax.set_xlabel('Relative Feature Importance')

plt.tight_layout()   
plt.show()


# find permutation importance
perm = permutation_importance(model, X_test, y_test, n_repeats=100)
sorted_ids = perm.importances_mean.argsort()
plt.barh(X_test.columns[sorted_ids], perm.importances_mean[sorted_ids])
plt.xlabel("Permutation Importance")
plt.show()

# now do shap values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# plot the shap values
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)

