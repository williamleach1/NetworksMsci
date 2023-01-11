import pandas as pd
from copy import deepcopy
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics 
import seaborn as sns
from sklearn.preprocessing import StandardScaler


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

# get the columns of the dataframe
uni_columns = df_uni_processed.columns.values.tolist()


keep1 = ["rchi", "pearson r", "spearmans r","global_clustering", "degree_assortativity", "hashimoto_radius",
        "diameter","density", "av_degree", "clustering", "L", "SWI", "asortivity","std_degree"]

keep = ['rchi', 'hashimoto_radius', 'diameter', "density", 
        "av_degree", "clustering", "L", "SWI", "asortivity","std_degree"]


keep2 = ['rchi','average_degree', 'degree_std_dev', 'global_clustering', 'degree_assortativity', 
         'hashimoto_radius']

# take only the columns that are needed
df_uni_processed = df_uni_processed[keep]

# only keep rows with rchi > 1
df_uni_processed = df_uni_processed[df_uni_processed['rchi'] > 1]

# set all columns data types to float
df_uni_processed = df_uni_processed.astype(float)

# Split to classes based on rchi
# rchi > 1.5 is class 0
# rchi < 1.5 is class 1
func = lambda x: 0 if x > 1.5 else 1

df_uni_processed_class = deepcopy(df_uni_processed)

df_uni_processed_class['rchi'] = df_uni_processed_class['rchi'].apply(func)


# split into X and Y. X is the features and Y is the target (rchi)
X = df_uni_processed_class.drop('rchi',axis=1)
Y = df_uni_processed_class['rchi']

# Apply standard scaler to X
scaler = StandardScaler()
scaler.fit(X)
X_scaled = pd.DataFrame(scaler.transform(X),columns = X.columns)

# split into train and test sets and return indices of the split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.25, random_state=7)

y_train_index = y_train.index.values.tolist()
y_test_index = y_test.index.values.tolist()
print('y_train_index')
print(y_train_index)
print('y_test_index')
print(y_test_index)

# now do linear regression
# define the model (sklearn linear regression)
model = LogisticRegression(random_state=0)

# fit the model
model.fit(X_train,y_train)

# make predictions for test data
y_pred = model.predict(X_test)

# get the actual rchi values for the test data by using the index
y_test_actual = df_uni_processed.loc[y_test_index]['rchi']

# display the predictions and actual values
for i in range(len(y_pred)):
    print('---------------------------------')
    print('prediction = ',y_pred[i],', expected = ',y_test[i])
    if y_pred[i] == y_test[i]:
        print('correct')
    else:
        print('incorrect')

    if y_pred[i] == 0:
        print('rchi > 1.5 predicted, rchi = ',y_test_actual.iloc[i]) 
    else:
        print('rchi < 1.5 predicted, rchi = ',y_test_actual.iloc[i])


# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))
# calculate precision and recall


# positives are good fits (1) with rchi < 1.5
# negatives are bad fits (0) with rchi > 1.5 
# precision = TP / (TP + FP) - how many of the predicted positives are actually positive
# recall = TP / (TP + FN) - how many of the actual positives are predicted positive

precision = metrics.precision_score(y_test, y_pred, )
recall = metrics.recall_score(y_test, y_pred)
print('Precision: %.2f' % (precision))
print('Recall: %.2f' % (recall))


cnf_matrix = metrics.confusion_matrix(y_test, y_pred) 
print(cnf_matrix)
# plot confusion matrix
class_names=[0,1] # name  of classes 
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names) 
# create heatmap 
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion matrix', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label')
plt.show()


feature_importance = abs(model.coef_[0])
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