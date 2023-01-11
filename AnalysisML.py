from AnalysisBase import *
import pandas as pd


Unipartite_df = pd.read_pickle('Data/unipartite.pkl')

# transpose the dataframes so that the index is the network name
Unipartite_df = Unipartite_df.transpose()

# load completed dataframes
df_uni_processed = pd.read_pickle('Output/RealUniNets/RealUniNets.pkl')

processed_networks = get_intersection_index(df_uni_processed, Unipartite_df)

# Take the subset of the unipartite dataframe that has not processed
Unipartite_df_processed = filter_dataframe_index(Unipartite_df, processed_networks)

# Now we want to join the two uniprtite dataframes together
# on the index (which is the network name)
df_uni_fit_features = join_dataframes(df_uni_processed, Unipartite_df_processed)

# save to excel
save_to_excel(df_uni_fit_features, 'Processed_With_Features_Uni.xlsx')

columns = get_columns(df_uni_fit_features)


keep_all = ['N', 'E', '1/ln(z)', '1/ln(z) err', 'Beta', 'Beta err', 'rchi',
        'pearson r', 'pearson p-val', 'spearmans r', 'spearmans p-val', 
        'density', 'av_degree', 'clustering', 'L', 'SWI', 'asortivity', 
        'std_degree','transition_gap', 'mixing_time', 'hashimoto_radius', 
        'diameter', 'knn_proj_1', 'knn_proj_2']

keep2 = ['N', 'E', 'rchi', 'density', 'av_degree', 'clustering', 'L', 'SWI', 'asortivity', 
        'std_degree','transition_gap', 'mixing_time', 'hashimoto_radius', 
        'diameter', 'knn_proj_1', 'knn_proj_2']

keep = ['rchi', 'hashimoto_radius', 'diameter', "density", 
        "av_degree", "clustering", "L", "SWI", "asortivity","std_degree"]

# take only the columns that are needed
df_uni_fit_features = filter_dataframe_columns(df_uni_fit_features, keep)

# now clean the dataframe
df_uni_fit_features = clean_dataframe(df_uni_fit_features)

# Show correlation matrix
#plot_correlation_matrix(df_uni_fit_features, 'pearson')
#plot_correlation_matrix(df_uni_fit_features, 'spearman')
#plot_correlation_matrix(df_uni_fit_features, 'kendall')

# Show the distribution of the features agaisnt the rchi
#for column in keep:
    #plot_distribution(df_uni_fit_features, 'rchi')
    #plt.pause(1)
    #plt.close()


# Test for normality - find none are normal
find_normallity(df_uni_fit_features)


median_rounded = round(find_median(df_uni_fit_features, 'rchi'),2)

filter_function = get_lambda_function(1.5)

df_uni_fit_class = deepcopy(df_uni_fit_features)

# filter the dataframe
df_uni_fit_class = apply_lambda_function(df_uni_fit_class, 
                                        'rchi',filter_function)
print(df_uni_fit_class)
# First do a train test split - 80% train, 20% test
# for the classification

X_class, Y_class = split_X_Y(df_uni_fit_class, 'rchi')
X, Y = split_X_Y(df_uni_fit_features, 'rchi')

X_train, X_test, Y_train, Y_test = split_scale_data(X, Y, 42)
X_train_class, X_test_class, Y_train_class, Y_test_class = split_scale_data(X_class, Y_class, 42, scale_data=False)

print(Y_train_class)



#First lets to XGBoost - easy to get feature importance


# Now do XGBoost - on the classification

# model parameters


n_est = 500
learning = 0.01
early_stop = 20

# train the model
model = XGBClassifier(max_depth=12, subsample=0.33, objective='binary:logistic',
                        n_estimators=n_est, 
                        learning_rate = learning)


eval_set = [(X_train_class, Y_train_class), (X_test_class, Y_test_class)]
model.fit(X_train_class, Y_train_class, early_stopping_rounds=early_stop, eval_metric=["error", "logloss"],
            eval_set=eval_set, verbose=True)

y_pred_class = model.predict(X_test_class)
predictions = y_pred_class


# print the predictions and the actual values
for i in range(len(predictions)):
    print("Predicted=%s, Actual=%s" % (predictions[i], Y_test_class.iloc[i]))

# evaluate predictions
accuracy = metrics.accuracy_score(Y_test_class, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))





'''


# Lets fo k-means clustering 
# First we need to find the optimal number of clusters
# We will use the elbow method
from sklearn.cluster import KMeans

# find the optimal number of clusters
# using the elbow method
# https://pythonprogramminglanguage.com/kmeans-elbow-method/
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df_uni_fit_features)
    distortions.append(kmeanModel.inertia_)

# Plot the elbow


plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

k = 3

# Now we can do the clustering
kmeans = KMeans(n_clusters=k, random_state=0).fit(df_uni_fit_features)

# Now we can get the labels
labels = kmeans.labels_

# Now we can get the centroids
centroids = kmeans.cluster_centers_

# Now we can get the inertia
inertia = kmeans.inertia_

# Now we can get the silhouette score
from sklearn.metrics import silhouette_score
s = silhouette_score(df_uni_fit_features, labels, metric='euclidean')
print(s)

# plot rchi against the columns
for column in keep:
    plt.scatter(df_uni_fit_features[column], df_uni_fit_features['rchi'], c=labels, s=50, cmap='viridis')
    plt.xlabel(column)
    plt.ylabel('rchi')
    plt.title('Density vs rchi')
    plt.show()

'''



