import requests
from pandas.io.json import json_normalize
import pandas as pd
import os
import numpy as np

# Downloading from netzschleuder API.
# It is slow so this code saves in messy df pkl format
#         if it has not been run before.
if os.path.exists("Input/messydata.pkl")==False:
    os.makedirs("Input")
    url = "https://networks.skewed.de/api/nets?full=True"
    df = pd.read_json(url)
    df.to_pickle("Input/messydata.pkl")
else:
    df = pd.read_pickle("Input/messydata.pkl")

finaldf = pd.DataFrame(index=df.index) # empty new dataframe with same index 

for c in range(len(df.axes[1])):
    s_c = s = df.iloc[:,c] # load column of dataframe
    c_nets = s_c.loc['nets'] # load sub-network list
    folder_name = s_c.name # get folder name for network
    full_dict = s_c.loc['analyses'] # get dictionary of features
    if len(c_nets)==1: # if no sub networks
        finaldf = pd.concat((finaldf,s_c),axis=1)
    if len(c_nets)>1: # if there are sub networks
        if len(c_nets)<100: # speed up code in testing
            for n in range(len(c_nets)): # for each sub network
                temp_s_c = s_c.copy(deep=True) # copy column
                net_name = c_nets[n] # get sub-name
                # prepare full name to access 
                full_name = str(folder_name)+'/'+str(net_name)
                # get analysis dictionary for sub-net
                single_dict = list(full_dict.values())[n]
                if type(single_dict) is dict:
                    # update column for sub network
                    temp_s_c = temp_s_c.rename(full_name)
                    temp_s_c['analyses'] = single_dict
                    temp_s_c['nets'] = full_name
                    # add to dataframe
                    finaldf = pd.concat((finaldf,temp_s_c),axis=1)
                else: # handling failure if type not dictionary
                    print("--------------------------")
                    print("full name: ",full_name)
                    print("full_dict",full_dict)
                    print("folder: ",folder_name)
                    print("nets: ",c_nets)
                    input("fail")
                    continue

# Extract dictionary of analyses to dataframe - make easier to read and search.
newdf = finaldf.transpose()
analyses_df = pd.DataFrame.from_records(newdf.analyses.tolist())
analyses_df = analyses_df.transpose()

# Join analyses dataframe to prior dataframe.
analyses_df.columns = finaldf.columns
join_df = pd.concat((finaldf,analyses_df),axis=0)

# Get list flattened of all tags.
tag_lists = join_df.loc["tags"].values.flatten()

# Unpack tag lists and find unique tags.
tags =  [tag_lists[i] for i in range(len(tag_lists))]
tags_flat = [item for sublist in tags for item in sublist]
unique_tags, counts = np.unique(tags_flat,return_counts=True)
print(unique_tags)
print(counts)

# Expand tags so that they appear as individual boolean rows.
join_df = join_df.transpose()
for tag in unique_tags:
    col_name = "Tag-"+tag
    join_df[col_name] = [ tag in tt for tt in join_df['tags'] ]

# Filter out multigraph.
filtered_df = join_df.loc[join_df['Tag-Multigraph']==False,]
# Filter out directed graphs.
filtered_df = filtered_df.loc[filtered_df['is_directed']==False,]
# Filter out Weighted graphs.
filtered_df = filtered_df.loc[filtered_df['Tag-Weighted']==False,]
# Select Unweighted graphs.
filtered_df = filtered_df.loc[filtered_df['Tag-Unweighted']==True,]
# Get new datafram with unipartite networks.
unipartite_df = filtered_df.loc[filtered_df['is_bipartite']==False,]
# Get new dataframe with bipartite networks.
bipartite_df = filtered_df.loc[filtered_df['is_bipartite']==True,]

# Transpose back to display
filtered_df = filtered_df.transpose()
print(filtered_df)

unipartite_df = unipartite_df.transpose()
print(unipartite_df)
# Save unpartite and bipartite as csv
os.makedirs('Data', exist_ok=True)  
unipartite_df.to_csv('Data/unipartite.csv')  
unipartite_df.to_pickle('Data/unipartite.pkl')  
bipartite_df = bipartite_df.transpose()
print(bipartite_df)

bipartite_df.to_csv('Data/bipartite.csv')  
bipartite_df.to_pickle('Data/bipartite.pkl')


