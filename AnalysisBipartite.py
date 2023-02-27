'''
Placeholders for the analysis of real bipartite graph results
'''

import pandas as pd
from AnalysisBase import *

df_bipartite = pd.read_pickle('Output/RealBipartiteNets/RealBipartiteNets.pkl')


print(df_bipartite[['rchi_uni', 'rchi_1', 'rchi_2']])

