'''
Placeholder file for analysis of second degree approximation of real networks
'''
import pandas as pd
from AnalysisBase import *

df_second = pd.read_pickle('Output/RealUniNets/RealUniNets_K2.pkl')
df_first = pd.read_pickle('Output/RealUniNets/RealUniNets.pkl')

print(df_second['rchi_second'])
print(df_first['rchi'])


