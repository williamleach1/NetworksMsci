import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations
import os
import time
from scipy import stats
from tqdm import tqdm
import scipy as sp
from igraph import Graph
import pandas as pd
from scipy import stats
from scipy import optimize

def BA_closeVSdeg(m_,N_, its, plot=False):
  df = pd.DataFrame({"k":[],"c":[]})
  for i in range(its):
    G = Graph.Erdos_Renyi(n=1000,p=0.01)
    degrees = G.degree()
    #print(np.mean(degrees))
    close = np.asarray(G.closeness())
    #print(len(degrees),len(inv_close))
    df_new = pd.DataFrame({"k":degrees,"c":close})
    #print(df_new.head())
    df = df.append(df_new,ignore_index=True)
  #print(df.head())
  df = df.groupby("k").agg({"c":['mean','std']})
  df = df.xs('c', axis=1, drop_level=True)
  df = df.reset_index('k')
  df = df.rename(columns={"mean":"mean c","std":"std c"})

  ks = df.loc[:,"k"].tolist()
  print('mean k is ' ,np.mean(ks)) 
  cs = np.asarray(df.loc[:,"mean c"].tolist())
  cs_err = np.asarray(df.loc[:,"std c"].tolist())
  plt.plot(ks,cs_err,'x')
  inv_c = 1/cs
  inv_c_err = cs_err/np.square(cs)
  
  if plot:
    plt.figure()
    plt.errorbar(ks,inv_c,yerr=inv_c_err,fmt='none',ecolor='black',capsize=2)
    plt.plot(ks,inv_c,'r.')
    plt.xscale("log")
    plt.show()  
  return ks, inv_c, inv_c_err, cs

kss, inv_cs, err, cs = BA_closeVSdeg(5,4000,100,True)

pc= np.corrcoef(cs, kss)
print(pc)

def analytic(k,a,b):
  return -a*np.log(k)+b

opt, pcov = optimize.curve_fit(analytic,kss,inv_cs,p0=[0.3,3.5])

print(opt)
print(pcov)
#print(inv_cs)
#print(err)
