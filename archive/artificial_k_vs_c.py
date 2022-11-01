from igraph import Graph
import time
import numpy as np
import pandas as pd
import scipy as sp 
from scipy import stats
from scipy import optimize
import matplotlib.pyplot as plt

def BA_closeVSdeg(m_,N_, its, plot=False):
  df = pd.DataFrame({"k":[],"c":[]})
  print("\n-----------------------------------------------\n")
  print("m = ",m_)
  print("N = ",N_)
  for i in range(its):
    G = Graph.Barabasi(N_,m_)
    degrees = np.asarray(G.degree())
    #print(np.mean(degrees))
    close = np.asarray(G.closeness())
    #print(len(degrees),len(inv_close))
    df_new = pd.DataFrame({"k":degrees,"c":close})
    #print(df_new.head())
    df = df.append(df_new,ignore_index=True)
  #print(df.head())
  print('mean k is ' ,np.mean(degrees)) 
  degs = np.asarray(df.loc[:,"k"].tolist())
  clos = np.asarray(df.loc[:,"c"].tolist())
  
  #rho, pc= stats.pearsonr(degs, clos)
  #print("pearson rho: ",rho)
  
  df = df.groupby("k").agg({"c":['mean','std','count']})
  df = df.xs('c', axis=1, drop_level=True)
  df = df.reset_index('k')
  df = df.rename(columns={"mean":"mean c","std":"std c","count":"n"})
  
  #print(df.head())

  ks = df.loc[:,"k"].tolist()
  cs = np.asarray(df.loc[:,"mean c"].tolist())
  count_cs = np.asarray(df.loc[:,"n"].tolist())
  cs_err = np.asarray(df.loc[:,"std c"].tolist())/np.sqrt(count_cs)
  plt.plot(ks,cs_err,'x')
  inv_c = 1/cs
  inv_c_err = cs_err/np.square(cs)
  
  if plot:
    plt.figure()
    plt.errorbar(ks,inv_c,yerr=inv_c_err,fmt='none',ecolor='black',capsize=2)
    plt.plot(ks,inv_c,'r.')
    plt.xscale("log")
    plt.show()  
  return ks, inv_c, inv_c_err



kss, inv_cs, err = BA_closeVSdeg(5,4000,100,True)

#rho, pc= stats.pearsonr(kss, 1/inv_cs)
#print(rho)

def analytic(k,a,b):
  return -a*np.log(k)+b
logk= np.log(kss)

fit, V = np.polyfit(logk,inv_cs,deg=1,cov=True)
fitted = np.poly1d(fit)
fit_inv_c = fitted(logk)
#opt, pcov = optimize.curve_fit(analytic,kss,inv_cs,p0=[0.3,3.5],sigma=err)

print("1/ln(z) fit :", -fit[0],"+/-",np.sqrt(V[0][0]))
print("B fit :",fit[1],"+/-",np.sqrt(V[1][1]))
#print(inv_cs)
#print(err)


#TO DO:
"""
* Filter out rows where k=0
* Filter out rows for with zero/nan std
* Filter out if only appears once?
* Return in nice format
* Get on nice graphs for ER, BA for multiple Ns with fit line
* Download and filter graph_tool networks
* Identify other network sources
* Run through fitting before agrregation - get better error
"""