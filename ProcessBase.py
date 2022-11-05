from graph_tool import Graph, generation, centrality, collection
import time
import numpy as np
import pandas as pd
import scipy as sp 
from scipy import stats
from scipy import optimize
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# Function to find closeness and degree for graph_tool graph
# remove values if degree 0
def GetKC(g):
    """Get closeness and degree for graph_tool graph
    Parameters  
    ----------                  
    g : graph_tool graph
        Graph to be analyzed  
    Returns
    -------     
    k : array
        Degree of each node
    c : array
        Closeness of each node"""
    c = centrality.closeness(g).get_array()
    k = g.get_total_degrees(g.get_vertices())
    #print(np.mean(k))
    c = c[k > 0]
    k = k[k > 0]
    k = k[~np.isnan(c)]
    c = c[~np.isnan(c)]
    return k, c

# function to aggregate over x to find mean and standard error in y
#           --> remove if only appears once
def aggregate(x, y):
    """Aggregate over x to find mean and standard error in y
    Parameters  
    ----------                  
    x : array
        x data
    y : array
        y data
    Returns
    -------     
    x_mean : array
        Mean x data
    y_mean : array
        Mean y data
    y_err : array
        Standard error of y data"""
    x_mean = np.unique(x)
    y_mean = np.zeros(len(x_mean))
    y_err = np.zeros(len(x_mean))
    for i in range(len(x_mean)):
        y_mean[i] = np.mean(y[x == x_mean[i]])
        y_err[i] = np.std(y[x == x_mean[i]])/np.sqrt(len(y[x == x_mean[i]]))
    return x_mean, y_mean, y_err




# class to plot in groups
#           --> options for legends ect - class based?
#           --> Options to save plots
#           --> Options to save data
class Plotter:
    def __init__(self):
        self.x = []
        self.y = []
        self.x_collapsed = []
        self.y_collapsed = []
        self.labels = []
        self.colors = []
        self.markers = []
        self.linestyles = []
        self.x_label = ''
        self.y_label = ''
        self.title = ''
        self.x_lim = [-np.inf, np.inf]
        self.y_lim = [-np.inf, np.inf]
        self.legend = False
        self.legend_loc = 'best'
        self.legend_title = ''
        self.data_collapse = False
        self.collapse_function = None

    def add_plot(self, x, y, label, color, marker, linestyle, collapse=False, collapse_function=None):
        self.x.append(x)
        self.y.append(y)
        self.labels.append(label)
        self.colors.append(color)
        self.markers.append(marker)
        self.linestyles.append(linestyle)
        if collapse:
            self.data_collapse = True
            self.collapse_function = collapse_function

    def plot(self,scale='log',legend=True,save=False):
        for i in range(len(self.x)):
            plt.plot(self.x[i], self.y[i], label=self.labels[i], color=self.colors[i], marker=self.markers[i], linestyle=self.linestyles[i])
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        plt.xlim(self.x_lim)
        plt.ylim(self.y_lim)
        if self.legend:
            plt.legend(loc=self.legend_loc, title=self.legend_title)
        if scale == 'log':
            plt.xscale('log')
        plt.show()

# Function to perform pearson test on unaggregated data.
def pearson(x, y):
    """Perform pearson test on unaggregated data.
    Parameters  
    ----------                  
    x : array
        x data
    y : array
        y data
    Returns
    -------     
    r : float
        Pearson correlation coefficient
    p : float
        p-value"""
    r, p = stats.pearsonr(x, y)
    return r, p

# Function to perform rank correlation test on unaggregated data.
def spearman(x, y):
    """Perform rank correlation test on unaggregated data.
    Parameters  
    ----------                  
    x : array
        x data
    y : array
        y data
    Returns
    -------     
    r : float
        Spearman correlation coefficient
    p : float
        p-value"""
    r, p = stats.spearmanr(x, y)
    return r, p

# Function to fit to
# and to perform straight line fit to unaggregated data
def func(k, a, b):
    return -a*np.log(k) + b

def Uni_1st(k,c,function,to_print=False):
    """Perform fit to specified function
    Parameters
    ----------
    k : array
        Degree of each node
    c : array   
        Closeness of each node
    function : function
        Function to fit to
    to_print : bool
        Print results
    Returns
    ------- 
    popt[0] : float         
        a               
    popt[1] : float
        b
    pcov[0,0] : float
        a error
    pcov[1,1] : float
        b error
    """
    popt, pcov = optimize.curve_fit(function, k, 1/c)
    if to_print:
        print("1/ln(z) fit :", popt[0],"+/-",np.sqrt(pcov[0][0]))
        print("B fit :",popt[1],"+/-",np.sqrt(pcov[1][1]))
    return popt[0], np.sqrt(pcov[0][0]), popt[1], np.sqrt(pcov[1][1])

# Function to perform chi square test on unaggregated data using standard deviation
def red_chi_square(k, c, function, popt, sigma):
    """Perform chi square test on unaggregated data
    Parameters  
    ----------                  
    k : array
        Degree of each node
    c : array
        Closeness of each node
    expected_inv_c : array
        Expected inverse closeness of each node
    Returns
    -------     
    chi : float
        Chi square value
    p : float
        p-value"""
    inv_c = 1/c
    expected_inv_c = function(k, *popt)
    chi = np.sum(((inv_c - expected_inv_c)**2)/(sigma**2))
    rchi = chi/(len(k)-2)
    return rchi

# Function to do all above for given graph

def process(g, to_print=False):
    """Perform all analysis on graph
    Parameters  
    ----------                  
    g : graph_tool graph
        Graph to be analyzed
    Returns
    -------     
    k : array
        Degree of each node
    c : array
        Closeness of each node
    a : float
        a
    a_err : float
        a error
    b : float
        b
    b_err : float
        b error
    rchi : float
        Reduced chi square value
    p : float
        p-value
    r : float
        Pearson correlation coefficient
    rp : float
        p-value
    rs : float
        Spearman correlation
    rsp : float
        p-value"""
    k, c = GetKC(g)
    a, a_err, b, b_err = Uni_1st(k, c, func, to_print=to_print)
    expected_inv_c = func(k, a, b)
    rchi = 1 #red_chi_square(k, c, expected_inv_c)
    r, rp = pearson(k, c)
    rs, rsp = spearman(k, c)
    if to_print:
        print("Reduced chi square:", rchi)
        print("Pearson correlation:", r)
        print("Pearson p-value:", rp)
        print("Spearman correlation:", rs)
        print("Spearman p-value:", rsp)
    return k, c, a, a_err, b, b_err, rchi, r, rp, rs, rsp

# Function to generate BA, ER, Config-BA

def BA(n, av_deg):
    """Generate BA graph
    Parameters  
    ----------                  
    n : int
        Number of nodes
    av_deg : int
        Average degree
    Returns
    -------     
    g : graph_tool graph
        Generated graph"""

    m = int(av_deg/2)
    g = generation.price_network(n, m, directed=False)
    return g

def ER(n, av_deg):
    """Generate ER graph
    Parameters  
    ----------                  
    n : int
        Number of nodes
    av_deg : float
        Average degree
    Returns
    -------     
    g : graph_tool graph
        Generated graph"""
    g = Graph()
    g.set_directed(False)
    p = av_deg/(n-1)
    for i in range(n):
        g.add_vertex()
    for a in range(n):
        for b in range(a+1,n):
            if (random.random() <p):
                g.add_edge(a,b)
    return g

'''
def config_BA(n, av_deg):
    """Generate Config-BA graph
    Parameters  
    ----------                  
    n : int
        Number of nodes
    av_deg : int
        Average degree
    Returns
    -------     
    g : graph_tool graph
        Generated graph"""
    m = int(av_deg/2)
    g = generation.price_network(n, m, directed=False)
    g = generation.random_rewire(g, model="configuration")
    return g
'''

# Run for BA, ER and Config. return as dataframe

def run(gen_func, ns, av_deg):
    """Perform all analysis on graph
    Parameters  
    ----------                  
    gen_func : function
        Function to generate graph
    ns : array
        Array of number of nodes
    av_deg : int
        Average degree
    Returns
    -------     
    df : dataframe
        Dataframe containing results"""
    
    final_df = pd.DataFrame(columns=["N","1/ln(z)", "1/ln(z) err", "Beta", 
                                "Beta err", "rchi", "pearson r","pearson p-val",
                                "spearmans r","spearmans p-val"])    
    for n in ns:
        g = gen_func(n, av_deg)
        k, c, a, a_err, b, b_err, rchi, r, rp, rs, rsp = process(g, to_print=False)
        temp_df = pd.DataFrame({"N": n, "1/ln(z)": a, "1/ln(z) err": a_err, "Beta": b, 
                        "Beta err": b_err, "rchi": rchi, "pearson r": r,
                         "pearson p-val": rp, "spearmans r": rs, "spearmans p-val": rsp}, index=[n])
        final_df = pd.concat([final_df, temp_df])
    return final_df
"""
ns = [1000,2000,4000,8000,16000]
av_degree = 10

df_BA = run(BA, ns, av_degree)
print('BA done')
print(df_BA)
df_ER = run(ER, ns, av_degree)
print('ER done')
print(df_ER)
"""
# Load in unipartite and run for each real networks
# Need to get column names for each network from the dataframe
unipartite = pd.read_pickle('Data/unipartite.pkl')
uni_network_names = unipartite.columns.values.tolist()

def load_graph(name):
    """Load graph from pickle
    Parameters  
    ----------                  
    name : string
        Name of graph
    Returns
    -------     
    g : graph_tool graph
        Loaded graph"""
    g = collection.ns[name]
    return g

def run_real(names):
    """Perform all analysis on graph
    Parameters  
    ----------                  
    name : string
        Name of graph
    Returns
    -------     
    df : dataframe
        Dataframe containing results"""
    final_df = pd.DataFrame(columns=["N","1/ln(z)", "1/ln(z) err", "Beta",
                                "Beta err", "rchi", "pearson r","pearson p-val",
                                "spearmans r","spearmans p-val"])
    num = len(names)
    for i in tqdm.tqdm((range(num))):
        g = load_graph(names[i])
        k, c, a, a_err, b, b_err, rchi, r, rp, rs, rsp = process(g, to_print=False)
        temp_df = pd.DataFrame({"N": len(g.get_vertices()), "1/ln(z)": a, "1/ln(z) err": a_err, 
                            "Beta": b, "Beta err": b_err, "rchi": rchi, "pearson r": r,
                            "pearson p-val": rp, "spearmans r": rs, "spearmans p-val": rsp}, index=[names[i]])
        final_df = pd.concat([final_df, temp_df])
    return final_df

df = run_real(uni_network_names[0:5])
print(df)