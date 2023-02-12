
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

params =    {'font.size' : 16,
            'axes.labelsize':16,
            'legend.fontsize': 14,
            'xtick.labelsize': 14,
            'ytick.labelsize': 18,
            'axes.titlesize': 16,
            'figure.titlesize': 16,
            'figure.figsize': (12, 9),}
plt.rcParams.update(params)
# class to plot in groups
#           --> options for legends ect - class based?
#           --> Options to save plots
#           --> Options to save data
#          --> Options to add line
class Plotter:
    def __init__(self,name):
        plt.figure(name)
        self.x = []
        self.y = []
        self.xfit = []
        self.yfit = []
        self.labels = []
        self.fitlables = []
        self.colors = []
        self.markers = []
        self.linestyles = []
        self.x_label = r'$k$'
        self.y_label = r'$\dfrac{1}{c}$'
        self.suptitle = 'Inverse Closeness against Degree'
        self.title = name
        self.legend = False
        self.legend_loc = 'best'
        self.legend_title = 'Series:'
        self.fitline = False
    def add_plot(self, x, y, yerr=None, label='Data', fitline=False, function=None, popt=None):
        self.x.append(x)
        self.y.append(y)
        self.labels.append(label)
        if fitline:
            self.fitline = True
            xs_unique = np.unique(x)
            self.xfit.append(xs_unique)
            self.yfit.append(function(xs_unique, *popt))
            self.fitlables.append(label+' fit')
        if yerr is not None:
            plt.errorbar(x, y, yerr=yerr, fmt = '.',markersize = 5,capsize=2,)
    # Options to change for non default plots (collapses etc)
    def change_sup_title(self, title):
        self.suptitle = title
    def change_title(self, title):
        self.title = title
    def change_x_label(self, label):
        self.x_label = label
    def change_y_label(self, label):
        self.y_label = label
    def change_x_lim(self, lim):
        self.x_lim = lim
    def change_y_lim(self, lim):
        self.y_lim = lim
    # plot graphs and fitlines
    def plot(self,scale='log',legend=False,save=False, savename=None):
        for i in range(len(self.x)):
            plt.plot(self.x[i], self.y[i], label=self.labels[i],marker= '.', linestyle='None')
            if self.fitline:
                plt.plot(self.xfit[i], self.yfit[i], color='black', linestyle='--')
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label, rotation=0)
        plt.title(self.title)
        self.legend = legend
        if self.legend:
            plt.legend(loc=self.legend_loc, title=self.legend_title)
        if scale == 'log':
            plt.xscale('log')
        if save:
            plt.savefig(savename)
        plt.close()