
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp


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