'''
Placeholder file for the Octopus process. Need to load in and run with a range of parameters
--> mostly interested in gamma (for powerlaw of degree distribution)
Also plot graph to see if can spot clusters that develop

'''
import ProcessBase as pb
import graph_tool.all as gt
import numpy as np
import matplotlib.pyplot as plt
from graph_tool import generation
import powerlaw as pl

def BA_octo(N,m,gamma):
    # Create empty graph
    # p \propto k^-gamma
    # m = number of edges to attach from new node to existing nodes
    # N = number of nodes
    # gamma = 3 + c/m
    # c = (gamma-3)*m
    g = generation.price_network(N, m, directed = False)
    return g

g = BA_octo(20000,4,2)

# Now plot the degree distribution
# First get the degree distribution
k = g.get_total_degrees(g.get_vertices())
# use powerlaw package to fit the degree distribution
fit = pl.Fit(k, discrete = True, fit_method = 'KS', xmin = 20)
# plot the degree distribution
fit.plot_pdf(color = 'b', linewidth = 2, ax = plt.gca(), label = 'data')
# plot the powerlaw fit
fit.power_law.plot_pdf(color = 'r', linestyle = '--', ax = plt.gca(), label = 'power law fit')
plt.legend()
plt.show()
# print the fit parameters
print(fit.power_law.alpha)
print(fit.power_law.sigma)
print(fit.power_law.xmin)
# print the goodness of fit
print('vs lognormal', fit.distribution_compare('power_law', 'lognormal'))
print('vs exp ', fit.distribution_compare('power_law', 'exponential'))
print('vs truncated power law', fit.distribution_compare('power_law', 'truncated_power_law'))
print('vs stretched exp', fit.distribution_compare('power_law', 'stretched_exponential'))
print('vs lognormal positive' ,fit.distribution_compare('power_law', 'lognormal_positive'))


