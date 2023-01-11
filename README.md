# NetworksMsci

Description of scripts
------------------------------------------------------------------------
get_networks.py 
- used to access Netzschleuder API  as JSON to pandas and then filter and unpack features.
- The JSON was triple nested and had to be unpacked to get seperate networks as list.
- It produces two data files for Unipartite and Bipartite graphs
- To be used as reference to download with graph_tool later
- Dataframe produced is saved in Data folder
- Contains useful information about the networks, will be apended to results later

ProcessBase.py
- Base functions for processing networks
- Functions to load real netorks (given graph_tool name)
- Functions to generate varied artificial networks
- Statisitics - pearson, spearman, reduced chi2, best fit
- Contains functions to fit for unipartite and bipartite networks
- Contains functions to find degree and closeness and find statistics for unipartite and bipartite networks
- Functions to save dataframs as html
- Function to generate file management tree

Plotter.py
- Functions to plot graph
- Encoded to plot k vs 1/c but can be changed to other labels
- Function to plot standard error and best fit line

ProcessUniArtificial.py
- Process artiifical networks
- Takes input as which model to process
- Also takes model parameter such as number of nodes and desired average degree (which encodes m and p for BA and ER respectively)
- Processes fit and statistics for each artificial network
- Runs commands to plot k vs 1/c and fit line
- Saves dataframes with stats as html in Output folder

ProcessUniReal.py
- Process real networks
- Takes input as list of networks neames to process
- Processes fit and statistics for each real network
- Runs commands to plot k vs 1/c and fit line and save
- Saves dataframe with stats as html in Output folder
- Only runs if not already processed
- Saved in Output folder

ProcessBipartiteReal.py
- Process real bipartite networks
- Takes input as list of networks neames to process
- Processes dual constrained fit and statistics for each real network
- Runs commands to plot k vs 1/c and fit lines and save
- Saves dataframe with stats as html in Output folder

ProcessBipartiteArtificial.py
- 

TO DO:
------------------------------------------------------------------------
TO DO - Base: 
* config-BA model needs adding and analysing.
* Data-collapse needs adding. 
* Process networks where N>300000
* Compare average (and standard deviation?) length of shortest path predicted to actual value
* Compare beta(fit) to predicited value by z_fit

TO DO - Bipartite: 
* Find, implement, and process config-BA-bipartite model
* Plot data collapse
* Compare average (and standard deviation?) length of shortest path predicted to actual value
* Compare alpha fit to predicted by value of z_a(fit) and z_b(fit)

TO DO - Second Degree vs Closeness:
* New fit-func for second degree. ----- IMPORTANT
* New second degree (and closeness) finding function for
    second degree vs closeness.
* Process artificial and real world unipartite networks for second
    degree vs closeness.
* Plot results - k2 vs 1/c, data collapse
* Compare average (and standard deviation?) length of shortest path predicted to actual value
* Compare gamma fit to predicted by value of z_(fit)

TO DO - global network statistics: 
* quantify how well the fit function fits for real networks (proportion)
* Look at correlation between goodness of fit and other network statistics (average degree, average path length, clustering coefficient, small-world coefficint, tree-ness, number loops, assortivity, etc.)
* Potential machine learning to predict goodness of fit from network statistics - could be useful for predicting network structure from network statistics

TO DO - Relation breakdown investigation:
* Test octopus (or star) to see if network structure causes fit to be worse
* Test other models with large scale (and well studied network structure) to see if fit is worse

TO DO - Investigating assumptions (constant branching ratio, similarities of branches, hard cut-off):
* Investigate Branching ratio numerically, compare to average found in fit.
    * Function for this is started
* Justify if constant branching ratio is a good assumption
* Justify if similaities of branches is good assumption and conditions for this
* Investigate exponential growth in nodes reached when traversing rooted tree
* Investigate Hard-Cut off numerically

TO DO - ring analysis:
* Investigate ring analysis in context of mean first passage time
* Can E&C simple approximation be used to find mean first passage time?


Installing graph-tool
=====================
1. Install conda on MacOs from https://docs.conda.io/en/latest/miniconda.html
2. Install graph-tool with conda as in https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions#macos-x
3. When in terminal should have (base) in front of the path - running conda activate gt will change this to (gt) which is the environment for graph-tool
4. Need to install other packages within the environment (once activated) with -> conda install <package>
    e.g. conda install numpy
    - can optionally install pip
    - matplotlib
    - requests
    - tqdm (for progress bar)
    - multiprocessing
    - pandas
    - scipy
    - seaborn
    - ipython
    - networkx
    - openpyxl

    