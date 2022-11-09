# NetworksMsci

Description of scripts
------------------------------------------------------------------------
get_networks.py 
- used to access Netzschleuder API  as JSON to pandas and then filter and unpack features.
- The JSON was triple nested and had to be unpacked to get seperate networks as list.
- It produces two data files for Unipartite and Bipartite graphs
- To be used as reference to download with graph_tool later
- Dataframe produced is saved in Data folder
- Contains useful information about the networks

ProcessBase.py
- Base functions for processing networks
- Functions to load real netorks (given graph_tool name)
- Functions to generate varied artificial networks (currently only ER and BA)
- Statisitics - pearson, spearman, reduced chi2, best fit
- Contains functions to fit for (currenty just unipartite but bipartite and second degree to be added)
- Contains functions to find degree and closeness and find statistics
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


TO DO:
------------------------------------------------------------------------
TO DO - Base:
* config-BA model needs adding and analysing.
* Generate visual of each network?
* Data-collapse needs adding.
* fix reduced chi-squared - DONE - NEED TO VERIFY - slight difference to E&C
* Handling repeats for artificial - is it neccesary?
* Process networks where N>300000
* Compare average (and standard deviation?) length of shortest path predicted to actual value

TO DO - Bipartite:
* New fit-func for bipartite.
* New closeness finding implementation for bipartite? - think not needed but depends on normalisation
* Find and implement bipartite artificial networks
* Process artificial and real bipartite networks - get statistics and fits
* Plot results - k vs 1/c, data collapse
* Compare average (and standard deviation?) length of shortest path predicted to actual value


TO DO - Second Degree vs Closeness:
* New fit-func for second degree.
* New second degree (and closeness) finding function for
    second degree vs closeness.
* Process artificial and real world unipartite networks for second
    degree vs closeness.
* Plot results - k2 vs 1/c, data collapse
* Compare average (and standard deviation?) length of shortest path predicted to actual value


Extensions and Investiagtions (side quests):
* Investigate Branching ratio numerically, compare to average found in fit.
* Justify if constant branching ratio is a good assumption
* Justify if similaities of branches is good assumption and conditions for this
* Investigate exponential growth in nodes reached when traversing rooted tree
* Investigate Hard-Cut off numerically

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
    - 


    