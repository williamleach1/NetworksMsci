# NetworksMsci

get_networks.py used to access Netzschleuder API
    and then filter and unpack features.
It produces two data files for Unipartite and Bipartite
    graphs to be used to download wit graph_tool later
No unusual dependencies

ProcessBase.py used to process artificial and eventually
    real world networks and complete stats, fitting and
    eventually plots.

TO DO - Base:
* config-BA model needs adding and analysing.
* Plots and data-collapse needs adding.
* fix reduced chi-squared
* Handling repeats for artificial - is it neccesary?
* Finish pipeline to download real networks and process them.
* Investigate beta compared to 

TO DO - Bipartite:
* New fit-func for bipartite.
* New closeness finding implementation for bipartite
* Find and implement bipartite artificial networks
* Process artificial and real bipartite networks

TO DO - Second Degree vs Closeness:
* New fit-func for second degree.
* New second degree (and closeness) finding function for
    second degree vs closeness.
* Process artificial and real world networks for second
    degree vs closeness.

Extensions and Investiagtions (side quests):
* Investigate Branching ratio numerically
* Investigate Hard-Cut off numerically

Installing graph-tool
=====================
1.) Install conda on MacOs from https://docs.conda.io/en/latest/miniconda.html
2.) Install graph-tool with conda as in https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions#macos-x
3.) When in terminal should have (base) in front of the path - running conda activate gt will change this to (gt) which is the environment for graph-tool
4.) Need to install other packages within the environment (once activated) with -> conda install <package>
    e.g. conda install numpy

    