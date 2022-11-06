from ProcessBase import *
from IPython.display import HTML
# Load in unipartite and run for each real networks
# Need to get column names for each network from the dataframe
unipartite_df = pd.read_pickle('Data/unipartite.pkl')

# Filter out num_vertices>2000000
unipartite_df = unipartite_df.transpose()
unipartite_df = unipartite_df.loc[unipartite_df['num_vertices']<100000,]
unipartite_df = unipartite_df.transpose()
uni_network_names = unipartite_df.columns.values.tolist()
print(len(uni_network_names))
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
    for i in tqdm((range(num))):
        try:
            g = load_graph(names[i])
            k, c, a, a_err, b, b_err, rchi, r, rp, rs, rsp = process(g,func, to_print=False)
            temp_df = pd.DataFrame({"N": len(g.get_vertices()), "1/ln(z)": a, "1/ln(z) err": a_err, 
                                "Beta": b, "Beta err": b_err, "rchi": rchi, "pearson r": r,
                                "pearson p-val": rp, "spearmans r": rs, "spearmans p-val": rsp}, index=[names[i]])
            final_df = pd.concat([final_df, temp_df])
        except OSError:
            print('File not found for: ', names[i])
            pass
        except KeyError:
            print('HTTP Error 401: UNAUTHORIZED \n for', names[i])

    return final_df



df = run_real(uni_network_names)
print(df)

html = df.to_html()
  
# write html to file
text_file = open("Output/index.html", "w")
text_file.write(html)
text_file.close()

HTML(df.to_html(classes='table table-stripped'))
