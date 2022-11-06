from ProcessBase import *
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
        k, c, a, a_err, b, b_err, rchi, r, rp, rs, rsp = process(g,func, to_print=False)
        temp_df = pd.DataFrame({"N": n, "1/ln(z)": a, "1/ln(z) err": a_err, "Beta": b, 
                        "Beta err": b_err, "rchi": rchi, "pearson r": r,
                         "pearson p-val": rp, "spearmans r": rs, "spearmans p-val": rsp}, index=[n])
        final_df = pd.concat([final_df, temp_df])
    return final_df

ns = [1000,2000,4000,8000,16000]
av_degree = 10

df_BA = run(BA, ns, av_degree)
print('BA done')
print(df_BA)
df_ER = run(ER, ns, av_degree)
print('ER done')
print(df_ER)

html_BA = df_BA.to_html()
html_ER = df_ER.to_html()
# write html to file
text_file = open("Output/index_BA.html", "w")
text_file.write(html_BA)
text_file.close()
text_file = open("Output/index_ER.html", "w")
text_file.write(html_ER)
text_file.close()
