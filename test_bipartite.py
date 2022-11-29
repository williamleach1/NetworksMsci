from graph_tool import Graph, centrality, collection, generation, topology
from graph_tool import stats as gt_stats
from ProcessBase import *
import random
import graph_tool.all as gt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from iminuit.util import describe
from scipy.optimize import curve_fit
params =    {'font.size' : 16,
            'axes.labelsize':16,
            'legend.fontsize': 14,
            'xtick.labelsize': 14,
            'ytick.labelsize': 18,
            'axes.titlesize': 16,
            'figure.titlesize': 16,
            'figure.figsize': (16, 12),}
plt.rcParams.update(params)


'''
# Lets try generate a small-bipartite graph

g = Graph()
g.set_directed(False)

N_max = 4000

edge_list = []

for i in range(7,N_max - 6):
    edge_list.append((i, i + 1))
    edge_list.append((i, i + 3))
    edge_list.append((i, i + 5))
    edge_list.append((i, i + 7))
    edge_list.append((i, i - 1))
    edge_list.append((i, i - 3))
    edge_list.append((i, i - 5))
    edge_list.append((i, i - 7))

print(edge_list)
# now rewire edges with probability beta
# only rewire odd to odd and even to even
beta = 0.

for e in edge_list:
    if random.random() > beta:
        if e[0] % 2 == 0:
            # even
            edge_list.remove(e)
            e = (e[0], random.randint(0, N_max//2 - 1)*2 + 1)
            edge_list.append(e)
            
        else:
            # odd
            edge_list.remove(e)
            e = (e[0], random.randint(0, N_max//2 - 1)*2)
            edge_list.append(e)

print(edge_list)



g.add_edge_list(edge_list)
'''



def GetKC_bipartite(g):
    k, c, inv_c, mean_k = GetKC(g)
    # get the bipartite sets
    test, partition = topology.is_bipartite(g, partition=True)
    print(test)
    print(partition.get_array())
    # split into odd and even
    partition_array = partition.get_array()
    # find number of 1s and 0s
    num_1s = 0
    num_0s = 0
    for i in partition_array:
        if i == 1:
            num_1s += 1
        else:
            num_0s += 1
    
    print('breakdown: ',num_1s, num_0s)

    k_1 = k[partition_array == 0]
    c_1 = c[partition_array == 0]
    inv_c_1 = inv_c[partition_array == 0]
    mean_k_1 = np.mean(k_1)

    k_2 = k[partition_array == 1]
    c_2 = c[partition_array == 1]
    inv_c_2 = inv_c[partition_array == 1]
    mean_k_2 = np.mean(k_2)
    return k_1, c_1, inv_c_1, k_2, c_2, inv_c_2, mean_k_1, mean_k_2

#g = load_graph('foursquare/NYC_restaurant_tips')
#g = load_graph('plant_pol_robertson')
#g = load_graph('crime')
#g = load_graph('nematode_mammal')

def Harry_1(k, a, b, alpha):
    return -2*np.log(k*(1+np.exp(b)))/(a+b)+alpha


def Harry_2(k, a, b, alpha):
    return -2*np.log(k*(1+np.exp(a)))/(a+b)+alpha


def fitter_test_dual(k1,k2,inv_c1,inv_c2,funcA,funcB,to_print=False):
    err_est_1 = 0.01#np.sqrt(inv_c1)
    err_est_2 = 0.01#np.sqrt(inv_c2)
    combined_LS = LeastSquares(k1,inv_c1,err_est_1, funcA) + LeastSquares(k2,inv_c2,err_est_2, funcB)
    print(f"{describe(combined_LS)=}")
    m = Minuit(combined_LS, a=1, b=1,alpha=1)
    m.migrad()
    if to_print:
        print(m.values)
    popt = [m.values['a'], m.values['b'], m.values['alpha']]
    errs = [m.errors['a'], m.errors['b'], m.errors['alpha']]
    return popt, errs

def fitter_test_dual_curve_fit(k1,k2,inv_c1,inv_c2,funcA,funcB,to_print=False):
    k = np.append(k1,k2)
    num_k1 = len(k1)
    num_k2 = len(k2)

    inv_c = np.append(inv_c1,inv_c2)

    def combo_func(k,a,b,alpha):
        data1 = funcA(k[:num_k1],a,b,alpha)
        data2 = funcB(k[num_k1:],a,b,alpha)
        return np.append(data1,data2)
    initial_guess = [1,1,1]


    popt, pcov = curve_fit(combo_func, k, inv_c, p0=initial_guess)
    if to_print:
        print(popt)

    errs = np.sqrt(np.diag(pcov))
    return popt, errs




def process(g,to_print=False):

    k_1, c_1, inv_c_1, k_2, c_2, inv_c_2, mean_k_1, mean_k_2 = GetKC_bipartite(g)
    
    popt, errs = fitter_test_dual_curve_fit(k_1,k_2,inv_c_1,inv_c_2, Harry_1, Harry_2, to_print=to_print)
    popt1, errs1 = fitter_test_dual(k_1,k_2,inv_c_1,inv_c_2, Harry_1, Harry_2, to_print=to_print)
    a ,b ,alpha = popt


    statistics_dict_1 = aggregate_dict(k_1, inv_c_1)
    statistics_dict_2 = aggregate_dict(k_2, inv_c_2)

    rchi_1 = red_chi_square(k_1, inv_c_1, Harry_1, popt,statistics_dict_1)
    rchi_2 = red_chi_square(k_2, inv_c_2, Harry_2, popt,statistics_dict_2)


    r1, rp1 = pearson(k_1, c_1)
    r2, rp2 = pearson(k_2, c_2)

    rs1, rsp1 = spearman(k_1, c_1)
    rs2, rsp2 = spearman(k_2, c_2)
    if to_print:
        print("Group 1 Reduced chi square:", rchi_1)
        print("Group 1 Pearson correlation:", r1)
        print("Group 1 Pearson p-value:", rp1)
        print("Group 1 Spearman correlation:", rs1)
        print("Group 1 Spearman p-value:", rsp1)
        print("Group 2 Reduced chi square:", rchi_2)
        print("Group 2 Pearson correlation:", r2)
        print("Group 2 Pearson p-value:", rp2)
        print("Group 2 Spearman correlation:", rs2)
        print("Group 2 Spearman p-value:", rsp2)
        print("a:", a)
        print("a error:", errs[0])
        print("b:", b)
        print("b error:", errs[1])
        print("alpha:", alpha)
        print("alpha error:", errs[2])


    output = [k_1, c_1, inv_c_1, k_2, c_2, inv_c_2, mean_k_1, mean_k_2, 
                rchi_1, rchi_2, r1, r2, rs1, rs2, rp1, rp2, rsp1, rsp2, 
                popt, errs, statistics_dict_1, statistics_dict_2]

    return output

Bipartite_df = pd.read_pickle('Data/bipartite.pkl')
upper_node_limit = 50000 # takes around 1 minute per run with 50000
# Filter out num_vertices>2000000
Bipartite_df = filter_num_verticies(Bipartite_df, upper_node_limit)
bipartite_network_names = Bipartite_df.columns.values.tolist()



for i in range(len(bipartite_network_names)):
    print('----------------------------------------------------------------')
    print(bipartite_network_names[i])
    print('----------------------------------------------------------------')
    g = load_graph(bipartite_network_names[i])
    output = process(g, to_print=True)
    
    # plot the odd and even
    k_1 = output[0]
    c_1 = output[1]
    inv_c_1 = output[2]
    k_2 = output[3]
    c_2 = output[4]
    inv_c_2 = output[5]
    mean_k_1 = output[6]
    mean_k_2 = output[7]
    rchi_1 = output[8]
    rchi_2 = output[9]
    r1 = output[10]
    r2 = output[11]
    rs1 = output[12]
    rs2 = output[13]
    rp1 = output[14]
    rp2 = output[15]
    rsp1 = output[16]
    rsp2 = output[17]
    popt = output[18]
    errs = output[19]
    statistics_dict_1 = output[20]
    statistics_dict_2 = output[21]


    ks_1, inv_c_mean_1, errs_1, stds_1, counts_1   = unpack_stat_dict(statistics_dict_1)
    ks_2, inv_c_mean_2, errs_2, stds_2, counts_2   = unpack_stat_dict(statistics_dict_2)

    plt.figure()
    #plt.plot(k_1, inv_c_1,'r.', label="Group 1", alpha=0.1)
    #plt.plot(k_2, inv_c_2,'b.', label="Group 2", alpha=0.1)

    plt.errorbar(ks_1, inv_c_mean_1, yerr=errs_1, fmt='.' ,markersize = 5,capsize=2,color='black')
    plt.plot(ks_1, inv_c_mean_1,'ro', label="Group 1 mean")


    plt.errorbar(ks_2, inv_c_mean_2, yerr=errs_2, fmt='.' ,markersize = 5,capsize=2,color='black')
    plt.plot(ks_2, inv_c_mean_2,'bo', label="Group 2 mean")

    plt.plot(k_1, Harry_1(k_1, *popt),'r', label="Group 1 fit")
    plt.plot(k_2, Harry_2(k_2, *popt),'b', label="Group 2 fit")
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("1/c")
    plt.xscale("log")
    plt.suptitle(bipartite_network_names[i])
    plt.title("a = %2f, b = %2f, alpha = %2f, rchi1=%2f, rchi2=%2f" % (popt[0], popt[1], popt[2], rchi_1, rchi_2))
    
    folder = 'Output/RealBipartiteNets/'+bipartite_network_names[i]+'/'

    plt.savefig('plots/'+str(np.round(rchi_1,3))+'_'+str(np.round(rchi_2,3))+'inv_c_vs_k.png')
    plt.savefig(folder+'inv_c_vs_k.png')
    df = pd.DataFrame({'mean k 1:': [mean_k_1], 'mean k 2:': [mean_k_2], 'rchi 1:': [rchi_1], 
                        'rchi 2:': [rchi_2], 'r 1:': [r1], 'r 2:': [r2], 'rs 1:': [rs1], 
                        'rs 2:': [rs2], 'rp 1:': [rp1], 'rp 2:': [rp2], 'rsp 1:': [rsp1],
                        'rsp 2:': [rsp2], 'a:': [popt[0]], 'a error:': [errs[0]], 'b:': [popt[1]], 
                        'b error:': [errs[1]], 'alpha:': [popt[2]], 'alpha error:': [errs[2]]})
    df.to_html(folder+'stats.html')
    






