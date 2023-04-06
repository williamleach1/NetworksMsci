import numpy as np
from ProcessBase import *
from scipy.optimize import curve_fit


def R(k_s,N_T,z):
    """Calculate R
    Parameters
    ----------
    k_s : int
        Degree
    N_T : int
        Number of nodes
    z : float
        Scaling parameter
    Returns
    -------
    R : float
        R"""
    ln_z = np.log(z)

    R = ( np.log(N_T-1) + np.log(N_T- k_s) - np.log(k_s) )/ ln_z
    return R

def term_of_sum(l, k_s, N_T, z):
    """Calculate term of sum
    Parameters
    ----------
    k_s : int
        Degree
    N_T : int
        Number of nodes
    z : float
        Scaling parameter
    Returns
    -------
    term_of_sum : float
        term_of_sum"""
    
    A = (N_T- k_s) / (k_s)

    term_of_sum = N_T / (1 + A * z **(-l))
    return term_of_sum

def inv_c_function(k_s, N_T, Z ):
    """Calculate inv_c
    Parameters
    ----------
    k_s : int
        Degree
    N_T : int
        Number of nodes
    Z : float
        Scaling parameter
    Returns
    -------
    inv_c : float
        inv_c"""
    
    R_s = R(k_s, N_T, Z)


    mult_factor = 1/(N_T-1)

    extra = 0

    for l in range(1, int(R_s)-1):
        extra += mult_factor * term_of_sum(l, k_s, N_T, Z)
    extra += mult_factor * term_of_sum(R_s-2, k_s, N_T, Z) * (R_s - int(R_s))

    first = (R_s*(N_T)-k_s)/(N_T)

    inv_c = first - extra
    
    #print(k_s, first, extra)

    return inv_c

def n_l(l,z,av_k,N):
    """Calculate n_l
    Parameters
    ----------
    l : int
        Distance
    z : float
        Scaling parameter
    k : int
        Degree
    L : float
        L_nk
    Returns
    -------
    n_l : float
        n_l"""

    n = term_of_sum(l, av_k, N, z) - term_of_sum(l-1, av_k, N, z)
    return n

g = ER(10000,10)

k, c,popt,pcov, rchi, r, rp, rs, rsp, statistics_dict, mean_k= process(g, 1, Real = True, Name = 'ER' )

inv_ln_z = popt[0]
beta = popt[1]
inv_ln_z_err = np.sqrt(pcov[0][0])
beta_err = np.sqrt(pcov[1][1])
inv_ln_z_u = ufloat(inv_ln_z, inv_ln_z_err)
beta_u = ufloat(beta, beta_err)

ln_z_u = 1/inv_ln_z_u
z_u = umath.exp(ln_z_u)

z = z_u.n

print('Old z: ', z)

N = len(k)


def inv_c_eval(k, N, z):
    return [inv_c_function(k[i], N, z) for i in range(len(k))]

func = lambda k, z: inv_c_eval(k, N, z)


popt, pcov = curve_fit(func, k, 1/c, p0 = [z] )

print('New z: ' , popt)

inv_c = func(k, popt[0])

ks, inv_c_mean, errs, stds, counts   = unpack_stat_dict(statistics_dict)

plt.plot(ks,inv_c_mean, 'o')
plt.plot(k, inv_c, 'o')

plt.xscale('log')

plt.show()

unq_dist, mean_count, std_count, err_count  = process_BFS(g, Real = True, Name = 'ER')


plt.figure()
plt.plot(unq_dist, mean_count, 'ro')

mean_k = np.mean(k)

Rs = []
for i in range(len(k)):
    Rs.append(R(k[i], N, 6.33))

R_max = max(Rs)

print('Predicted Max: ' ,R_max)

print('Actual Max: ', max(unq_dist))

ls = np.arange(1, R_max)

n_ls_list = []

for k_i in k:
    n_ls = []
    for l in ls:
        n_ls.append(n_l(l, 6.33, k_i, N))
    n_ls_list.append(n_ls)

n_ls = np.mean(n_ls_list, axis = 0)

bell_func  = lambda l, z : n_l(l, z, mean_k, N)


popt_bell, pcov_bell = curve_fit(bell_func, unq_dist, mean_count, p0 = 20 )

print('Bell z: ', popt_bell)
print('Bell z_err: ', np.sqrt(pcov_bell[0][0]))

n_ls = bell_func(unq_dist, *popt_bell)

#print(n_ls)
    
plt.plot(unq_dist, n_ls , 'bo')
plt.show()



