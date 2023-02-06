
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
import ProcessBase as pb

g = pb.load_graph('celegans_interactomes/Microarray')

k, c,popt,pcov, rchi, r, rp, rs, rsp, statistics_dict, mean_k= pb.process(g, to_print=False)

ks, inv_c_mean, errs, stds, counts   = pb.unpack_stat_dict(statistics_dict)

plt.figure()
plt.plot(k,c)
plt.show()
