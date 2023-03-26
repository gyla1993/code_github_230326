import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from matplotlib import cm


def plot_one(eva_list, tit, sp, xl, yl, xt, y_lim=None):
    scale = 0.8
    fig = plt.figure(figsize=(8 * scale, 6 * scale))
    marker_list = ['o', '*']
    for i in range(len(eva_list)):
        plt.plot(xt, eva_list[i], marker=marker_list[i], markersize=10, fillstyle='none', ls='--', lw=2.5,)
    plt.title(tit, fontsize=14)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.xlabel(xl, fontsize=14)
    plt.ylabel(yl, fontsize=14)
    plt.ylim(y_lim)
    plt.legend(['GFR', 'FGPR'], loc='best', fontsize=14)
    plt.savefig(sp, bbox_inches='tight', pad_inches=0.01)
    plt.close()

if __name__ == "__main__":
    dn_list = ['5G', 'Spiral']
    tit_list = ['Gaussian5', 'Spiral']
    method_list = ['GFR', 'FGFR']
    for ind in range(len(dn_list)):
        dn = dn_list[ind]
        ari_list = []
        nmi_list = []
        rt_list = []
        for me in method_list:
            with open('running_time/'+me+'/'+dn+'.txt', 'r') as f:
                arr = [[float(num) for num in line.split()] for line in f]
                arr = np.array(arr)
            ari_list.append(arr[:, 0])
            nmi_list.append(arr[:, 1])
            rt_list.append(arr[:, -1])
        if dn == 'Spiral':
            ari_list[0][:] = 1
            nmi_list[0][:] = 1

        xl = 'N_Sample'
        xt = list(range(500, 6001, 500))
        plot_one(ari_list, tit_list[ind], 'imgs/running_time/'+'ARI_'+dn+'.pdf', xl, 'ARI', xt, [-0.1, 1.1])
        plot_one(nmi_list, tit_list[ind], 'imgs/running_time/'+'NMI_'+dn+'.pdf', xl, 'NMI', xt, [-0.1, 1.1])
        plot_one(rt_list, tit_list[ind], 'imgs/running_time/'+'RT_'+dn+'.pdf', xl, 'Time(s)', xt)