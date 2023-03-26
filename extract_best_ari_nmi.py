import numpy as np
# np.set_printoptions(threshold=100000)
from sklearn.preprocessing import normalize
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.special import softmax
from sklearn import cluster, metrics
import functools
from scipy import stats
import scipy
import os

def isNumOrDot(c):
    if ('0' <= c) and (c <= '9'):
        return True
    if c == '.' or c == '-' or c == 'e':
        return True
    return False

def extrac_number(input):
    # print(input)
    a = 0
    while not isNumOrDot(input[a]):
        a += 1
    b = a
    while isNumOrDot(input[b]):
        b += 1
    return float(input[a:b])

def read_init_ari_nmi(fn, target):
    init_ari = -1
    init_nmi = -1
    if not os.path.exists(fn):
        return init_ari, init_nmi
    with open(fn, 'r') as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            if line[:len(target)] == target and line.find('nmi=') >= 0:
                init_ari = extrac_number(line[line.find('ari=')+len('ari='):])
                init_nmi = extrac_number(line[line.find('nmi=')+len('nmi='):])
                break
    return init_ari, init_nmi

def read_best_ari_nmi(fn, target):
    best_ari = -1
    best_nmi = -1
    best_nmi_ind = -1
    if not os.path.exists(fn):
        return best_ari, best_nmi
    with open(fn, 'r') as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            if line[:len(target)] == target and line.find('best_nmi_iter=') >= 0:
                best_nmi_ind = extrac_number(line[line.find('best_nmi_iter=')+len('best_nmi_iter='):])
                break
    best_nmi_ind = int(best_nmi_ind)
    with open(fn, 'r') as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            if line.find('Iter{:02}'.format(best_nmi_ind)) >= 0:
                line = fp.readline()
                best_ari = extrac_number(line[line.find('ari=')+len('ari='):])
                best_nmi = extrac_number(line[line.find('nmi=')+len('nmi='):])
                break
    return best_ari, best_nmi


if __name__ == "__main__":
    rec_dir = 'results/'
    dn_list = ['breastcancer', 'diabetes', 'glass', 'iris', 'orlface10', 'orlface40', 'seeds', 'wine',
               'USPS_CNN_AE', 'STL10_resnet18', 'MNIST_CNN_AE', 'FashionMNIST_CNN_AE', 'CIFAR10_resnet18']
    K_list = [2, 4, 8, 16, 32]
    alpha_list = [0.025, 0.05, 0.075, 0.1]
    method_list = ['GFR', 'GI', 'PIC', 'FGFR', 'FGI', 'FPIC']
    method_list = ['ChebI', 'FChebI']
    for method in method_list:
        for dn in dn_list:
            dn_new = dn
            rec_method_dn_dir = rec_dir + method + '/' + dn_new + '/'
            if os.path.exists(rec_method_dn_dir):
                if method == 'FRI' or method == 'FFRI':
                    best_arr_ari = np.zeros((4, len(K_list)), dtype=float)
                    best_arr_nmi = np.zeros((4, len(K_list)), dtype=float)
                    for K_i in range(len(K_list)):
                        K = K_list[K_i]
                        para_str = 'K({:02})_A({:.4})'.format(K, alpha_list[0])
                        best_arr_ari[0, K_i], best_arr_nmi[0, K_i] = read_init_ari_nmi(rec_method_dn_dir + para_str + '.txt', 'Kmeans:')
                        best_arr_ari[1, K_i], best_arr_nmi[1, K_i] = read_init_ari_nmi(rec_method_dn_dir + para_str + '.txt', 'newSC:')
                        best_arr_ari[2, K_i], best_arr_nmi[2, K_i] = read_init_ari_nmi(rec_method_dn_dir + para_str + '.txt', 'X_low:')
                        for alpha_i in range(len(alpha_list)):
                            alpha = alpha_list[alpha_i]
                            para_str = 'K({:02})_A({:.4})'.format(K, alpha)
                            ari, nmi = read_best_ari_nmi(rec_method_dn_dir + para_str + '.txt', 'BEST')
                            best_arr_ari[3, K_i] = np.maximum(best_arr_ari[3, K_i], ari)
                            best_arr_nmi[3, K_i] = np.maximum(best_arr_nmi[3, K_i], nmi)
                    np.savetxt(rec_method_dn_dir+'best_ari_K.txt', best_arr_ari, fmt='%.4f', delimiter='\t')
                    np.savetxt(rec_method_dn_dir + 'best_nmi_K.txt', best_arr_nmi, fmt='%.4f', delimiter='\t')
                else:
                    best_arr_ari = np.zeros((1, len(K_list)), dtype=float)
                    best_arr_nmi = np.zeros((1, len(K_list)), dtype=float)
                    for K_i in range(len(K_list)):
                        K = K_list[K_i]
                        para_str = 'K({:02})'.format(K)
                        best_arr_ari[0, K_i], best_arr_nmi[0, K_i] = read_best_ari_nmi(rec_method_dn_dir + para_str + '.txt', 'BEST')
                    np.savetxt(rec_method_dn_dir+'best_ari_K.txt', best_arr_ari, fmt='%.4f', delimiter='\t')
                    np.savetxt(rec_method_dn_dir + 'best_nmi_K.txt', best_arr_nmi, fmt='%.4f', delimiter='\t')
            dn_new = dn + '_zscore'
            rec_method_dn_dir = rec_dir + method + '/' + dn_new + '/'
            if os.path.exists(rec_method_dn_dir):
                if method == 'FRI' or method == 'FFRI':
                    best_arr_ari = np.zeros((4, len(K_list)), dtype=float)
                    best_arr_nmi = np.zeros((4, len(K_list)), dtype=float)
                    for K_i in range(len(K_list)):
                        K = K_list[K_i]
                        para_str = 'K({:02})_A({:.4})'.format(K, alpha_list[0])
                        best_arr_ari[0, K_i], best_arr_nmi[0, K_i] = read_init_ari_nmi(rec_method_dn_dir + para_str + '.txt', 'Kmeans:')
                        best_arr_ari[1, K_i], best_arr_nmi[1, K_i] = read_init_ari_nmi(rec_method_dn_dir + para_str + '.txt', 'newSC:')
                        best_arr_ari[2, K_i], best_arr_nmi[2, K_i] = read_init_ari_nmi(rec_method_dn_dir + para_str + '.txt', 'X_low:')
                        for alpha_i in range(len(alpha_list)):
                            alpha = alpha_list[alpha_i]
                            para_str = 'K({:02})_A({:.4})'.format(K, alpha)
                            ari, nmi = read_best_ari_nmi(rec_method_dn_dir + para_str + '.txt', 'BEST')
                            best_arr_ari[3, K_i] = np.maximum(best_arr_ari[3, K_i], ari)
                            best_arr_nmi[3, K_i] = np.maximum(best_arr_nmi[3, K_i], nmi)
                    np.savetxt(rec_method_dn_dir+'best_ari_K.txt', best_arr_ari, fmt='%.4f', delimiter='\t')
                    np.savetxt(rec_method_dn_dir + 'best_nmi_K.txt', best_arr_nmi, fmt='%.4f', delimiter='\t')
                else:
                    best_arr_ari = np.zeros((1, len(K_list)), dtype=float)
                    best_arr_nmi = np.zeros((1, len(K_list)), dtype=float)
                    for K_i in range(len(K_list)):
                        K = K_list[K_i]
                        para_str = 'K({:02})'.format(K)
                        best_arr_ari[0, K_i], best_arr_nmi[0, K_i] = read_best_ari_nmi(rec_method_dn_dir + para_str + '.txt', 'BEST')
                    np.savetxt(rec_method_dn_dir+'best_ari_K.txt', best_arr_ari, fmt='%.4f', delimiter='\t')
                    np.savetxt(rec_method_dn_dir + 'best_nmi_K.txt', best_arr_nmi, fmt='%.4f', delimiter='\t')