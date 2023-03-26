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

def read_init_nmi(fn, target):
    init_nmi = -1
    if not os.path.exists(fn):
        return init_nmi
    with open(fn, 'r') as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            if line[:len(target)] == target and line.find('nmi=') >= 0:
                init_nmi = extrac_number(line[line.find('nmi=')+len('nmi='):])
                break
    return init_nmi

def read_best_nmi(fn, target):
    best_nmi = -1
    if not os.path.exists(fn):
        return best_nmi
    with open(fn, 'r') as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            if line[:len(target)] == target and line.find('best_nmi=') >= 0:
                best_nmi = extrac_number(line[line.find('best_nmi=')+len('best_nmi='):])
    return best_nmi

if __name__ == "__main__":
    rec_dir = 'results/'
    dn_list = ['breastcancer', 'diabetes', 'glass', 'iris', 'orlface10', 'orlface40', 'seeds', 'wine',
               'USPS_CNN_AE', 'STL10_resnet18', 'MNIST_CNN_AE', 'FashionMNIST_CNN_AE', 'CIFAR10_resnet18']
    K_list = [2, 4, 8, 16, 32]
    alpha_list = [0.025, 0.05, 0.075, 0.1]
    method_list = ['GFR', 'GI', 'PIC', 'FGFR', 'FGI', 'FPIC']
    for method in method_list:
        for dn in dn_list:
            dn_new = dn
            rec_method_dn_dir = rec_dir + method + '/' + dn_new + '/'
            if os.path.exists(rec_method_dn_dir):
                if method == 'FRI' or method == 'FFRI':
                    best_arr = np.zeros((len(alpha_list) + 3, len(K_list)), dtype=float)
                    for K_i in range(len(K_list)):
                        K = K_list[K_i]
                        para_str = 'K({:02})_A({:.4})'.format(K, alpha_list[0])
                        best_arr[0, K_i] = read_init_nmi(rec_method_dn_dir + para_str + '.txt', 'Kmeans:')
                        best_arr[1, K_i] = read_init_nmi(rec_method_dn_dir + para_str + '.txt', 'newSC:')
                        best_arr[2, K_i] = read_init_nmi(rec_method_dn_dir + para_str + '.txt', 'X_low:')
                        for alpha_i in range(len(alpha_list)):
                            alpha = alpha_list[alpha_i]
                            para_str = 'K({:02})_A({:.4})'.format(K, alpha)
                            best_arr[alpha_i + 3, K_i] = read_best_nmi(rec_method_dn_dir + para_str + '.txt', 'BEST')
                    np.savetxt(rec_method_dn_dir+'best_nmi_summary.txt', best_arr, fmt='%.4f', delimiter='\t')
                else:
                    best_arr = np.zeros((1, len(K_list)), dtype=float)
                    for K_i in range(len(K_list)):
                        K = K_list[K_i]
                        para_str = 'K({:02})'.format(K)
                        best_arr[0, K_i] = read_best_nmi(rec_method_dn_dir + para_str + '.txt', 'BEST')
                    np.savetxt(rec_method_dn_dir + 'best_nmi_summary.txt', best_arr, fmt='%.4f', delimiter='\t')
            dn_new = dn + '_zscore'
            rec_method_dn_dir = rec_dir + method + '/' + dn_new + '/'
            if os.path.exists(rec_method_dn_dir):
                if method == 'FRI' or method == 'FFRI':
                    best_arr = np.zeros((len(alpha_list)+3, len(K_list)), dtype=float)
                    for K_i in range(len(K_list)):
                        K = K_list[K_i]
                        para_str = 'K({:02})_A({:.4})'.format(K, alpha_list[0])
                        best_arr[0, K_i] = read_init_nmi(rec_method_dn_dir+para_str+'.txt', 'Kmeans:')
                        best_arr[1, K_i] = read_init_nmi(rec_method_dn_dir + para_str + '.txt', 'newSC:')
                        best_arr[2, K_i] = read_init_nmi(rec_method_dn_dir + para_str + '.txt', 'X_low:')
                        for alpha_i in range(len(alpha_list)):
                            alpha = alpha_list[alpha_i]
                            para_str = 'K({:02})_A({:.4})'.format(K, alpha)
                            best_arr[alpha_i+3, K_i] = read_best_nmi(rec_method_dn_dir+para_str+'.txt', 'BEST')
                    np.savetxt(rec_method_dn_dir+'best_nmi_summary.txt', best_arr, fmt='%.4f', delimiter='\t')
                else:
                    best_arr = np.zeros((1, len(K_list)), dtype=float)
                    for K_i in range(len(K_list)):
                        K = K_list[K_i]
                        para_str = 'K({:02})'.format(K)
                        best_arr[0, K_i] = read_best_nmi(rec_method_dn_dir + para_str + '.txt', 'BEST')
                    np.savetxt(rec_method_dn_dir + 'best_nmi_summary.txt', best_arr, fmt='%.4f', delimiter='\t')