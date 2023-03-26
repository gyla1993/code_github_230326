# Python 3.6
# Required packages: numpy, scikit-learn, cuml (if gpu acceleration is used).
import numpy as np
from sklearn.preprocessing import normalize
from sklearn import cluster, metrics
import os

def perform_kmeans(x, category, gpu_kemans=False):
    if gpu_kemans: # Acceleration by using gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        from cuml.cluster import KMeans
        clustering = KMeans(n_clusters=category, n_init=10, random_state=0, output_type='numpy').fit(x)
        return clustering.labels_, clustering.cluster_centers_
    else:
        clustering = cluster.KMeans(n_clusters=category, random_state=0).fit(x)
        return clustering.labels_, clustering.cluster_centers_

def calc_score(labels, preds):
    labels = np.squeeze(labels)
    preds = np.squeeze(preds)
    labels = labels.astype(int)
    preds = preds.astype(int)
    ari = metrics.adjusted_rand_score(labels, preds)
    nmi = metrics.normalized_mutual_info_score(labels, preds)
    ami = metrics.adjusted_mutual_info_score(labels, preds)
    completeness = metrics.completeness_score(labels, preds)
    purity = metrics.completeness_score(preds, labels)
    n_cluster = np.max(preds) + 1
    return ari, nmi, ami, completeness, purity, n_cluster

def calc_dist(X, Y):
    X_ = np.sum(X*X, axis=1, keepdims=True)
    Y_ = np.sum(Y*Y, axis=1, keepdims=True)
    dist_2 = X_+Y_.T-2*np.matmul(X, Y.T)
    dist_2[dist_2 < 0] = 0
    return np.sqrt(dist_2/X.shape[1])

def modified_softmax_normalized(E, K):
    shift = 0.0001
    E = E+shift     # To avoid zeros
    ind_Kth = np.argpartition(E, kth=K, axis=1)[:, K-1:K]
    temperature = np.mean(E[np.arange(E.shape[0]).reshape(E.shape[0], 1), ind_Kth])
    S = np.exp((-E)/(temperature+1e-12))
    S_Kth = S[np.arange(S.shape[0]).reshape(S.shape[0], 1), ind_Kth]
    S[S < S_Kth] = 0
    S = normalize(S)
    D = np.matmul(S, np.matmul(S.T, np.ones((S.shape[0], 1))))+1e-10
    S_hat = (1./np.sqrt(D))*S
    return S_hat

def row_normalized_affinity(E, K):
    shift = 0.0001
    E = E+shift     # To avoid zeros
    ind_Kth = np.argpartition(E, kth=K, axis=1)[:, K-1:K]
    temperature = np.mean(E[np.arange(E.shape[0]).reshape(E.shape[0], 1), ind_Kth])
    # print(temperature)
    S = np.exp((-E)/(temperature+1e-12))
    S_Kth = S[np.arange(S.shape[0]).reshape(S.shape[0], 1), ind_Kth]
    S[S < S_Kth] = 0
    S = normalize(S)
    D = np.matmul(S, np.matmul(S.T, np.ones((S.shape[0], 1))))+1e-10
    A_hat = np.matmul((1./D)*S, S.T)
    return A_hat

def low_frequency_components_svd(aff):
    aff = np.nan_to_num(aff)
    U, s, VH = np.linalg.svd(aff, full_matrices=False)
    s = s.reshape((s.shape[0], 1))
    return U, s, VH

def extract_supporting_points(samples, M, noise_threshold):
    pr, ce = perform_kmeans(samples, M)
    flag = np.zeros(M, dtype=bool)
    for i in range(M):
        if np.sum(pr == i) >= noise_threshold:
            flag[i] = True
    return ce[flag, :]

def FGFR(X, K, C, alpha, L, M, noise_threshold, save_path=None, label=None):
    best_nmi = -1
    best_nmi_iter = -1
    if save_path is not None:
        prediction, _ = perform_kmeans(X, C)
        ari, nmi, ami, completeness, purity, n_cluster = calc_score(label, prediction)
        with open(save_path, 'w') as f:
            f.write('Iter{:02}\n'.format(0))
            f.write('Kmeans: ari={:.6} nmi={:.6} ami={:.6} completeness={:.6} purity={:.6}\n'.
                    format(ari, nmi, ami, completeness, purity))

    for l in range(L):
        X = X-np.mean(X, axis=0)
        Y = extract_supporting_points(samples=X, M=M, noise_threshold=noise_threshold)
        E = calc_dist(X, Y)
        S_hat = modified_softmax_normalized(E, K)
        flag = True
        try:
            U, Sigma, VT = low_frequency_components_svd(S_hat)
        except Exception as exc:
            flag = False
        if flag:
            U_low = U[:, :C]
            X_low = np.matmul(U_low, np.matmul(U_low.T, X))
            X_high = X-X_low
            X = (1 + alpha) * X_low + (1 - alpha) * X_high
        if save_path is not None:
            prediction, _ = perform_kmeans(X, C)
            ari, nmi, ami, completeness, purity, n_cluster = calc_score(label, prediction)
            if nmi > best_nmi:
                best_nmi = nmi
                best_nmi_iter = l+1
            with open(save_path, 'a') as f:
                f.write('Iter{:02}\n'.format(l+1))
                f.write('Kmeans: ari={:.6} nmi={:.6} ami={:.6} completeness={:.6} purity={:.6}\n'.
                        format(ari, nmi, ami, completeness, purity))
            prediction, _ = perform_kmeans(U_low, C)
            ari, nmi, ami, completeness, purity, n_cluster = calc_score(label, prediction)
            with open(save_path, 'a') as f:
                f.write('newSC: ari={:.6} nmi={:.6} ami={:.6} completeness={:.6} purity={:.6}\n'.
                        format(ari, nmi, ami, completeness, purity))
            prediction, _ = perform_kmeans(X_low, C)
            ari, nmi, ami, completeness, purity, n_cluster = calc_score(label, prediction)
            with open(save_path, 'a') as f:
                f.write('X_low: ari={:.6} nmi={:.6} ami={:.6} completeness={:.6} purity={:.6}\n'.
                        format(ari, nmi, ami, completeness, purity))
    if save_path is not None:
        with open(save_path, 'a') as f:
            f.write('BEST: best_nmi={:.6} best_nmi_iter={}\n'.format(best_nmi, best_nmi_iter))
    return X, best_nmi, best_nmi_iter

def FGF(X, K, C, L, M, noise_threshold, save_path=None, label=None):
    best_nmi = -1
    best_nmi_iter = -1
    X = X - np.mean(X, axis=0)
    Y = extract_supporting_points(samples=X, M=M, noise_threshold=noise_threshold)
    E = calc_dist(X, Y)
    S_hat = modified_softmax_normalized(E, K)
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write('')
    for l in range(L):
        X = np.matmul(S_hat, np.matmul(S_hat.T, X))
        if save_path is not None:
            prediction, _ = perform_kmeans(X, C)
            ari, nmi, ami, completeness, purity, n_cluster = calc_score(label, prediction)
            if nmi > best_nmi:
                best_nmi = nmi
                best_nmi_iter = l + 1
            with open(save_path, 'a') as f:
                f.write('Iter{:02}\n'.format(l + 1))
                f.write('Kmeans: ari={:.6} nmi={:.6} ami={:.6} completeness={:.6} purity={:.6}\n'.
                        format(ari, nmi, ami, completeness, purity))
    if save_path is not None:
        with open(save_path, 'a') as f:
            f.write('BEST: best_nmi={:.6} best_nmi_iter={}\n'.format(best_nmi, best_nmi_iter))
    return X, best_nmi, best_nmi_iter

def FPIC(X, K, C, L, M, noise_threshold, save_path=None, label=None):
    best_nmi = -1
    best_nmi_iter = -1
    X = X - np.mean(X, axis=0)
    Y = extract_supporting_points(samples=X, M=M, noise_threshold=noise_threshold)
    E = calc_dist(X, Y)
    S_hat, S = row_normalized_affinity(E, K)
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write('')
    v = np.matmul(S_hat, np.matmul(S.T, np.ones((X.shape[0], 1), dtype=float)))
    v = v / np.linalg.norm(v, ord=1)
    for l in range(L):
        v = np.matmul(S_hat, np.matmul(S.T, v))
        v = v/np.linalg.norm(v, ord=1)
        if save_path is not None:
            prediction, _ = perform_kmeans(v, C)
            ari, nmi, ami, completeness, purity, n_cluster = calc_score(label, prediction)
            if nmi > best_nmi:
                best_nmi = nmi
                best_nmi_iter = l+1
            with open(save_path, 'a') as f:
                f.write('Iter{:02}\n'.format(l+1))
                f.write('Kmeans: ari={:.6} nmi={:.6} ami={:.6} completeness={:.6} purity={:.6}\n'.
                        format(ari, nmi, ami, completeness, purity))
    if save_path is not None:
        with open(save_path, 'a') as f:
            f.write('BEST: best_nmi={:.6} best_nmi_iter={}\n'.format(best_nmi, best_nmi_iter))
    return X, best_nmi, best_nmi_iter

if __name__ == "__main__":
    method_opt = 'GFR'  # 'GFR' or 'GF' or 'PIC'.
                        # The clustering results of KM++, LF and CSC are obtained as intermediate results of GFR.
    rec_dir = 'results/{}/'.format(method_opt)
    K_list = [4, 8, 16, 32]
    dn_list = ['USPS_CNN_AE', 'MNIST_CNN_AE', 'FMNIST_CNN_AE', 'STL10_ResNet18', 'CIFAR10_resnet18']
    M = 500
    noise_threshold = 3
    for dn in dn_list:
        data_label = np.load('data_npy/' + dn + '.npy')
        label = data_label[:, -1].astype(int)
        data = data_label[:, :-1]
        N = data.shape[0]
        dim = data.shape[1]
        unique_label = np.unique(label)
        C = unique_label.shape[0]
        label_ = np.zeros_like(label)
        for i in range(C):
            label_[label == unique_label[i]] = i
        label = label_
        rec_dn_dir = rec_dir + dn + '/'
        if not os.path.exists(rec_dn_dir):
            os.mkdir(rec_dn_dir)

        if method_opt == 'GFR':
            L = 30
            alpha_list = [0.025, 0.05, 0.075, 0.1]
            for K in K_list:
                for alpha in alpha_list:
                    para_str = 'K({:02})_A({:.4})'.format(K, alpha)
                    FGFR(data, K, C, alpha, L, M, noise_threshold, save_path=rec_dn_dir + para_str + '.txt',
                         label=label)

        if method_opt == 'GF':
            L = 30
            for K in K_list:
                para_str = 'K({:02})'.format(K)
                FGF(data, K, C, L, M, noise_threshold, save_path=rec_dn_dir + para_str + '.txt', label=label)

        if method_opt == 'PIC':
            L = 180
            for K in K_list:
                para_str = 'K({:02})'.format(K)
                FPIC(data, K, C, L, M, noise_threshold, save_path=rec_dn_dir + para_str + '.txt', label=label)












