from scipy import stats
import numpy as np

if __name__ == "__main__":
    dn_list = ['USPS_CNN_AE', 'STL10_resnet18', 'MNIST_CNN_AE', 'FashionMNIST_CNN_AE', 'CIFAR10_resnet18']
    for dn in dn_list:
        data_label = np.load('data_npy/'+dn+'.npy')
        data = data_label[:, :-1]
        data = stats.zscore(data, axis=0, ddof=1)
        data_label[:, :-1] = data
        np.save('data_npy/'+dn+'_zscore.npy', data_label)