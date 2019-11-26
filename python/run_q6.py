import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

dim = 32
# do PCA
mean = np.mean(train_x, axis=0)
x = train_x - mean
U, S, V = np.linalg.svd(x)
proj = V[:dim] # 32 * 1024

# rebuild a low-rank version
test_data = scipy.io.loadmat('../data/nist36_test.mat')
test_x = test_data['test_data']
test_mean = test_x - mean
lrank = test_mean @ proj.T # 1800 * 32

# rebuild it
recon = lrank @ proj # 1800 * 1024
recon += mean

# visualize reconstructed test set
idx = np.random.choice(np.arange(test_x.shape[0]), 5, replace=False)

"""
for i in idx:
    plt.subplot(2, 1, 1)
    plt.imshow(test_x[i].reshape(32, 32).T)
    plt.subplot(2, 1, 2)
    plt.imshow(recon[i].reshape(32, 32).T)
    plt.show()
"""
# build valid dataset
recon_valid = (valid_x - mean) @ proj.T @ proj + mean

# visualize the comparison and compute PSNR
import string
letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
classes = np.array([28, 18, 11, 33, 22]) # 2, S, L, 7, W
valid_y = valid_data['valid_labels']
for c in classes:
    print("selected class: ", letters[c])
    idx = np.argmax(valid_y, axis=1)
    target_idx = np.where(idx == c)[0]
    for i in range(2):
        plt.subplot(2, 1, 1)
        plt.imshow(valid_x[target_idx[i]].reshape(32, 32).T)
        plt.subplot(2, 1, 2)
        plt.imshow(recon_valid[target_idx[i]].reshape(32, 32).T)
        plt.show()

# evaluate PSNR
psnr_total = 0
for i in range(valid_x.shape[0]):
    psnr_total += psnr(valid_x[i], recon_valid[i])
print("ave PSNR: ", psnr_total / valid_x.shape[0])

