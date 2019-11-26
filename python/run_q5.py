import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
initialize_weights(1024, hidden_size, params, 'layer1')
initialize_weights(hidden_size, hidden_size, params, 'layer2')
initialize_weights(hidden_size, hidden_size, params, 'layer3')
initialize_weights(hidden_size, 1024, params, 'output')
keys = [x for x in params.keys()]
for key in keys:
    params['m_' + key] = np.zeros(params[key].shape)

# should look like your previous training loops
train_loss = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        # forward
        h1 = forward(xb, params, 'layer1', relu)
        h2 = forward(h1, params, 'layer2', relu)
        h3 = forward(h2, params, 'layer3', relu)
        h4 = forward(h3, params,'output', sigmoid)
        # loss
        # be sure to add loss and accuracy to epoch totals
        total_loss += np.sum((xb - h4) ** 2)
        # backward
        delta1 = -2 * (xb - h4).astype(float)
        delta2 = backwards(delta1, params, 'output', sigmoid_deriv)
        delta3 = backwards(delta2, params, 'layer3', relu_deriv)
        delta4 = backwards(delta3, params, 'layer2', relu_deriv)
        backwards(delta4, params, 'layer1', relu_deriv)
        # apply gradient
        for k,v in sorted(list(params.items())):
            if 'grad' in k:
                name = k.split('_')[1]
                params['m_' + name] = 0.9 * params['m_' + name] - learning_rate * v
                params[name] += params['m_' + name]
    train_loss.append(total_loss)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr, total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# Q5.2 plot training loss curve
import matplotlib.pyplot as plt
# plot loss
plt.plot(train_loss)
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')
plt.show()

# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
h1 = forward(valid_x, params, 'layer1', relu)
h2 = forward(h1, params, 'layer2', relu)
h3 = forward(h2, params, 'layer3', relu)
h4 = forward(h3, params, 'output', sigmoid)

valid_y = valid_data['valid_labels']
classes = np.random.choice(np.arange(36), size=5, replace=False)
import string
letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
for c in classes:
    print("selected class: ", letters[c])
    idx = np.argmax(valid_y, axis=1)
    target_idx = np.where(idx == c)[0]
    for i in range(2):
        plt.subplot(2, 1, 1)
        plt.imshow(valid_x[target_idx[i]].reshape(32, 32).T)
        plt.subplot(2, 1, 2)
        plt.imshow(h4[target_idx[i]].reshape(32, 32).T)
        plt.show()

# Q5.3.2
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
psnr_total = 0
for i in range(valid_x.shape[0]):
    psnr_total += psnr(valid_x[i], h4[i])
print("ave PSNR: ", psnr_total / valid_x.shape[0])
