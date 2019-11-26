import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 48
learning_rate = 5e-3
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(1024, hidden_size, params, 'layer1')
initialize_weights(hidden_size, 36, params, 'output')


# with default settings, you should get loss < 150 and accuracy > 80%
train_loss, train_acc = [], []
valid_loss, valid_acc = [], []
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        # forward
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params,'output', softmax) #5x4
        # loss
        # be sure to add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc
        # backward
        delta1 = probs - yb
        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2, params, 'layer1', sigmoid_deriv)
        # apply gradient
        for k,v in sorted(list(params.items())):
            if 'grad' in k:
                name = k.split('_')[1]
                params[name] -= learning_rate * v
    total_loss /= batch_num
    total_acc /= batch_num
    train_loss.append(total_loss)
    train_acc.append(total_acc)

    # validation
    h1 = forward(valid_x, params, 'layer1')
    probs = forward(h1, params,'output', softmax)
    loss, acc = compute_loss_and_acc(valid_y, probs)
    valid_loss.append(loss / batch_num)
    valid_acc.append(acc)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

import matplotlib.pyplot as plt
# plot acc
plt.plot(train_acc)
plt.plot(valid_acc)
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

# plot loss
plt.plot(train_loss)
plt.plot(valid_loss)
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

# run on validation set and report accuracy! should be above 75%
h1 = forward(valid_x, params, 'layer1')
probs = forward(h1, params,'output', softmax)
loss, valid_acc = compute_loss_and_acc(valid_y, probs)
print('Validation accuracy: ',valid_acc)

# run on test set and report accuracy with best network!
test_data = scipy.io.loadmat('../data/nist36_test.mat')
test_x, test_y = test_data['test_data'], test_data['test_labels']
h1 = forward(test_x, params, 'layer1')
test_probs = forward(h1, params,'output', softmax)
loss, test_acc = compute_loss_and_acc(test_y, test_probs)
print('Test accuracy: ', test_acc)

if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here
# original
init_params = {}
initialize_weights(1024, hidden_size, init_params, 'layer1')
fig = plt.figure()
grid = ImageGrid(fig, 111, (8, 8)) # 64 hidden layers
for i in range(hidden_size):
    grid[i].imshow(init_params['Wlayer1'][:, i].reshape((32, 32))) # 1024 wieghts per hidden layer
plt.title('Original Wlayer1')
plt.show()

# learned
with open('q3_weights.pickle', 'rb') as handle:
    saved_params = pickle.load(handle)
saved_Wlayer1 = saved_params['Wlayer1']
fig = plt.figure()
grid = ImageGrid(fig, 111, (8, 8))
for i in range(hidden_size):
    grid[i].imshow(saved_params['Wlayer1'][:, i].reshape((32, 32)))
plt.title('Learned Wlayer1')
plt.show()

# Q3.1.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
y_true = np.argmax(test_y, axis=1)
y_pred = np.argmax(test_probs, axis=1)
for i in range(y_true.shape[0]):
    confusion_matrix[y_true[i], y_pred[i]] += 1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
