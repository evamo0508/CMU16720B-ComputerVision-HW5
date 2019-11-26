import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

train_data = scipy.io.loadmat('../data/nist36_train.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']
train_x = np.expand_dims(train_x.reshape((-1, 32, 32)), axis=1)

max_iters = 80
# pick a batch size, learning rate
batch_size = 32
learning_rate = 1e-2
hidden_size = 64

# dataloader
train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x).type(torch.float32), torch.from_numpy(train_y).type(torch.LongTensor)), batch_size=batch_size, shuffle=True)

# network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, 1, 1)
        self.conv2 = nn.Conv2d(2, 4, 3, 1, 1)
        self.fc1 = nn.Linear(8*8*4, 36)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 8*8*4)
        x = self.fc1(x)
        return x

# model & optimizer
model = Net()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# training
train_loss, train_acc = [], []
for itr in range(max_iters):
    total_loss, total_acc = 0, 0
    for xb, yb in train_loader:
        y_probs = model(xb)
        y_true = torch.argmax(yb, dim=1)
        y_pred = torch.argmax(y_probs, dim=1)
        loss = F.cross_entropy(y_probs, y_true) # do softmax and nll_loss

        total_loss += loss.item()
        total_acc += y_pred.eq(y_true.data).cpu().sum().item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    train_loss.append(total_loss)
    total_acc /= train_x.shape[0]
    train_acc.append(total_acc)
    if itr % 2 == 0 or itr == max_iters - 1:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

import matplotlib.pyplot as plt
# plot acc
plt.plot(train_acc)
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')
plt.show()

# plot loss
plt.plot(train_loss)
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')
plt.show()


