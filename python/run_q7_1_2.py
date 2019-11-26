import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

max_iters = 10
# pick a batch size, learning rate
batch_size = 100
learning_rate = 1e-2

root = './data'
trans = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=batch_size,
                    shuffle=True)
print('==>>> total training batch number: {}'.format(len(train_loader)))

# network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, 1, 1)
        self.conv2 = nn.Conv2d(2, 4, 3, 1, 1)
        self.fc1 = nn.Linear(7*7*4, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 7*7*4)
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
        xb, y_true = Variable(xb), Variable(yb)
        y_probs = model(xb)
        y_pred = torch.argmax(y_probs, dim=1)
        loss = F.cross_entropy(y_probs, y_true)

        total_loss += loss.item()
        total_acc += y_pred.eq(y_true.data).cpu().sum().item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    train_loss.append(total_loss)
    total_acc /= len(train_set)
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
