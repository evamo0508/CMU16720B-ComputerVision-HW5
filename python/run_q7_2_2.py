import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder

from q4 import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

max_iters = 20

# data
data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
train_dataset = ImageFolder(root='../data/oxford-flowers17/train',
                                           transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=16,
                                             shuffle=True,
                                             num_workers=4)
print('==>>> total training batch number: {}'.format(len(train_loader)))

# network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, 1, 1)
        self.conv2 = nn.Conv2d(8, 12, 5, 1, 1)
        self.conv3 = nn.Conv2d(12, 15, 5, 1, 1)
        self.fc1 = nn.Linear(26*26*15, 100)
        self.fc2 = nn.Linear(100, 17)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 26*26*15)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# model & optimizer
model = Net()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

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
    total_acc /= len(train_dataset)
    train_acc.append(total_acc)
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
