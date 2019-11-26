import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dset
import torch.nn.functional as F
from torchvision.datasets import ImageFolder

max_iters = 10

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
                                             batch_size=32,
                                             shuffle=True,
                                             num_workers=4)

# model & optimizer
model = models.squeezenet1_0(pretrained=True)
model.classifier[1] = nn.Conv2d(512, len(train_dataset.classes),
                                kernel_size=(1, 1), stride=(1, 1))
model.num_classes = len(train_dataset.classes)

# stage 1
for param in model.features.parameters():
    param.requires_grad = False
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

model.train()
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

# stage 2
for param in model.parameters():
    param.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=1e-5)

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
