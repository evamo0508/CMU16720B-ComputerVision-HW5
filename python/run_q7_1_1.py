import scipy.io
import torch
from torch.utils.data import DataLoader, TensorDataset

train_data = scipy.io.loadmat('../data/nist36_train.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']

max_iters = 80
# pick a batch size, learning rate
batch_size = 32
learning_rate = 1e-2
hidden_size = 64

# dataloader
train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x).type(torch.float32), torch.from_numpy(train_y).type(torch.LongTensor)), batch_size=batch_size, shuffle=True)

# model
model = torch.nn.Sequential(
            torch.nn.Linear(1024, hidden_size),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_size, 36))

# training
train_loss, train_acc = [], []
for itr in range(max_iters):
    total_loss, total_acc = 0, 0
    for xb, yb in train_loader:
        y_probs = model(xb)
        y_true = torch.argmax(yb, dim=1)
        y_pred = torch.argmax(y_probs, dim=1)
        loss = torch.nn.functional.cross_entropy(y_probs, y_true) # do softmax and nll_loss

        total_loss += loss.item()
        total_acc += y_pred.eq(y_true.data).cpu().sum().item()

        loss.backward()
        with torch.no_grad(): # only considering weights, not gradients
            for param in model.parameters():
                param -= learning_rate * param.grad
        model.zero_grad()
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


