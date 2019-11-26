import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from q4 import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

max_iters = 10
# pick a batch size, learning rate
batch_size = 100
learning_rate = 1e-2

root = './data'
dset.EMNIST.url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
trans = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
train_set = dset.EMNIST(root=root, split='balanced', train=True, transform=trans, download=True)
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
        self.fc1 = nn.Linear(7*7*4, 47)

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

torch.save(model.state_dict(), "q7_1_4_model_parameter.pkl")
#checkpoint = torch.load('q7_1_4_model_parameter.pkl')
#model.load_state_dict(checkpoint)

### evaluate on findLetters bounded boxes
for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    # find the rows using..RANSAC, counting, clustering, etc.
    y_centers = [(bbox[2] + bbox[0]) / 2 for bbox in bboxes]
    sorted_bboxes = [x for _, x in sorted(zip(y_centers, bboxes))]
    sorted(y_centers)
    current_y = (sorted_bboxes[0][0] + sorted_bboxes[0][2]) / 2
    count = 1
    rows = []
    row = []
    for i in range(len(sorted_bboxes)):
        if abs(y_centers[i] - current_y) > 0.07 * im1.shape[0]:
            row = sorted(row, key=lambda x: x[1])
            rows.append(row)
            row = [sorted_bboxes[i]]
            count = 1
            current_y = y_centers[i]
        else:
            row.append(sorted_bboxes[i])
            count += 1
            current_y += (y_centers[i] - current_y) / count
    row = sorted(row, key=lambda x: x[1])
    rows.append(row)
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    chars = []
    for row in rows:
        char = []
        for bbox in row:
            y1, x1, y2, x2 = bbox
            crop = bw[y1:y2, x1:x2]
            H, W = crop.shape
            L = max(H, W) * 1.4
            h, w = int((L - H) / 2), int((L - W) / 2)
            crop = np.pad(crop, ((h, h), (w, w)), 'constant', constant_values=(1, 1))
            crop = skimage.morphology.erosion(crop, skimage.morphology.square(5))
            crop = skimage.exposure.adjust_gamma(crop, 5)
            crop = skimage.transform.resize(crop, (28, 28)).T
            char.append(1.0 - crop) #EMNIST works with character in white, background in black
        char = np.stack(char, axis=0)
        chars.append(char)

    # run the crops through your neural network and print them out
    import string
    letters = np.array([str(_) for _ in range(10)] + [_ for _ in string.ascii_uppercase[:26]] + ["a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t"])

    # ground truth
    if img == '01_list.jpg':
        chars_true = ["TODOLIST", "1MAKEATODOLIST", "2CHECKOFFTHEFIRST", "THINGONTODOLIST",
                      "3REALIZEYOUHAVEALREADY", "COMPLETED2THINGS", "4REWARDYOURSELFWITH", "ANAP"]
    elif img == '02_letters.jpg':
        chars_true = ["ABCDEFG", "HIJKLMN", "OPQRSTU", "VWXYZ", "1234567890"]
    elif img == '03_haiku.jpg':
        chars_true = ["HAIKUSAREEASY", "BUTSOMETIMESTHEYDONTMAKESENSE", "REFRIGERATOR"]
    elif img == '04_deep.jpg':
        chars_true = ["DEEPLEARNING", "DEEPERLEARNING", "DEEPESTLEARNING"]

    # predict
    acc, count = 0, 0
    for i, char in enumerate(chars):
        char_torch = []
        for c in char:
            c_torch = trans(np.expand_dims(c, axis=2)).type(torch.float32)
            char_torch.append(c_torch)
        char_torch = torch.stack(char_torch, dim=0)
        y_probs = model(char_torch)
        y_pred = torch.argmax(y_probs, dim=1).numpy()
        char_pred = letters[y_pred]
        char_true = chars_true[i]
        for j, c in enumerate(char_pred):
            if char_true[j] == c.upper():
                acc += 1
        count += len(char_pred)
        print(char_pred)
    print("accuracy: ", acc / count)

