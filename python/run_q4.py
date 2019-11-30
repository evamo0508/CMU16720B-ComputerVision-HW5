import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.widgets

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *


class Index(object):
    ind = 0

    def next(self, event):
        self.ind += 1
        i = self.ind % len(freqs)
        ydata = np.sin(2*np.pi*freqs[i]*t)
        l.set_ydata(ydata)
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        i = self.ind % len(freqs)
        ydata = np.sin(2*np.pi*freqs[i]*t)
        l.set_ydata(ydata)
        plt.draw()



# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)


    #################
    callback = Index()
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = matplotlib.widgets.Button(axnext, 'Apply')
    bnext.on_clicked(callback.next)
    rax = plt.axes([0.05, 0.4, 0.1, 0.15])
    labels = ["Bold", "Italic", "Underline"]
    check = matplotlib.widgets.CheckButtons(rax, labels)
    #################

    plt.show()
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

            #plt.figure()
            #plt.imshow(crop, cmap='gray')
            #plt.show()

            crop = skimage.transform.resize(crop, (32, 32)).T.flatten()
            char.append(crop)
        char = np.stack(char, axis=0)
        chars.append(char)

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))

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
        h1 = forward(char, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        y_pred = np.argmax(probs, axis=1)
        char_pred = letters[y_pred]
        char_true = chars_true[i]
        for j, c in enumerate(char_pred):
            if char_true[j] == c:
                acc += 1
        count += len(char_pred)
        print(char_pred)
    print("accuracy: ", acc / count)

