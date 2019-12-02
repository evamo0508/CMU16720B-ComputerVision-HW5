import numpy as np
import os
import cv2
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

# from nn import *
# from q4 import *
from img_effect import *
from text_effect import *

class GUI(object):
    ind = 0
    def apply(self, event):
        pass
    def check_effects(self, label):
        index = labels.index(label)

def load_img():
    return cv2.imread('images/receipt.jpg')
    #return skimage.img_as_float(skimage.io.imread('images/receipt.jpg'))

def loop(img):
    while True:
        f = parse_input(input())
        img = f(img, bboxes)
        update_img(img)

def parse_input(Input):
    # return function name
    pass

def update_img(img):
    bboxes, bw = find_bboxes(img)
    print(bboxes)

    plt.imshow(bw, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                            fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)

    # Set up GUI
    callback = GUI()
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = matplotlib.widgets.Button(axnext, 'Apply')
    bnext.on_clicked(callback.apply)
    rax = plt.axes([0.05, 0.4, 0.1, 0.15])
    labels = ["Bold", "Italic", "Underline"]
    check = matplotlib.widgets.CheckButtons(rax, labels)
    # check.on_clicked(func)
    plt.show()

if __name__ == "__main__":
    img = load_img()
    img = warp(img)
    #bboxes, bw = find_bboxes(img)
    # im1 = italic(img, bboxes)
    #im1 = bold(img, bboxes)
    #im1 = italic(im1, bboxes)
    plt.imshow(img, cmap='gray')
    plt.show()
