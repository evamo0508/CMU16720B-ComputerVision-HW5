import numpy as np
import os
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
from text_effect import *

class GUI(object):
    ind = 0
    def apply(self, event):
        pass
    def check_effects(self, label):
        index = labels.index(label)



def load_img():
    return skimage.img_as_float(skimage.io.imread('images/01_list.jpg'))

def warp(img):
    pass

def find_bboxes(img):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions

    # denoise
    denoise = skimage.restoration.denoise_bilateral(img, multichannel=True)
    # greyscale
    grey = skimage.color.rgb2gray(denoise)
    # threshold
    th = skimage.filters.threshold_otsu(grey)
    # morphology
    opening = skimage.morphology.closing(grey < th, skimage.morphology.square(2))
    # remove artifacts connected to image border
    cleared = skimage.segmentation.clear_border(opening)
    # label
    label = skimage.measure.label(cleared, connectivity=2)
    # skip small boxes
    props = skimage.measure.regionprops(label)
    area = [x.area for x in props]
    mean_area = sum(area) / len(area)
    bboxes = [x.bbox for x in props if x.area > mean_area / 2.1]
    bw = 1.0 - cleared  # character: 0.0, background: 1.0
    return bboxes, bw

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
    # img = warp(img)
    bboxes = find_bboxes(img)
    print(len(bboxes))
    # loop(img)
    # update_img(img)
    im1 = italic(img, bboxes)
    plt.imshow(im1)
    plt.show()
