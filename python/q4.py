import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions

    # denoise
    denoise = skimage.restoration.denoise_bilateral(image, multichannel=True)
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
    bw = 1.0 - cleared # character: 0.0, background: 1.0
    return bboxes, bw
