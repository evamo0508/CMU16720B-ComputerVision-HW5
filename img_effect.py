import numpy as np
import cv2
from skimage.filters import threshold_local
import imutils
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
from text_effect import *

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def warp(img):
    # compute the ratio of the old height
    # to the new height, clone it, and resize it
    ratio = img.shape[0] / 500.0
    orig = img.copy()
    img = imutils.resize(img, height = 500)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        ### todo: incomplete document with more than 4 edges
        if len(approx) == 4:
            screenCnt = approx
            break

    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset = 10, method = "gaussian")
    warped = (warped > T).astype("uint8") * 255
    warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)

    # show the original and scanned images
    #print("STEP 3: Apply perspective transform")
    cv2.imshow("Original", imutils.resize(orig, height = 650))
    cv2.imshow("Scanned", imutils.resize(warped, height = 650))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return warped

def starwars(img, bboxes):
    # Crop out the text area
    rows = sortBoxes2Rows(img, bboxes)
    H, W = img.shape[:2]
    last_row = rows[-1]
    max_y = 0
    for box in last_row:
        if box[2] > max_y:
            max_y = box[2]
    min_x = W
    max_x = 0
    for row in rows:
        if row[0][1] < min_x:
            min_x = row[0][1]
        if row[-1][3] > max_x:
            max_x = row[-1][3]
    img = img[0:max_y + 10, min_x - 10:max_x + 10]

    H, W = img.shape[:2]

    rect = np.array([
                    [0, 0],
                    [W - 1, 0],
                    [W - 1, H - 1],
                    [0, H - 1]], dtype = "float32")
    dst = np.array([
                    [0.4 * W, 0.2 * H],
                    [0.6 * W, 0.2 * H],
                    [W - 1, 0.6 * H],
                    [0, 0.6 * H]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (W, H), cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    mask = (warped[:, :, 0] == 255)
    warped[mask] = (0, 0, 0)
    warped[np.logical_not(mask)] = (255, 255, 0)
    #cv2.imshow("Starwars", warped)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return warped

def find_bboxes(img):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # denoise
    denoise = skimage.restoration.denoise_bilateral(gray, multichannel=False)

    # threshold
    th = skimage.filters.threshold_otsu(denoise)

    # morphology
    opening = skimage.morphology.closing(denoise < th, skimage.morphology.square(2))

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

if __name__ == "__main__":
    image = cv2.imread(args["image"])
    pts = np.array(eval(args["coords"]), dtype = "float32")

    # apply the four point tranform to obtain a "birds eye view" of
    # the image
    warped = four_point_transform(image, pts)

    # show the original and warped images
    cv2.imshow("Original", image)
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)
