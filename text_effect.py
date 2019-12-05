import numpy as np
import cv2

def sortBoxes2Rows(img, bboxes):

    # find the rows using..RANSAC, counting, clustering, etc.
    y_centers = [(bbox[2] + bbox[0]) / 2 for bbox in bboxes]
    sorted_bboxes = [x for _, x in sorted(zip(y_centers, bboxes))]
    sorted(y_centers)
    current_y = (sorted_bboxes[0][0] + sorted_bboxes[0][2]) / 2
    count = 1
    rows = []
    row = []
    for i in range(len(sorted_bboxes)):
        if abs(y_centers[i] - current_y) > 0.07 * img.shape[0]:
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
    return rows

def bold(img, bboxes):
    print("bold")
    kernel = np.ones((10, 10), np.uint8)
    img_new = cv2.erode(img, kernel, iterations=1)
    # img_new = cv2.dilate(img_new, kernel, iterations=1)
    for bbox in bboxes:
        H = np.arange(bbox[0], bbox[2]).astype(int)
        W = np.arange(bbox[1], bbox[3]).astype(int)

        # Meshgrid
        WW, HH = np.meshgrid(W, H)

        # Now we flatten both arrays
        ind_Orig_Y = HH.flatten().astype(int)
        ind_Orig_X = WW.flatten().astype(int)


        # match pixels
        img[ind_Orig_Y, ind_Orig_X] = img_new[ind_Orig_Y, ind_Orig_X]
    return img

def italic(img, bboxes):
    print("italic")
    # Warp matrix
    A = np.array(([1, 0, 0], [-0.5, 1, 0], [0, 0, 1]))

    # Pre-allocated empty image
    result = np.zeros(img.shape)
    for bbox in bboxes:
        # ul = np.array(([bbox[0], bbox[1], 1]))
        br = np.array(([bbox[2], bbox[3], 1]))
        # ur = np.array(([bbox[0], bbox[3], 1]))
        bl = np.array(([bbox[2], bbox[1], 1]))
        # print("upper left: ", ul)
        # print("upper right: ", ur)
        # print("bottom left: ", bl)
        # print("bottom right: ", br)
        # ul_w = np.floor(np.dot(A, ul)).astype(int)
        # br_w = np.floor(np.dot(A, br)).astype(int)
        # ur_w = np.floor(np.dot(A, ur)).astype(int)
        bl_w = np.floor(np.dot(A, bl)).astype(int)
        # print("upper left warped: ", ul_w)
        # print("upper right warped: ", ur_w)
        # print("bottom left warped: ", bl_w)
        # print("bottom right warped: ", br_w)

        bias = bl_w[1] - bl[1]

        H = np.arange(bbox[0], bbox[2]).astype(int)
        W = np.arange(bbox[1], bbox[3]).astype(int)

        # Meshgrid
        WW, HH = np.meshgrid(W, H)

        # Now we flatten both arrays
        ind_Orig_Y = HH.flatten().astype(int)
        ind_Orig_X = WW.flatten().astype(int)
        ind_Orig = np.array([ind_Orig_Y, ind_Orig_X, np.ones(len(ind_Orig_X))])

        ind_Des = np.floor(np.dot(A, ind_Orig)).astype(int)
        ind_Des_Y, ind_Des_X = ind_Des[0], ind_Des[1]

        # match the pixels
        result[ind_Des_Y, ind_Des_X] = img[ind_Orig_Y, ind_Orig_X]
        # make all pixels at the original char the same color as the paper
        img[ind_Orig_Y, ind_Orig_X] = img[br[0]+5, br[1]+5]
        img[ind_Des_Y, ind_Des_X - bias] = result[ind_Des_Y, ind_Des_X]

    return img

def underline(img, bboxes):
    print("underline")
    rows = sortBoxes2Rows(img, bboxes)

    for row in rows:
        min_x = row[0][1]
        max_x = row[-1][3]
        max_y = 0
        for box in row:
            if box[2] > max_y:
                max_y = box[2]
        cv2.line(img, (min_x, max_y), (max_x, max_y), (0, 0, 0), 15)
    return img

def highlight(img, bboxes):
    print("highlight")
    rows = sortBoxes2Rows(img, bboxes)
    img_new = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for row in rows:
        min_x = row[0][1]
        max_x = row[-1][3]
        min_y = img.shape[0]
        max_y = 0
        for box in row:
            if box[2] > max_y:
                max_y = box[2]
            if box[0] < min_y:
                min_y = box[0]
        for i in range(min_y - 10, max_y + 10):
            for j in range(min_x - 10, max_x + 10):
                if img[i, j] > 20:
                    img_new[i, j, 0] = 255
                    img_new[i, j, 1] = 255
                    img_new[i, j, 2] = 0
    return img_new

def strikethrough(img, bboxes):
    print("strikethrough")
    rows = sortBoxes2Rows(img, bboxes)

    for row in rows:
        min_x = row[0][1]
        max_x = row[-1][3]
        max_y = 0
        sum_y = 0
        for box in row:
            sum_y += (box[0] + box[2]) / 2
        avg_y = int(sum_y / len(row))
        cv2.line(img, (min_x, avg_y), (max_x, avg_y), (0, 0, 0), 15)
    return img

