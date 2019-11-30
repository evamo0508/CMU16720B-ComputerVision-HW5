import numpy as np

def bold(img, bboxes):
    pass

def italic(img, bboxes):
    # Pre-allocate empty result image, same shape
    A = np.array(([1, 0, 0], [-0.5, 1, 0], [0, 0, 1]))

    result = np.zeros(img.shape)
    # print(bboxes)
    bboxes = bboxes[0]
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
        # print("bias", bias)

        # result = np.zeros(( - bbox[0], bbox[3] - bbox[1]))

        # H = np.arange(h)
        H = np.arange(bbox[0], bbox[2]).astype(int)
        # W = np.arange(w)
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
        img[ind_Orig_Y, ind_Orig_X] = img[br[0], br[1]]
        img[ind_Des_Y, ind_Des_X - bias] = result[ind_Des_Y, ind_Des_X]


    # plt.imshow(bw, cmap='gray')

    return img

def underline(img, bboxes):
    pass

def highlight(img, bboxes):
    pass

def change_color(img, bboxes):
    pass
