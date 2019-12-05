import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.widgets import Button, CheckButtons, RectangleSelector


# from nn import *
# from q4 import *
from img_effect import *
from text_effect import *

class GUI(object):
    def __init__(self, img, bboxes):
        self.img = img
        self.bboxes = bboxes
        self.selected_bboxes = []
        self.labels = ["Bold", "Italic", "Underline", "Highlight", "StrikeThrough", "StarWars"]
        self.f = [False for x in self.labels]
        self.fList = [bold, italic, underline, highlight, strikethrough, starwars]
        fig, self.ax = plt.subplots()

        # Button
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Apply')
        bnext.on_clicked(self.apply)

        rax = plt.axes([0.01, 0.4, 0.25, 0.5])
        check = CheckButtons(rax, self.labels)
        check.on_clicked(self.check_effects)

        # Select region
        # drawtype is 'box' or 'line' or 'none'
        toggle_selector.RS = RectangleSelector(self.ax, self.line_select_callback,
                                       drawtype='box', useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)
        print("\n      click  -->  release")
        plt.connect('key_press_event', toggle_selector)
        self.ax.imshow(img, cmap='gray')
        plt.show()

    def apply(self, event):
        for i, flag in enumerate(self.f):
            if flag and i == 5: #starwars
                self.img = self.fList[i](self.img, self.bboxes)
                self.bboxes, _ = find_bboxes(self.img)
            elif flag:
                self.img = self.fList[i](self.img, self.selected_bboxes)
        self.ax.imshow(self.img)

    def check_effects(self, label):
        index = self.labels.index(label)
        self.f[index] = not self.f[index]

    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        self.selected_bboxes = [x for x in self.bboxes if (x1 < x[1] and x[1] < x2
                                                       and x1 < x[3] and x[3] < x2
                                                       and y1 < x[0] and x[0] < y2
                                                       and y1 < x[2] and x[2] < y2)]

def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)

def load_img():
    return cv2.imread('images/simon2.jpg')

if __name__ == "__main__":
    img = load_img()
    img = warp(img)
    bboxes, bw = find_bboxes(img)
    gui = GUI(img, bboxes)
    cv2.imwrite('images/result.jpg', gui.img)


