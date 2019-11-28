import numpy as np

def load_img():
    pass

def warp(img):
    pass

def find_bboxes(img):
    pass

def loop(img):
    while True:
        f = parse_input(input())
        img = f(img, bboxes)
        update_img(img)

def parse_input(Input):
    # return function name
    pass

def update_img(img):
    # sth with GUI
    pass

if __name__ == "__main__":
    img = load_img()
    img = warp(img)
    bboxes = find_bboxes(img)
    loop(img)
