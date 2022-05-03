import numpy as np
from PIL import Image
import math
import operator
from functools import reduce

def image2array(img):
    image_input = Image.open(img)
    arr = np.array(image_input)
    assert(len(arr.shape) == 3)
    h = arr.shape[0]
    w = arr.shape[1]
    c = arr.shape[-1]
    arr = arr.reshape(-1)
    return arr,h,w,c

def array2image(arr,h,w,c):
    arr = arr.reshape(h,w,c)
    img = Image.fromarray(arr)
    return img

# test

def image_contrast(img1, img2):

    image1 = Image.open(img1)
    image2 = Image.open(img2)

    h1 = image1.histogram()
    h2 = image2.histogram()

    result = math.sqrt(reduce(operator.add,  list(map(lambda a, b: (a-b)**2, h1, h2)))/len(h1))
    return result

if __name__ == "__main__":
    arr,h,w,c = image2array("clip1.png")
    print(arr)
    img = array2image(arr,h,w,c)
    img.save("clip1new.png")
    print(image_contrast("clip1.png","clip1new.png"))
