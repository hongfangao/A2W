from unittest import result
from sklearn import linear_model
import os
import numpy as np
from PIL import Image
from tqdm import trange

from image2array import array2image,image2array
from music2array import audio2array,array2audio


def build_train(root):
    audios = []
    imgs = []
    for file in os.listdir(root):
        for subf in os.listdir(root + file):
            if subf.endswith("wav"):
                audios.append(root + file + "/" + subf)
            elif subf.endswith("png"):
                imgs.append(root + file + "/" + subf)
    audios.sort()
    imgs.sort()
    audioarrs = []
    for audio in audios:
        audioarrs.append(audio2array(audio))
    imgarrs = []
    for img in imgs:
        imgarrs.append(image2array(img))
    X_train = []
    for audio in audioarrs:
        X_train.append(audio[0])

    Y_train = []
    for img in imgarrs:
        Y_train.append(img[0])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    return X_train,Y_train,imgarrs[0][1],imgarrs[0][2],imgarrs[0][3]

def train(root):
    model_linear = linear_model.LinearRegression()
    X_train,Y_train,h,w,c = build_train(root)
    print("begin fitting with linear regression model.")
    model_linear.fit(X_train,Y_train)
    print("end fitting.")
    return model_linear,h,w,c

def test_single(audio,outputpath,model,h,w,c):
    print("begin testing.")
    testaudioarr = audio2array(audio)[0].reshape(1,-1)
    result = model.predict(np.array(testaudioarr))
    opimg = array2image(np.uint8(result),h,w,c)
    os.mkdir(outputpath)
    opimg.save(outputpath + "/" + audio[0:-3] + "png")
    print("end testing.")

def test_folder(audiopath,outputpath,model,h,w,c):
    print("begin testing.")
    for file in os.listdir(audiopath):
        testaudioarr = audio2array(audiopath + file)[0].reshape(1,-1)
        result = model.predict(np.array(testaudioarr))
        opimg = array2image(np.uint8(result),h,w,c)
        os.mkdir(outputpath)
        opimg.save(outputpath + "/" + file[0:-3] + "png")
    print("end testing.")

def train_and_test_single(root,audio,outputpath):
    model_linear = linear_model.LinearRegression()
    X_train,Y_train,h,w,c = build_train(root)
    print("begin fitting with linear regression model.")
    model_linear.fit(X_train,Y_train)
    print("end training.")
    testaudioarr = audio2array(audio)[0].reshape(1,-1)
    result = model_linear.predict(np.array(testaudioarr))
    opimg = array2image(np.uint8(result),h,w,c)
    os.mkdir(outputpath)
    opimg.save(outputpath + "/" + audio[0:-3] + "png")

def train_and_test_folder(root,audiopath,outputpath):
    model_linear = linear_model.LinearRegression()
    X_train,Y_train,h,w,c = build_train(root)
    print("begin fitting with linear regression model.")
    model_linear.fit(X_train,Y_train)
    print("end training.")
    for file in os.listdir(audiopath):
        testaudioarr = audio2array(audiopath + file)[0].reshape(1,-1)
        result = model_linear.predict(np.array(testaudioarr))
        opimg = array2image(np.uint8(result),h,w,c)
        os.mkdir(outputpath)
        opimg.save(outputpath + "/" + file[0:-3] + "png")


## build train
root = "trainsets/"
audio = "testin.wav"
outputpath  = "out"

if __name__ == "__main__":
    model,h,w,c = train(root)
    test_single(audio,outputpath,model,h,w,c)


