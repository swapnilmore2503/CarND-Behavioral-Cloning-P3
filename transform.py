# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 16:48:24 2017

@author: Swapnil
"""

import numpy as np
import cv2
from numba import vectorize, uint8, int32

def translate(img, xlimit=4, ylimit=4):
    rows, cols, ch = img.shape
    tr_x = xlimit*np.random.uniform() - xlimit/2
    tr_y = ylimit*np.random.uniform() - ylimit/2
    Trans_M = np.float32([1,0,tr_x], [0,1,tr_y])
    return cv2.warpAffine(img, Trans_M, (cols, rows))

def rotate(img, a_limit = 10):
    rows, cols, ch = img.shape
    a_rot = a_limit*np.random.uniform() - a_limit/2
    Rot_M = cv2.getRotationMatrix2D((cols/2, rows/2), a_rot, 1)
    return cv2.warpAffine(img, Rot_M, (cols, rows))


def flip(img):
    img1 = img.copy()
    img1 = cv2.flip(img1, 0)
    return img1

def bright(img, b_limit = 100):
    b_random = b_limit*np.random.uniform() - b_limit/2
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img1[:,:,2] = img1[:,:,2] + b_random
    img1 = cv2.cvtColor(img1, cv2.COLOR_HSV2BGR)
    return img1


def contrast(img, c_limit = 50):
    c_random = c_limit*np.random.uniform() - c_limit
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    img1[:,:,1] = img1[:,:,1] + c_random
    img1 = cv2.cvtColor(img1, cv2.COLOR_HLS2BGR)
    return img1

def shadow(img):
    y_t = 32*np.random.uniform()
    x_t = 0
    x_b = 32
    y_b = 32*np.random.uniform()
    
    img_hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    shadow_mask = 0*img_hls[:,:,1]
    
    X_mesh = np.mgrid[0:img.shape[0],0:img.shape[1]][0]
    Y_mesh = np.mgrid[0:img.shape[0],0:img.shape[1]][1]
    
    shadow_mask[((X_mesh - x_t)*(y_b - y_t) - (x_b - x_t) * (Y_mesh - y_t) >= 0)] = 1
   
    if(np.random.randint(2)==1):
        random_light = 0.55
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        
        if(np.random.randint(2)==1):
            img_hls[:,:,1][cond1] = img_hls[:,:,1][cond1] * random_light
        else:
            img_hls[:,:,1][cond0] = img_hls[:,:,1][cond0] * random_light    
    
    img = cv2.cvtColor(img_hls,cv2.COLOR_HLS2BGR)
    return img


def normalize(img):
    img = np.array(img, dtype=np.float32)
    img = (img - 127.5)/127.5
    return img

def augment(img):
    img1 = bright(img)
    #img = contrast(img)
    img1 = shadow(img)
    return img1 #normalize(img)
    
