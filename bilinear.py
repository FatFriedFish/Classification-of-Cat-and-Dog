# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:12:23 2018

@author: Yuhan.Long
"""

import cv2
import numpy as np
import math

def bilinear_resize(img, m, n):
    height, width, channel = img.shape
    resize = np.zeros((m, n, channel), np.uint8);
    
    sh = height / m
    sw = width / n
    
    for row in range(m):
        y0 = int(row * sh)
        y1 = y0 if y0 + 1 >= height else y0 + 1
        u = row*sh - y0
        for col in range(n):
            x0 = int(col * sw)
            x1 = x0 if x0 + 1 >= width else x0 + 1
            v = col * sw - x0
            
            # coefficients
            a, b, c, d = ((1-u)*(1-v)), ((1-u)*v), (u*(1-v)), (u*v)
            
            value = [0,0,0]
            for l in range(3):
                value[l] = int(img[y0, x0, l] * a + img[y1, x0,l] * b + img[y0, x1,l] * c + img[y1, x1, l] * d)
            resize[row, col] = (value[0], value[1], value[2])
    return resize

