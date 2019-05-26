# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 18:22:18 2018

@author: user
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import data,io
image = data.camera()
plt.figure(figsize= (15,15))
plt.subplot(221)
plt.imshow(image)

h = np.zeros(256,dtype = int)

for i in range(len(image[1:,:])):
    for j in range(len(image[:,1:])):
        h[image[i,j]] = h[image[i,j]] + 1;

plt.subplot(222)
plt.plot(range(len(h)),h)


def fun(lmin,lmax,x):
      
    res = np.clip((255 - 0) * (x - lmin)/(lmax - lmin), 0 , 255)
    res = math.floor(res)
    return res

image2 = np.array(image)  #create the same shape array

M,N = image.shape
cut = M * N * 5/100

for ii in range(len(h)):
    cut = cut - h[ii]
    if cut <= 0:
        lmin = ii
        break

cut = M * N * 5/100
for jj in range(len(h)):
    cut = cut - h[255-jj]
    if cut <= 0:
        lmax = 256 - jj
        break
        
print('lmin = ',lmin,'lmax =',lmax)

for row in range(M):
    for col in range(N):
        image2[row][col] = fun(lmin,lmax,image[row][col])

h2 = np.histogram(image2, bins = range(0,257))[0]
plt.subplot(223)
plt.imshow(image2)
plt.subplot(224)
#plt.hist(image2.ravel() , bins = range(0,257))
plt.plot(h2)

"""
from skimage import io 
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
import os

im = data.camera()
#im = data.camera()
h = np.histogram(im,bins = range(0,257))[0]

plt.figure(1,figsize = (15,15))
plt.subplot(221)
plt.imshow(im)

plt.subplot(222)
plt.plot(h)

lmin = 10
lmax = 246
#how to obtain a linear function that converts the range(7,200) to (0,255)

def fun(lmin,lmax,x):
    
    #lmin = float(lmin)
    #lmax = float(lmax)
    #res = np.clip((255 - 0) * (x - lmin)/(lmax - lmin), 0 , 255)
    #return int(res)
    return np.clip((255 - 0) * (x - lmin)/(lmax - lmin), 0 , 255)

im2 = np.array(im)  #create the same shape array
M,N = im.shape

for row in range(M):
    for col in range(N):
        im2[row][col] = fun(lmin,lmax,im[row][col])

h2 = np.histogram(im2, bins = range(0,257))[0]
plt.subplot(223)
plt.imshow(im2)
plt.subplot(224)
plt.plot(h2)
"""