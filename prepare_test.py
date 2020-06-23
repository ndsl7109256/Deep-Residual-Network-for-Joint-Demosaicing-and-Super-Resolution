import numpy as np
from numpy import genfromtxt
from numpy import asarray
import math
import copy
import os
from PIL import Image 

patch_size = 64 #input = 64x64
label_size = 128 #output = 128x128

#get RGGB bayer image
def bayer_reverse(img):
    height,width,c = img.shape;
    tmp = np.zeros([height,width]);
    for i in range( height ):
        for j in range( width ):
            if i % 2 == 0 :
                if j % 2 == 0:
                    tmp[i][j] = img[i][j][0];#R
                else:
                    tmp[i][j] = img[i][j][1];#G
            else :
                if j % 2 == 0:
                    tmp[i][j] = img[i][j][1];#G
                else:
                    tmp[i][j] = img[i][j][2];#B

    return tmp;



def main():
    
    if not os.path.exists('koda'):
        os.makedirs('kodap')
    
    entries = os.listdir('koda/')
    for entry in entries:
        print(entry)
        path = 'koda/'+entry
        img = Image.open(path)
        zoom = img.resize((math.floor(img.size[0]/2),math.floor(img.size[1]/2)) )
        zoom = asarray(zoom)
        zoom = bayer_reverse(zoom)
        im = Image.fromarray(zoom)
        im = im.convert("L")
        path = 'kodap/'+entry
        im.save(path)
