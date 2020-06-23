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

#split image to prepare the train set
def split(img,name):
    height,width,c = img.shape;
    print(img.shape)
    count = 0;
    for i in range(0,height,20):#step = 20
        for j in range(0,width,20):
            if( i + label_size < height and j + label_size < width ):
                tmp = np.zeros([label_size,label_size,3]);
                tmp = img[ i : i + label_size, j : j + label_size,:];
                #save splite label
                path = 'label/'+name.split('.')[0] +'_'+str(count)+'.png';
                im = Image.fromarray(tmp)
                im.save(path)

                zoom = im.resize((patch_size,patch_size)) 
                gray =  np.zeros([label_size,label_size]);
                zoom = np.array(zoom)

                gray = bayer_reverse(zoom)
                path = 'patch/'+name.split('.')[0] +'_'+str(count)+'.png';
                im = Image.fromarray(gray)
                im = im.convert("L")
                im.save(path)

                count = count + 1
  



def main():
    
    if not os.path.exists('patch'):
        os.makedirs('patch')
    
    if not os.path.exists('label'):
        os.makedirs('label')
    
    entries = os.listdir('koda/')
    for entry in entries:
        print(entry)
        path = 'koda/'+entry
        img = Image.open(path)
        split(img,entry)

  

if __name__ == '__main__':
    main()