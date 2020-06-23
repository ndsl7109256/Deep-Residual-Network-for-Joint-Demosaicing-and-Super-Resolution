import numpy as np
#from tensorflow.keras import layers
from keras.preprocessing import image
import tensorflow as tf 
from keras.models import Model,load_model
from keras.utils import to_categorical
import os
#import matplotlib.pyplot as plt
import keras
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.layers import Lambda
import random
from PIL import Image 
from random import shuffle
from keras.optimizers import Adam

batch_sz = 16
oti = 'adam'
lr = 0.0001
e_num = 50


def main():
  # With .npy read directly 
  # train_image = np.load('train_image.npy')
  # train_label = np.load('train_label.npy')

  train_image = []
  train_label = []

  entries = os.listdir('./patch')
  for entry in entries:
    im = image.load_img('./patch/'+entry, target_size = (64, 64))
    img = image.img_to_array(im)
    img = img[:,:,0]
    img = img[:,:,np.newaxis]
    train_image.append(img)
  train_image= np.stack(train_image)

  print(train_image.shape)# (x,128,128,1)
  # np.save('train_image',train_image)

  entries = os.listdir('./label')
  for entry in entries:
    im = image.load_img('./label/'+entry, target_size = (128, 128))
    img = image.img_to_array(im)
    train_label.append(img)
  train_label = np.stack(train_label)

  print(train_label.shape)# (x,256,256,3)
  
  # np.save('train_label',train_label)
  

  #Shuffle
  index = [i for i in range(train_image.shape[0])]
  shuffle(index)
  train_image = train_image[index,:,:,:];
  train_label = train_label[index,:,:,:];


  inputs = keras.Input(shape=(None,None,1))
  ##STAGE 1
  ####Conv with a stride of 2
  x = keras.layers.Conv2D(filters = 256, #feature map number
                     kernel_size = 5, 
                     strides = 2,  # 2
                     activation = 'relu',
                     padding = 'same', 
                     input_shape = (None,None,1))(inputs)
 
  ####Sub-pixel Conv
  #x = SubpixelConv2D(input_shape=(64,64,256), scale=2)(x)
  sub_layer = Lambda(lambda x:tf.nn.depth_to_space(x,2))
  x = sub_layer(inputs=x)
  
  ####Conv, PReLU 
  x = keras.layers.Conv2D(filters = 256, #feature map number
                     kernel_size = 3, 
                     strides = 1,  # 2
                     padding = 'same',
                     input_shape = (None,None,64))(x)
  
  x = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(x)

  ##STAGE 2
  ####Residual Block
  

  for i in range(16):
    stage_2_Conv1_1 = keras.layers.Conv2D(filters = 256, #feature map number
                       kernel_size = 3, 
                       strides = 1,  # 2
                       padding = 'same',
                       input_shape = (None,None,64))(x)
    
    stage_2_PReLu1 = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(stage_2_Conv1_1)
    stage_2_Conv1_2 = keras.layers.Conv2D(filters = 256, #feature map number
                       kernel_size = 3, 
                       strides = 1,  # 2
                       padding = 'same',
                       input_shape = (None,None,64))(stage_2_PReLu1)
   
    
    x = keras.layers.Add()([stage_2_Conv1_2,x])

    
  ##STAGE 3
  ####Sub-pixel Conv
  #x = SubpixelConv2D2((64,64,256), scale=2)(x)
  sub_layer = Lambda(lambda x:tf.nn.depth_to_space(x,2))
  x = sub_layer(inputs=x)
  
  ####Conv, PReLU
  x = keras.layers.Conv2D(filters = 256, #feature map number
                     kernel_size = 3, 
                     strides = 1,  # 2
                     padding = 'same',
                     input_shape = (None,None,64))(x)

  x = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(x)
  ####Conv
  outputs = keras.layers.Conv2D(filters = 3, #feature map number
                     kernel_size = 3, 
                     strides = 1,  # 2
                     padding = 'same',
                     activation ='relu',
                     input_shape = (None,None,64))(x)

  rrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='min', min_lr=0.0000001)

 
  model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
  model.summary()
  model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss = 'mean_squared_error', metrics = ['mse'])
  checkpoint = ModelCheckpoint('model.hdf5',verbose=1, monitor='val_loss', 
                                save_best_only=True,save_weights_only=True)
  history = model.fit(train_image, train_label, epochs=e_num, batch_size=batch_sz,verbose=1,
              validation_split = 0.1,callbacks=[checkpoint,rrp],shuffle = True)
  model.save("Demosaic.h5")

  



if __name__ == '__main__':
    main()
