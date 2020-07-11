import numpy as np
#from tensorflow.keras import layers
from keras.preprocessing import image
import tensorflow as tf 
from keras.models import Model,load_model
from keras.utils import to_categorical
import os
#import matplotlib.pyplot as plt
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda
import random
from PIL import Image 
from random import shuffle

batch_sz = 4
oti = 'adam'
lr = 0.0001
e_num = 10


# class Histories(Callback):

#     def on_train_begin(self,logs={}):
#         self.losses = []
#         self.accuracies = []

#     def on_batch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))
#         self.accuracies.append(logs.get('acc'))

#https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/convolutional/subpixelupscaling.py
#https://github.com/twairball/keras-subpixel-conv
def SubpixelConv2D(input_shape, scale=4):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    Ref:
        [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158
    :param input_shape: tensor shape, (batch, height, width, channel)
    :param scale: upsampling scale. Default=4
    :return:
    """
    # upsample using depth_to_space
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        import tensorflow as tf 
        return tf.nn.depth_to_space(x, scale)


    return Lambda(subpixel, output_shape=subpixel_shape, name='subpixel')

def SubpixelConv2D2(input_shape, scale=4):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    Ref:
        [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158
    :param input_shape: tensor shape, (batch, height, width, channel)
    :param scale: upsampling scale. Default=4
    :return:
    """
    # upsample using depth_to_space
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel2(x):
        import tensorflow as tf 
        return tf.nn.depth_to_space(x, scale)


    return Lambda(subpixel2, output_shape=subpixel_shape, name='subpixel_2')


#http://ethen8181.github.io/machine-learning/keras/resnet_cam/resnet_cam.html


##https://ithelp.ithome.com.tw/articles/10223034


def main():
  # train_image = np.load('train_image.npy')
  # train_label = np.load('train_label.npy')

  # test_image = np.load('test_img.npy')
  # test_label = np.load('test_lab.npy')


  train_image = []
  train_label = []

  entries = os.listdir('./p')
  for entry in entries:
    im = image.load_img('./p/'+entry, target_size = (64, 64))
    img = image.img_to_array(im)
    img = img[:,:,0]
    img = img[:,:,np.newaxis]
    train_image.append(img)
  train_image= np.stack(train_image)

  print(train_image.shape)# (x,128,128,1)
  # np.save('train_image',train_image)

  entries = os.listdir('./l')
  for entry in entries:
    im = image.load_img('./l/'+entry, target_size = (128, 128))
    img = image.img_to_array(im)
    train_label.append(img)
  train_label = np.stack(train_label)

  print(train_label.shape)# (x,256,256,3)
  
  # np.save('train_label',train_label)

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
  
  # x = keras.layers.Conv2DTranspose(filters = 64, kernel_size = 4,
  # 						strides=2,
  # 						padding='same',
  # 						output_padding=None, 
  # 						data_format=None, 
  # 						dilation_rate=(1, 1), 
  # 						activation='relu', 
  # 						use_bias=True, 
  # 						kernel_initializer='glorot_uniform', 
  # 						bias_initializer='zeros', 
  # 						kernel_regularizer=None, 
  # 						bias_regularizer=None, 
  # 						activity_regularizer=None, 
  # 						kernel_constraint=None, 
  # 						bias_constraint=None)(x)
  
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
    x = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(x)


    
  ##STAGE 3
  ####Sub-pixel Conv
  #x = SubpixelConv2D2((64,64,256), scale=2)(x)
  sub_layer = Lambda(lambda x:tf.nn.depth_to_space(x,2))
  x = sub_layer(inputs=x)
  
  # x = keras.layers.Conv2DTranspose(filters = 64, kernel_size = 3,
  # 						strides=2, 
  # 						padding='same',
  # 						output_padding=(1,1),
  # 						data_format=None, 
  # 						dilation_rate=(1, 1), 
  # 						activation='relu', 
  # 						use_bias=True, 
  # 						kernel_initializer='glorot_uniform', 
  # 						bias_initializer='zeros', 
  # 						kernel_regularizer=None, 
  # 						bias_regularizer=None, 
  # 						activity_regularizer=None, 
  # 						kernel_constraint=None, 
  # 						bias_constraint=None)(stage_2_Concat_2)
  
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


 
  model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
  model.summary()
  model.compile(optimizer=keras.optimizers.Nadam(lr), loss = 'mean_squared_error', metrics = ['mse'])
  #histories = Histories()
  checkpoint = ModelCheckpoint('model.hdf5',verbose=1, monitor='val_loss', 
                                save_best_only=True,save_weights_only=True)
  history = model.fit(train_image, train_label, epochs=e_num, batch_size=batch_sz,verbose=1,
              validation_split = 0.1,callbacks=[checkpoint],shuffle = True)
  model.save("trashn.h5")
  # loss, accuracy = model.evaluate(test_image,test_label)
  # print(loss)
  



if __name__ == '__main__':
    main()
