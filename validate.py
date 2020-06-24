from keras.models import load_model
from keras.layers import Lambda
from keras.preprocessing import image
import keras
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
import os
import math
oti = 'adam'
lr = 0.0001

def get_cpsnr(RGB1,RGB2,b):
	RGB1 = RGB1.astype('double'); 
	RGB2 = RGB2.astype('double');
	diff = RGB1[b:-1-b,b:-1-b,:]-RGB2[b:-1-b,b:-1-b,:];
	num = np.size(diff[:,:,1]);
	MSE_R = np.sum( np.power(diff[:,:,2],2) );
	MSE_G = np.sum( np.power(diff[:,:,1],2) );
	MSE_B = np.sum( np.power(diff[:,:,0],2) );
	CMSE = (MSE_R + MSE_G + MSE_B)/(3*num);
	CPSNR = 10*math.log(255*255/CMSE,10);
	return CPSNR;

def create_model():
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
	    #x = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(x)

	  
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


	 
	model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
	return model

model = create_model()
model.load_weights('model.hdf5')

sum = 0
#read your test image
entries = os.listdir('./kodap/')
for entry in entries:
         path = './kodap/'+entry


         test_image = image.load_img(path)
         test_image = image.img_to_array(test_image)

         test_image = test_image[:,:,0]
         test_image = test_image[np.newaxis,:,:,np.newaxis]
         
         out = model.predict(test_image)
         path = './koda/'+entry
         ori = image.load_img(path)
         ori = image.img_to_array(ori)
         out = out[0];
         print(get_cpsnr(out,ori,12) );
         sum+=get_cpsnr(out,ori,12);
         out = image.array_to_img(out)
         #plt.imshow(out)
         #plt.show()
print('avg')        
print(sum/len(entries))
