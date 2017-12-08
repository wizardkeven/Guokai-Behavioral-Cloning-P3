from keras.layers import Dense,Dropout,Conv2D,Lambda,Cropping2D,Reshape,Flatten
from keras.models import Sequential,save_model
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model

# model = load_model('model.h5')
#======================start of model==========================
# train model
model = Sequential()

#=========preprocess layers==========
# lambda layer to normalize input color
model.add(Lambda(lambda x: x/127.5-1.0,input_shape = (160,320,3),output_shape=(160,320,3),name ='lambda_normalize_layer'))
# cropping layer to trim off unrelavent horizontal parts
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
# resize layer to resize input image to 64x64 in order to reduce training time
model.add(Lambda(lambda image: K.tf.image.resize_images(image, (64,64))))

# convolutional layer for color space auto-adaptation inspired from medium post
model.add(Conv2D(3,(1,1),activation='elu',name='cv_1_3x1x1')) 

#==========basic feature learning layers==============
# input shape: 64x64x3, output shape: 32x32x24
model.add(Conv2D(24,(5,5),padding='same',strides=(2, 2),activation='elu',name='cv_2_24x5x5'))
# input shape: 32x32x24, output shape: 16x16x36
model.add(Conv2D(36,(3,3),padding='same',strides=(2, 2),activation='elu',name='cv_3_36x3x3'))
# input shape: 16x16x36, output shape: 8x8x48
model.add(Conv2D(48,(3,3),padding='same',strides=(2, 2),activation='elu',name='cv_4_48x3x3'))
# dropout for 0.5
model.add(Dropout(0.5))

# complex feature learning layers
# input shape: 8x8x48, output shape: 6x6x64
model.add(Conv2D(64,(3,3),activation='elu',name='cv_5_64x3x3'))
# input shape: 6x6x64, output shape: 4x4x64
model.add(Conv2D(64,(3,3),activation='elu',name='cv_6_64x3x3'))
# dropout for 0.5
model.add(Dropout(0.5))

# flatten layer
# input shape: 4x4x64, output shape: 1024
model.add(Flatten())

# Fully-connected layers
# input shape: 1024 output shape: 100
model.add(Dense(100,name='fc_1_100'))
# input shape: 100 output shape: 50
model.add(Dense(50,name='fc_2_50'))
# input shape: 50 output shape: 10
model.add(Dense(10,name='fc_3_10'))
# dropout for 0.5
model.add(Dropout(0.5))

# regression output layer
model.add(Dense(1))

model_summary = model.summary()

# plot_model(model, to_file='model.png',show_shapes=True)