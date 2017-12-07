import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense,Dropout,Conv2D,Lambda,Cropping2D,Reshape,Flatten
from keras.models import Sequential,save_model
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# data samples directories
data_dir = 'data/'
samples = []
with open(os.path.join(data_dir,'driving_log.csv')) as f:
    reader = csv.reader(f)
    for line in reader:
        samples.append(line)

# pop out first descriptive line
samples.pop(0)

# steer angles
steer_angle = []

# images
c_images_dir = []
l_images_dir = []
r_images_dir = []
for line in samples:
#     print(type(line[0]))
#     image = cv2.imread()
    c_images_dir.append(os.path.join(data_dir,line[0].strip())) # convert read in bgr image to rgb image
    
#     image1 = cv2.imread()
    l_images_dir.append(os.path.join(data_dir,line[1].strip())) # convert read in bgr image to rgb image
    
#     image = cv2.imread()
    r_images_dir.append(os.path.join(data_dir,line[2].strip())) # convert read in bgr image to rgb image
    steer_angle.append(float(line[3]))

    # steer angle for left-view images and right-view images
steer_angle = np.array(steer_angle)
l_steer_angle, r_steer_angle = steer_angle+0.25 , steer_angle-0.25
y_train = np.concatenate((steer_angle,l_steer_angle,r_steer_angle))
# image directories for all images
X_train_dir = np.concatenate((c_images_dir,l_images_dir,r_images_dir))

print('Total steer angle size is    : {}'.format(y_train.shape))
print('Total training images size is :{}'.format(X_train_dir.shape))

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

rows,cols = 160,320
# randomly translate images with steer angle adjustment
def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    
    return image_tr,steer_ang


def preprocess_image_file_train(input_x, input_y):
#     rand_index = np.random.randint(len(input_x))
    y_steer = input_y
    i_path = input_x
    image = cv2.imread(i_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image,y_steer = trans_image(image,y_steer,100)
    image = augment_brightness_camera_images(image)
#     image = preprocessImage(image)
    image = np.array(image)
    ind_flip = np.random.randint(2)
    if ind_flip==0:
        image = cv2.flip(image,1)
        y_steer = -y_steer
    
    return image,y_steer



model = Sequential()

# preprocess layers
model.add(Lambda(lambda x: x/127.5-1.0,input_shape = (160,320,3),output_shape=(160,320,3),name ='lambda_normalize_layer'))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
# model.add(Reshape((64,64,3)))

model.add(Lambda(lambda image: K.tf.image.resize_images(image, (64,64))))
# color space self-adaptation layer
model.add(Conv2D(3,(1,1),activation='elu',name='cv_1_3x1x1')) 

# basic feature learning layers
# input shape: 64x64x3, output shape: 32x32x24
model.add(Conv2D(24,(5,5),padding='same',strides=(2, 2),activation='elu',name='cv_2_24x5x5'))
# input shape: 32x32x24, output shape: 16x16x36
model.add(Conv2D(36,(3,3),padding='same',strides=(2, 2),activation='elu',name='cv_3_36x3x3'))
# input shape: 16x16x36, output shape: 8x8x48
model.add(Conv2D(48,(3,3),padding='same',strides=(2, 2),activation='elu',name='cv_4_48x3x3'))
model.add(Dropout(0.5))

# complicated feature learning layers
# input shape: 8x8x48, output shape: 6x6x64
model.add(Conv2D(64,(3,3),activation='elu',name='cv_5_64x3x3'))
# input shape: 6x6x64, output shape: 4x4x64
model.add(Conv2D(64,(3,3),activation='elu',name='cv_6_64x3x3'))
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
model.add(Dropout(0.5))

# regression output layer
model.add(Dense(1))

def generator(input_x,input_y, batch_size=128, isTest = False):
    num_samples = len(input_x)
    while 1: # Loop forever so the generator never terminates
        shuffle(input_x,input_y)
        for offset in range(0, num_samples, batch_size):
            batch_x,batch_y = input_x[offset:offset+batch_size],input_y[offset:offset+batch_size]

            images = []
            angles = []
            for sample_x,sample_y in zip(batch_x,batch_y):
#                 batch_sample = batch_sample.split(',')
                if isTest:
                    name = sample_x
                    center_image = cv2.imread(name)
                    center_image = center_image[:,:,::-1]
                    center_angle = float(sample_y)
                else:
                    keep_pr = 0
                    while keep_pr == 0:
                        center_image,center_angle = preprocess_image_file_train(sample_x,sample_y)
                        if abs(center_angle)<.01:
                            keep_pr = 0
                        else:
                            keep_pr = 1
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            np_images = np.array(images)
            np_angles= np.array(angles)
            yield shuffle(np_images, np_angles)

X_train,X_test,Y_train,Y_test = train_test_split( X_train_dir,y_train,test_size=0.2)


epoch = 20
batch_size = 64

train_r_generator = generator(X_train,Y_train,batch_size)
test_r_generator = generator(X_test,Y_test,batch_size,True)

model.compile(optimizer='rmsprop',loss='mse')

# model.fit_generator(generator=train_r_generator,
#                     samples_per_epoch=20000, 
# #                     validation_data=test_r_generator, 
# #                     nb_val_samples=len(Y_test),
#                     nb_epoch=epoch,
#                     verbose=1)
current_min_loss = 1

history_object = model.fit_generator(
    generator = train_r_generator, 
    steps_per_epoch = len(X_train)/batch_size,
    epochs=epoch,
    validation_data=test_r_generator,
    validation_steps = len(X_test)/batch_size)

current_loss = history_object.history['val_loss']
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5') 

