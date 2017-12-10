# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/example1.jpg "Original image"
[image2]: ./examples/example1_bci.jpg "Random Brightness Enhencement"
[image3]: ./examples/tans.jap "Randomly translated image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of three parts: 

**Preprocess layers:**
* Lambda layer to normalize color channel of input RGB image 160*320*3 with x/127.0-1.0 to range [-0.5,0.5]
*  Cropping layer to trim unuseful horizontal part as suggested in course 
* Lambda layer to resize input image to 64x64 to alleviate training workload

**Convolutional layers:**
* First convolution layer with 3 1x1 filters to adapt to color channels as suggested in some posts in _Medium_
* Following convolutional layers with 5x5,3x3 filter sizes, depths from 24 to 64 (model.py lines 102-152), each with ELU as activation function to introduce nonlinearity, strides 2x2 for first three cv layers to downsample inputs.

**Fully-connected layers:**

The model also includes four fully-connected layers after convolutional layers (flattened for linear input to fc layer) to combine learned features for output. The last fc layer is output as continuous steering angles. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 125,133,147). 

The model was trained and validated on udacity data sets of all three camera (center-camera, left-camera and right-camera) with adding biases steer angles to left-camero and right-camera datas to ensure that the model was not overfitting (code line 46-52). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an rmsprop optimizer, so the learning rate was not tuned manually (model.py line 210).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road with steering angles slightly adjusted for 0.25(adding angles for left-camera and subtracting 0.25 for right-camera). 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to build as deep as possible with smaller size of filters. As road textures and lane lines are limited  for first track, I think it not necessary to build a network with significant huge amounts of filters for each layer. As I saw in some posts, some implementations are adopting the idea of transfer learning like VGG in courses which I think could be not beneficial according to the task we are facing. So I prefer to apply the Nvidia model presented in courses with some slight adjustments (more preprosessing layers, adding color adaptation layers, using smaller filters in some layers) since it is deep enough but with smaller set of parameters(197,959).

My first step was to use a simple linear network model similar to the one in course to start and test on simulator. After several experimens similar to lecturer, I applied the network that I used in Traffic-sign-Classifier. I thought this model might be appropriate because it got a validating accuracy of 97.0% for traffic sign. 

In the first stage, I use the data collected by myself. I tried collecting with reverse track, recovering from deviation place even totally outside of track. I was thinking this could be a perfect data set, but the model was performing terribly with traininf and validation accuracy around 0.5 again and again no matter how I trained and preprocessed. Then I found it was completely wrong to evaluate the model with "accuracy" for this project after seeing some posts on forum and [**_medium_**](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9) because the output is a continuous float number not classifier. It is better to consider the mean-squared-error as this means how predicted steer angles similar to human behavior. Additionally, he data that I collected was some bad "behavior" as I the images that I used at that time are all from center-camera in which contain most of 0 for steer angles. So I decided to use the data set of udacity of all cameras.

I used the translation and brightness methods in the post as whatever I write myself, it wll be similar as I use the same parameters. 

Then I started on a jupyter notebook for convenience. I applied the architeture described above and prepared data with concatenating all images and add universally steer angle adjusting for left-camera and right-camera. I decided to use generator instead of train on memery when I found my computer was working really slow after I loaded all images for train. Therefore I modified the generator in udacity courses to yield different data set for train and test. The final generator will generate preprocessed images with brightness adjusting and randmon translation of adjusted steer angles for training and original images fro test.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my model had a low mean squared error on the test set but a high mean squared error on the train set with 5 epochs. This implied that the model was underfitting. So I increased to 20 epochs and the output of two loss converged. 

To combat the overfitting, I modified the model so that there are some dropout layers but not too many because data set is limited.

Then I trained my model for 20 epochs and get a converged training loss and validating loss.

The final step was to run the simulator to see how well the car was driving around track one. The trained model works perfectly on track 1 even I add speed to 15 compared to 9. But it fell soon after launching on track 2. But overall, this is sufficient for this project.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)


| Layer (type)         		|     Output Shape 	        |   Param       |
|:---------------------:|:----------------------:|:-----------------------:|                            
| lambda_normalize_layer    |  (Lamb (None, 160, 320, 3)      | 0        |
| cropping2d_1 (Cropping2D)  |  (None, 65, 320, 3)     |   0|
| lambda_1 (Lambda)         |   (None, 64, 64, 3)     |    0|
| cv_1_3x1x1 (Conv2D)       |   (None, 64, 64, 3)     |    12|
| cv_2_24x5x5 (Conv2D)      |   (None, 32, 32, 24)    |    1824|
| cv_3_36x3x3 (Conv2D)      |   (None, 16, 16, 36)    |    7812|
| cv_4_48x3x3 (Conv2D)      |   (None, 8, 8, 48)      |    15600|
| dropout_1 (Dropout)       |   (None, 8, 8, 48)      |    0
| cv_5_64x3x3 (Conv2D)      |   (None, 6, 6, 64)       |   27712|
| cv_6_64x3x3 (Conv2D)      |   (None, 4, 4, 64)      |    36928|
| dropout_2 (Dropout)        |  (None, 4, 4, 64)      |    0|
| flatten_1 (Flatten)       |   (None, 1024)        |      0|
| fc_1_100 (Dense)          |   (None, 100)         |      102500|
| fc_2_50 (Dense)           |   (None, 50)         |       5050|
| fc_3_10 (Dense)           |   (None, 10)        |        510|
| dropout_3 (Dropout)       |   (None, 10)        |        0|
| dense_1 (Dense)           |   (None, 1)          |       11|
| Total params | |197,959     |        
| Trainable params | | 197,959| 
| Non-trainable params | | 0|

#### 3. Conclusion

In this project, I learned a new idea about CNN. Different from the previous projects, We need collect training data ourselves and the output is continuous instead of discrete. This brings two new problems:
1. How to get good data. The word "trash in trash out" has been literally proven in this project. We can not get a model performing better than the training data. Also as model cannot judge intelligently which data are effective which should be ignored, the features appeared the most frequently will be learned. Speaking of this project, the given data set of center-camera are consited of most of 0 for steer angles and not suficient data for turning and recovery, so the model will learn to say "0" when given input. This is the reason why the starting model always drove straight to deviate from the track.
2. The importance of data augmentation. I didn't realized the augmentation is so critical to the success of certain tasks. I used it for getting more images in traffic sign classifier project. But the real turning point for this project is just introducing image translate and steer angles adjustment. The change is so obvious that I got a perfect driving behavior without modifying model architecure compared to direct falling out of track with previous bad data.

To augment the data sat, I add random brightness enhancement and randomly translate images and angles thinking that this would help model learn how to handle different sunlight enviroment and deviation situations. For example, here is an image that has then been passed to add random brightness enhancement:

![alt text]][image1]

This project lead me to a new stage when thinking about machine learning and neural network. It taught me the importance of proper data in success of a task and also appropriate method to adjust training data to yield better performance with same data set.
