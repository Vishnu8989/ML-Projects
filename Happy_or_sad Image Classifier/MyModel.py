#!/usr/bin/env python
# coding: utf-8

# Below is code with a link to a happy or sad dataset which contains 80 images, 40 happy and 40 sad. 
# Create a convolutional neural network that trains to 100% accuracy on these images,  which cancels training upon hitting training accuracy of >.999
# 
# Hint -- it will work best with 3 convolutional layers.

# In[1]:
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab happy-or-sad.zip from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location

# In[15]:


# GRADED FUNCTION: train_happy_sad_model

# Please write your code only where you are indicated.
# please do not remove # model fitting inline comments.

DESIRED_ACCURACY = 0.999

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if logs.get('acc') is not None and logs.get('acc') > .999:
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

# This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,'relu'),
    tf.keras.layers.Dense(1,'sigmoid')
])

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
                    optimizer=RMSprop(lr=.001),
                    metrics=['acc'])
    

# This code block should create an instance of an ImageDataGenerator called train_datagen 
# And a train_generator by calling train_datagen.flow_from_directory

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255)

# Please use a target_size of 150 X 150.
train_generator = train_datagen.flow_from_directory(
    './h-or-s',
    target_size=(150,150),
    batch_size  =10,
    class_mode = 'binary'
)
# Expected output: 'Found 80 images belonging to 2 classes'

# This code block should call model.fit_generator and train for
# a number of epochs.
# model fitting
history = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs = 15,
    verbose = 1,
    callbacks = [callbacks])
# model fitting


# In[16]:


# The Expected output: "Reached 99.9% accuracy so cancelling training!""


# In[17]:


path = './download.jpg'
img = image.load_img(path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images,batch_size=10)
print(classes[0])
if classes[0] > 0.5:
    print("Image is a Happy")
else:
    print("Image is a Sad")

path = './sad.jpg'
img = image.load_img(path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if classes[0] > 0.5:
    print("Image is a Happy")
else:
    print("Image is a Sad")
