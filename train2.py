# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 00:29:57 2017

@author: Swapnil
"""
import csv
import cv2
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Dropout, Cropping2D, Lambda
from keras.layers.convolutional import Conv2D
import transform as tm
from keras import regularizers
from keras.backend import tf

foldr = '/Project 3 - Behavioral Cloning/CarND-Behavioral-Cloning-P3'
drt = '/media/smore/Delta/Udacity/Self Driving/Projects' + foldr
lines = []
with open(drt + '/data3/driving_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

with open(drt + '/data4/driving_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_set, valid_set = train_test_split(lines, test_size=0.2)

#train_datagen = ImageDataGenerator(featurewise_center=True, \
#                                   featurewise_std_normalization=True, \
#                                   samplewise_center=True, \
#                                   samplewise_std_normalization=True,
#                                   horizontal_flip=True,
#                                   width_shift_range=0.2,
#                                   height_shift_range=0.05,
#                                   rescale=1./255
#                                   )
#valid_datagen = ImageDataGenerator(featurewise_center=True, \
#                                   featurewise_std_normalization=True, \
#                                   samplewise_center=True, \
#                                   samplewise_std_normalization=True,
#                                   rescale=1./255
#                                   )
      
def generator(samples, datatype = "test", batch_size=128):
    while(1):
        n_samples = len(samples)
        shuffle(samples)
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                steer_c = float(batch_sample[3])
                
                correction = 0.2
                steer_l = steer_c + correction
                steer_r = steer_c - correction
                
                path_c = batch_sample[0]
                path_l = batch_sample[1]
                path_r = batch_sample[2]
                
                filename = path_c.split('/')[-1]
                current_path = drt + '/data3/IMG/' + filename
                image_c = cv2.imread(current_path)
                
                filename = path_l.split('/')[-1]
                current_path = drt + '/data3/IMG/' + filename
                image_l = cv2.imread(current_path)
                
                filename = path_r.split('/')[-1]
                current_path = drt + '/data3/IMG/' + filename
                image_r = cv2.imread(current_path)
                # import pdb; pdb.set_trace()
#                image_c = tm.normalize(image_c)
#                image_l = tm.normalize(image_l)
#                image_r = tm.normalize(image_r)
                c_img = image_c.copy()
                l_img = image_l.copy()
                r_img = image_r.copy()
            
                images.extend([image_c, image_l, image_r])
                angles.extend([steer_c, steer_l, steer_r])
                              
                if(datatype == "train"):
                    image_c_aug = tm.augment(c_img)
                    image_l_aug = tm.augment(l_img)
                    image_r_aug = tm.augment(r_img)

                    images.extend([image_c_aug, image_l_aug, image_r_aug])
                    angles.extend([steer_c, steer_l, steer_r])
    
            images = np.array(images)
            angles = np.array(angles)
            yield shuffle(images, angles)

train_generator = generator(train_set, datatype="train")
validation_generator = generator(valid_set)

inputs = Input(shape = (160, 320, 3))

x = Lambda(lambda x: x / 255.0 - 0.5)(inputs)

x = Cropping2D(cropping=((70,25), (0,0)))(x)

#x = Lambda(lambda img: tf.image.resize_images(img, (66,200)))(x)

x = Conv2D(24, kernel_size=(5,5), strides=(2,2), padding='valid', activation=\
           'relu')(x)
#x = Dropout(0.5)(x)
x = Conv2D(36, kernel_size=(5,5), strides=(2,2), padding='valid', activation=\
           'relu')(x)
#x = Dropout(0.5)(x)
x = Conv2D(48, kernel_size=(5,5), strides=(2,2), padding='valid', activation=\
           'relu')(x)
#x = Dropout(0.5)(x)
x = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation=\
           'relu')(x)
#x = Dropout(0.5)(x)
x = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation=\
           'relu')(x)
#x = Dropout(0.5)(x)
x = Flatten()(x)
#x = Dropout(0.5)(x)
x = Dense(100, activation='relu')(x)
          #activity_regularizer=regularizers.l2(0.01))"""(x)
#x = Dropout(0.5)(x)
x = Dense(50, activation='relu')(x)
          #activity_regularizer=regularizers.l2(0.01))(x)
#x = Dropout(0.5)(x)
x = Dense(10, activation='relu')(x)
          #activity_regularizer=regularizers.l2(0.01))(x)
#x = Dropout(0.5)(x)
logits = Dense(1)(x) #, kernel_regularizer=regularizers.l2(0.01))(x)

model = Model(inputs=inputs, outputs=logits)
Adam = keras.optimizers.Adam()
model.compile(loss='mse', optimizer=Adam)
model.fit_generator(train_generator, steps_per_epoch=len(train_set)/128, \
                    validation_data=validation_generator, validation_steps= \
                    len(valid_set)/128, epochs=1)

model.save('model2.h5')
print("Model Saved")
