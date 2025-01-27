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
import math
import keras
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.merge import Concatenate
import transform as tm

lines = []
with open('data2/driving_log.csv', 'r') as csvfile:
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
        
def generator(samples, datatype = "test", batch_size=1):
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
                
                filename = path_c.split('\\')[-1]
                current_path = 'data2/IMG/' + filename
                image_c = cv2.imread(current_path)
                
                filename = path_l.split('\\')[-1]
                current_path = 'data2/IMG/' + filename
                image_l = cv2.imread(current_path)
                
                filename = path_r.split('\\')[-1]
                current_path = 'data2/IMG/' + filename
                image_r = cv2.imread(current_path)
                
                c_img = image_c.copy()
                l_img = image_l.copy()
                r_img = image_r.copy()
                
                image_c = tm.normalize(image_c)
                image_l = tm.normalize(image_l)
                image_r = tm.normalize(image_r)
                
                images.extend(image_c, image_l, image_r)
                angles.extend(steer_c, steer_l, steer_r)
                              
                if(datatype == "train"):
                    for i in range(3):
                        image_c_aug = tm.augment(c_img)
                        image_l_aug = tm.augment(l_img)
                        image_r_aug = tm.augment(r_img)
                        
                        if(np.random.uniform > 0.333):
                            image_c_aug = tm.flip(image_c_aug)
                            image_l_aug = tm.flip(image_l_aug)
                            image_r_aug = tm.flip(image_r_aug)
                            mc = -1.0 * steer_c
                            ml = -1.0 * steer_l
                            mr = -1.0 * steer_r
                        else:
                            mc = steer_c
                            ml = steer_l
                            mr = steer_r
                        images.extend(image_c_aug, image_l_aug, image_r_aug)
                        angles.extend(mc, ml, mr)
    
            images = np.array(images)
            angles = np.array(angles)
            yield shuffle(images, angles)

train_generator = generator(train_set, datatype="train")
validation_generator = generator(valid_set)

def Inception(x, d_out):
    d1 = math.floor(d_out/4)
    d2 = math.floor(d_out/2)
    # Inception Module Layer 1: 1x1 Convolution
    conv1x1a = Conv2D(d_out, 1, padding='same', activation='relu')(x)
    conv1x1a = Dropout(0.5)(conv1x1a)
    
    # Inception Module Layer 2: 1x1 Convolution -> 3x3 Convolution -> 3x3 Convolution
    conv1x1b = Conv2D(d1, 1, padding='same', activation='relu')(x)
    conv3x3 = Conv2D(d2, 3, padding='same', activation='relu')(conv1x1b)
    conv3x3 = Conv2D(d_out, 3, padding='same', activation='relu')(conv3x3)
    conv3x3 = Dropout(0.5)(conv3x3)
    
    # Inception Module Layer 3: 1x1 Convolution -> 5x5 Convolution -> 5x5 Convolution
    conv1x1c = Conv2D(d1, 1, padding='same', activation='relu')(x)
    conv5x5 = Conv2D(d2, 5, padding='same', activation='relu')(conv1x1c)
    conv5x5 = Conv2D(d_out, 5, padding='same', activation='relu')(conv5x5)
    conv5x5 = Dropout(0.5)(conv5x5)
    
    # Inception Module Layer 4: MaxPooling -> 1x1 Convolution -> 1x1 Convolution -> 1x1 Convolution
    maxpool = MaxPool2D(pool_size=(3,3), padding='same', strides=(1,1))(x)
    conv1x1d = Conv2D(d1, 1, padding='same', activation='relu')(maxpool)
    conv1x1d = Conv2D(d2, 1, padding='same', activation='relu')(conv1x1d)
    conv1x1d = Conv2D(d_out, 1, padding='same', activation='relu')(conv1x1d)
    conv1x1d = Dropout(0.5)(conv1x1d)

    output = Concatenate(axis = -1)([conv1x1a, conv3x3, conv5x5, conv1x1d])
    
    return output
    
inputs = Input(shape = (160, 320, 3))

x = Cropping2D(cropping=((60,20), (0,0)))(inputs)
x = Conv2D(3, 1, padding='same', activation='relu')(x)
x = Conv2D(16, 3, padding='same', activation='relu')(x)
x = Dropout(0.5)(x)
x = Inception(x, 16)
x = Inception(x, 16)
x = Inception(x, 16)
x = MaxPool2D(strides=(4,4))(x)
x = Conv2D(64, 1, padding='same', activation='relu')(x)
x = Conv2D(128, 5, padding='same', activation='relu')(x)
x = Dropout(0.5)(x)
x = Inception(x, 128)
x = MaxPool2D(strides=(4,4))(x)
x = Conv2D(512, 1, padding='same', activation='relu')(x)
x = Conv2D(256, 5, padding='same', activation='relu')(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(rate=0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(rate=0.5)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(rate=0.5)(x)
x = Dense(16, activation='relu')(x)
x = Dropout(rate=0.5)(x)
logits = Dense(1)(x)

model = Model(inputs=inputs, outputs=logits)
Adam = keras.optimizers.Adam(lr=0.01)
model.compile(loss='mse', optimizer=Adam, metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=len(train_set), \
                    validation_data=validation_generator, validation_steps= \
                    len(valid_set), epochs=1)

model.save('model.h5')
print("Model Saved")
