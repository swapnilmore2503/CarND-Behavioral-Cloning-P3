
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
from keras.preprocessing.image import ImageDataGenerator
import math
import keras

lines = []
with open('data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_set, valid_set = train_test_split(lines, test_size=0.2)

train_datagen = ImageDataGenerator(featurewise_center=True, \
                                   featurewise_std_normalization=True, \
                                   samplewise_center=True, \
                                   samplewise_std_normalization=True,
                                   horizontal_flip=True
                                   )
valid_datagen = ImageDataGenerator(featurewise_center=True, \
                                   featurewise_std_normalization=True, \
                                   samplewise_center=True, \
                                   samplewise_std_normalization=True
                                   )
        
def generator(samples, datatype="valid", batch_size=4):
    n_samples = len(samples)
    while(1):
        shuffle(samples)
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('\\')[-1]
                    current_path = 'data2/IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                    if(i == 1):
                        measurement = float(line[3]) + 1
                        angles.append(measurement)
                    elif(i == 2):
                        measurement = float(line[3]) - 1
                        angles.append(measurement)
                    else:
                        measurement = float(line[3])
                        angles.append(measurement)

            images = np.array(images, dtype = np.float64)
            angles = np.array(angles, dtype = np.float64)
            if(datatype == "train"):
                train_datagen.fit(images)
                generate = train_datagen.flow(images, angles)
            else:
                valid_datagen.fit(images)
                generate = valid_datagen.flow(images, angles)
            return generate

train_generator = generator(train_set, datatype="train")
validation_generator = generator(valid_set)

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.merge import Concatenate

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

    output = Concatenate(axis = 3)([conv1x1a, conv3x3, conv5x5, conv1x1d])
    
    return output
    
inputs = Input(shape = (160, 320, 3))

x = Cropping2D(cropping=((60,20), (0,0)))(inputs)
x = Conv2D(3, 1, padding='same', activation='relu')(x)
x = Conv2D(16, 3, padding='same', activation='relu')(x)
x = Dropout(0.5)(x)
x = Inception(x, 16)
x = MaxPool2D()(x)
x = Conv2D(64, 1, padding='same', activation='relu')(x)
x = Conv2D(128, 5, padding='same', activation='relu')(x)
x = Dropout(0.5)(x)
x = Inception(x, 128)
x = MaxPool2D()(x)
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
Adam = keras.optimizers.Adam(lr=0.0005)
model.compile(loss='mse', optimizer=Adam, metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=len(train_set), \
                    validation_data=validation_generator, validation_steps= \
                    len(valid_set), epochs=6)

model.save('model.h5')
print("Model Saved")
