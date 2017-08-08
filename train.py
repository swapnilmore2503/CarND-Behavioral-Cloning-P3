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
        
def generator(samples, datatype="valid", batch_size=2):
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
                    filename = source_path.split('/')[-1]
                    image = cv2.imread(filename)
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

            images = np.array(images)
            angles = np.array(angles)
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

inputs = Input(shape = (160, 320, 3))

x = Cropping2D(cropping=((60,20), (0,0)))(inputs)
x = Conv2D(16, 5, padding='same', activation='relu')(x)
x = MaxPool2D()(x)
x = Conv2D(32, 5, padding='same', activation='relu')(x)
x = MaxPool2D()(x)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)
x = Dropout(rate=0.5)(x)
x = Dense(16, activation='relu')(x)
x = Dropout(rate=0.5)(x)
logits = Dense(1)(x)

model = Model(inputs=inputs, outputs=logits)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=len(train_set), \
                    validation_data=validation_generator, validation_steps= \
                    len(valid_set), epochs=6)

model.save('model.h5')
print("Model Saved")