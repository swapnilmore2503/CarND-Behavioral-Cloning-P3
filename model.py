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
import keras
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Dropout, Cropping2D, Lambda
from keras.layers.convolutional import Conv2D
import transform as tm
from keras import regularizers
from keras.backend import tf

foldr = '/Project 3 - Behavioral Cloning/CarND-Behavioral-Cloning-P3'
drt = '/media/smore/Delta/Udacity/Self Driving/Projects' + foldr

### Read csvfiles
def readcsv():
    lines = []
    with open(drt + '/data3/driving_log.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    with open(drt + '/data4/driving_log.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

### Define generator function to generate images
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

### Define Model
def CloNet(train_set, valid_set):

    # Set sizes
    n_train = len(train_set)
    n_valid = len(valid_set)
    
    # Generate train and valid images
    train_generator = generator(train_set, datatype="train")
    validation_generator = generator(valid_set)

    # Nvidia Autonomous Model
    inputs = Input(shape = (160, 320, 3))

    x = Lambda(lambda x: x / 255.0 - 0.5)(inputs) # Out Size: 160x320x3
    x = Cropping2D(cropping=((70,25), (0,0)))(x) # Out Size: 65x320x3 
    #x = Lambda(lambda img: tf.image.resize_images(img, (66,200)))(x)

    x = Conv2D(24, kernel_size=(5,5), strides=(2,2), padding='valid', activation=\
               'relu')(x) # Out Size: 31x158x24 

    x = Conv2D(36, kernel_size=(5,5), strides=(2,2), padding='valid', activation=\
               'relu')(x) # Out Size: 14x77x36 

    x = Conv2D(48, kernel_size=(5,5), strides=(2,2), padding='valid', activation=\
               'relu')(x) # Out Size: 5x37x48 

    x = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation=\
               'relu')(x) # Out Size: 3x35x64

    x = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation=\
               'relu')(x) # Out Size: 1x33x64

    x = Flatten()(x) # Out Size: 2122
    x = Dropout(0.75)
    x = Dense(100, activation='relu')(x) # Out Size: 100
    x = Dense(50, activation='relu')(x) # Out Size: 50
    x = Dense(10, activation='relu')(x) # Out Size: 10

    logits = Dense(1)(x) # Out Size: 1

    # Fit the model to the train images and validate the model
    model = Model(inputs=inputs, outputs=logits)
    Adam = keras.optimizers.Adam()
    model.compile(loss='mse', optimizer=Adam)
    history_object = model.fit_generator(train_generator, steps_per_epoch=n_train/128, \
                        validation_data=validation_generator, validation_steps= \
                        n_valid/128, epochs=1, verbose=1)

    # Save the model
    model.save('model.h5')
    print("Model Saved")
    
    return history_object

### Define the main function
def main():
    lines = readcsv()
    
    # Split the data into train and validation sets
    train_set, valid_set = train_test_split(lines, test_size=0.2)

    # Fit the model
    history = CloNet(train_set, valid_set)

    ## Visualize model
    # Print the keys in the history object
    print(history.keys())

    # Plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    pass

if __name__ == "__main__":
    main()

    
    
    
    
    
