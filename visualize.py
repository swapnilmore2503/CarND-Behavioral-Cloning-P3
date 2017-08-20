import cv2
import numpy as np
import glob
import transform as tm
import matplotlib.pyplot as plt

foldr = '/Project 3 - Behavioral Cloning/CarND-Behavioral-Cloning-P3'
drt = '/media/smore/Delta/Udacity/Self Driving/Projects' + foldr
images = glob.glob(drt + "/data3/IMG/*.jpg")

n_files = len(images)

n = np.random.randint(0, n_files)

path = images[n]

img = cv2.imread(path)
img1 = tm.augment(img)

b,g,r = cv2.split(img)
img = cv2.merge((r,g,b))

b,g,r = cv2.split(img1)
img1 = cv2.merge((r,g,b))


plt.figure(figsize = (16,4))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(img1)
plt.title("Augmented Image")
plt.show()
