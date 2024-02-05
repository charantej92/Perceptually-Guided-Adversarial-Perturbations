#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:53:19 2020

@author: charan
"""
# File to generate adversarial examples (experiments on mean component)
 
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import cvxpy as cp
from skimage.measure import compare_ssim as ssim
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from fun_iter_cf import *
import sys 

# Download network
model = tf.keras.applications.MobileNetV2(include_top=True,
                                                     weights='imagenet')
model.trainable = False

# ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions


# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]


 
image_raw = tf.io.read_file('1.png') # input file

image = tf.image.decode_png(image_raw, channels=3, dtype=tf.dtypes.uint8, name=None)
image = preprocess(image)
image_probs = model.predict(image)
plt.figure()
plt.imshow(image[0]*0.5+0.5) # To change [-1, 1] to [0,1]
_, image_class, class_confidence = get_imagenet_label(image_probs)
plt.title('{}'.format(image_class))
plt.axis('off')	
plt.show()

# Get the input label of the image.
label = np.argmax(image_probs)

# gradient check
gradient = create_grad_cw(model,image, label)[0].numpy()
G_vec = gradient.reshape(-1,1)

if ((np.max(G_vec) == 0) & (np.max(G_vec) == 0)):
    sys.exit('Gradients are zero')
    #print("Gradients are zero")


# Generate adversarial example
iter = 20
e = 0.005
adv_image = iter_PGA_cf(model, image, label, e,iter)


print("Adversarial example:")
image_probs_adv = model.predict(adv_image)
y_adv = np.argmax(image_probs_adv)
print("Groundtruth and Predicted output of the model: ", label, y_adv)

# SSIM between the two images 
qual, qual_grad,qual_map = ssim(image[0].numpy().reshape(224,224,3),adv_image[0].numpy().reshape(224,224,3),multichannel=True, gradient = True, full = True)  
print("SSIM value between original adversarial sample is: ",qual)



plt.figure()
plt.imshow(0.5*adv_image[0]+0.5) # To change [-1, 1] to [0,1]
_, image_class_adv, class_confidence_adv = get_imagenet_label(image_probs_adv)
plt.title('{}, SSIM: {:.2f}, e: {:.4f}'.format(image_class_adv, qual,e))
plt.axis('off')
plt.show()
