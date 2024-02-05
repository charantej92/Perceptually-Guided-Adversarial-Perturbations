#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:09:18 2020

@author: charan
"""
#Functions useful for generating adversarial examples
import tensorflow as tf
import numpy as np
import cvxpy as cp

# Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = image[None, ...]
  return image

def prep_adv(image):
  image = tf.cast(image, tf.float32)
  image = image[None, ...]
  return image


# Function to calculate gradients with loss:CW
def cw_loss(label, logits):
   depth = 1000    
   label_mask = tf.one_hot(label, depth) 
   correct_logit = tf.reduce_sum(label_mask * logits, axis=1)
   wrong_logit = tf.reduce_max((1-label_mask) * logits - 1e4*label_mask, axis=1)
   loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
   return loss

def create_grad_cw(model, input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = model(input_image)
    loss = cw_loss(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  return gradient


def ssim_opt(I,G,eps_2):
  N = I.shape[0]
  c_1 = (0.01)**2 # ssim parameter
  c_2 = (0.03)**2 # ssim parameter
  #print(np.mean(I))
  i_m = np.mean(I) 
  I = I- np.mean(I) * np.ones(I.shape)
  k2_1 = np.sqrt(((np.linalg.norm(I,2)**2))*((1/(1-eps_2**2)**2)-1)+(c_2)*(eps_2**2/(1-eps_2**2)))
  k2_2 = (I)*(1/(1-eps_2**2))
  x = cp.Variable(G.shape)
  y = cp.Variable(1)
  # Create two constraints.
  constraints =[cp.norm(x-y-k2_2, 2) <= k2_1,
                  x>= -1.0, 
                  x<= 1.0,
                  cp.sum(x) == N*y,
                  cp.sum(x) == N*i_m]
               
  # Form and solve problem.                
  prob = cp.Problem(cp.Maximize(G.T*x), constraints)
  # try:   
  #   result = prob.solve() 
  # except SolverError:
  #   result = prob.solve(solver=SCS)
  print('optimal value', prob.solve(solver=cp.SCS))
  #print("Dual variable values: ",constraints[0].dual_value)
  if x.value is None:
    return I
  else:
    return x.value


### Closed Form Solution ###

def ssim_cf(I,G,eps_2):
  #N = I.shape[0]
  #c_1 = (0.01)**2 # ssim parameter
  c_2 = (0.03)**2 # ssim parameter
  i_m = np.mean(I)
  I = I- np.mean(I) * np.ones(I.shape)
  k_1 = np.sqrt(((np.linalg.norm(I,2)**2))*((1/(1-eps_2**2)**2)-1)+(c_2)*(eps_2**2/(1-eps_2**2)))
  k_2 = (I)*(1/(1-eps_2**2))
  
  alpha_1 = i_m * np.ones(I.shape) +k_2
  alpha_2 = k_1  
  C = G/np.linalg.norm(G,2)  
  #x = alpha_1 - alpha_2 * C
  x = alpha_1 + alpha_2 * C  
  return x


def create_adversarial_ssim_cf(gradient, image, label,eps_2):
  #I = 0.5*np.reshape(image,(-1,1))+0.5
  I = np.reshape(image,(-1,1))
  G = np.reshape(gradient,(-1,1))
  #G = (G-np.min(G))/(np.max(G)-np.min(G))
  #G = 2*(G-0.5)
  out = ssim_cf(I, G,eps_2)
  delta_adv = np.reshape(np.clip(out, -1, 1),image.shape)
  delta_adv = tf.cast(delta_adv, tf.float32)
  return delta_adv

# Itereatively search for adversarial example
def iter_PGA_cf(model, image, label, e,iter):
  gradient = create_grad_cw(model,image, label)[0].numpy()
  for i in range(iter):
        x_adv = create_adversarial_ssim_cf(gradient, image, label, e)
        y_adv = np.argmax(model.predict(x_adv))
        
        if (np.abs(label-y_adv) > 0.5):
           print('adversarial example found at iteration: ',i)
           return x_adv
        else:
           gradient = create_grad_cw(model,x_adv, label)[0].numpy()
           image = x_adv
  return x_adv

