# Imports

import json
import os

import cv2
from skimage import filters
from einops import rearrange
import statistics
import seaborn as sns; sns.set_theme()

import random
import traceback
import nibabel as nib
import scipy 

import numpy as np
from numpy import save
import matplotlib.pyplot as plt
import time
from datetime import datetime

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from tensorflow.keras import layers
from tensorflow.keras import backend as K
from plot_keras_history import plot_history

from sklearn.model_selection import ParameterGrid
from sklearn import metrics

from scripts.evalresults import *
from scripts.utils import *
    


# Model implementation: Variational Autoencoder

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon   
    
    

def VAE():

  input_img = tf.keras.Input(shape=(image_size, image_size, num_channels))    
  adversarial_inp = tf.expand_dims(input_img[:, :, :, 0], axis=-1)
  normal_inp = tf.expand_dims(input_img[:, :, :, 1], axis=-1)
  
  
  # Encoding the normal input  
  x_norm = layers.Conv2D(32 , 5, activation=layers.LeakyReLU(), strides=2, padding="same")(normal_inp)
  x_norm = layers.Conv2D(64 , 5, activation=layers.LeakyReLU(), strides=2, padding="same")(x_norm)
  x_norm = layers.Conv2D(128, 5, activation=layers.LeakyReLU(), strides=2, padding="same")(x_norm)    
  x_norm = layers.Conv2D(128, 5, activation=layers.LeakyReLU(), strides=2, padding="same")(x_norm)   
  x_norm = layers.Conv2D(16 , 1, activation=layers.LeakyReLU(), strides=1, padding="same")(x_norm)   
  x_norm = layers.Flatten()(x_norm)
  encoded_norm = layers.Dense(intermediate_dim, activation=layers.LeakyReLU())(x_norm)   
  z_mean_norm = layers.Dense(latent_dim, name="z_mean")(encoded_norm)
  z_log_var_norm = layers.Dense(latent_dim, name="z_log_var")(encoded_norm)
  z_norm = Sampling()([z_mean_norm, z_log_var_norm])
  
  
  # Encoding the adversarial input
  x_adv = layers.Conv2D(32 , 5, activation=layers.LeakyReLU(), strides=2, padding="same")(adversarial_inp)
  x_adv = layers.Conv2D(64 , 5, activation=layers.LeakyReLU(), strides=2, padding="same")(x_adv)
  x_adv = layers.Conv2D(128, 5, activation=layers.LeakyReLU(), strides=2, padding="same")(x_adv)    
  x_adv = layers.Conv2D(128, 5, activation=layers.LeakyReLU(), strides=2, padding="same")(x_adv)   
  x_adv = layers.Conv2D(16 , 1, activation=layers.LeakyReLU(), strides=1, padding="same")(x_adv)   
  x_adv = layers.Flatten()(x_adv)
  encoded_adv = layers.Dense(intermediate_dim, activation=layers.LeakyReLU())(x_adv)    
  z_mean_adv = layers.Dense(latent_dim, name="z_mean")(encoded_adv)
  z_log_var_adv = layers.Dense(latent_dim, name="z_log_var")(encoded_adv)
  z_adv = Sampling()([z_mean_adv, z_log_var_adv])
  
    
  # Decoding the adversarial input
  d_adv = layers.Dense(16 * 16 * 16, activation=layers.LeakyReLU())(z_adv)
  d_adv = layers.Reshape((16, 16, 16))(d_adv)
  d_adv = layers.Conv2D(128, 1, strides=1, activation=layers.LeakyReLU(), padding="same")(d_adv)    
  d_adv = layers.Conv2DTranspose(128, 5, strides=2, activation=layers.LeakyReLU(), padding="same")(d_adv) 
  d_adv = layers.Conv2DTranspose(64 , 5, strides=2, activation=layers.LeakyReLU(), padding="same")(d_adv)
  d_adv = layers.Conv2DTranspose(32 , 5, strides=2, activation=layers.LeakyReLU(), padding="same")(d_adv)
  d_adv = layers.Conv2DTranspose(32 , 5, strides=2, activation=layers.LeakyReLU(), padding="same")(d_adv)
  decoded_adv = layers.Conv2D(1, 1, activation=layers.LeakyReLU(), padding='same')(d_adv)
  
  
  model = tf.keras.Model(input_img, decoded_adv)
  model.compile(optimizer=opt, loss=AE_loss(gamma, z_norm, z_adv), metrics=['mse'], experimental_run_tf_function=False)
  
  return model
  


# Autoencoder loss function

def AE_loss(gamma, norm_latent, adv_latent):

    def loss(y_true, y_pred):
        # AE loss which is the weighted sum of rec. loss and latent loss
        return K.mean(K.square(y_pred - y_true), axis=-1) + gamma * K.mean(K.square(norm_latent - adv_latent), axis=-1)

    return loss
    
    
        
# Configure the hyperparameters

model_name = 'Variational Autoencoder'
numEpochs = 50
learning_rate = 0.00001
rate = 0.  
image_size = 256
batch_size = 1
latent_dim = 512
intermediate_dim = 2048

num_channels = 2
gamma = 0.1



# Configure training and testing sets 

test_mslub_path = './data/MSLUB_Flair_2c.npy'
test_brats_path = './data/BraTS_Flair_2c.npy'


data_dir  = './data/OASIS_adv/'
train_paths = list_of_paths(data_dir)

nb_train_files = 66
data_gen = data_generator_2c(train_paths[:nb_train_files], batch_size)
training_steps = (256 / batch_size) * nb_train_files

nb_val_files = 5
val_gen = data_generator_2c(train_paths[-nb_val_files:], batch_size)
validation_steps = (256 / batch_size) * nb_val_files


ckpts_dir = './saved/VAE_adv/'     
ckpts_path = os.path.join(ckpts_dir, 'Model_Ckpts.h5')
fig_path = os.path.join(ckpts_dir, 'History_plot.png')
predicted_oasis_path = os.path.join(ckpts_dir, 'Predicted_oasis.npy')
predicted_mslub_path = os.path.join(ckpts_dir, 'Predicted_mslub.npy')
predicted_brats_path = os.path.join(ckpts_dir, 'Predicted_brats.npy')


      
# Configure the training
opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
          
calbks = tf.keras.callbacks.ModelCheckpoint(filepath=ckpts_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=2)
tqdm_callback = tfa.callbacks.TQDMProgressBar() 

model = VAE()
model.summary()
#model.load_weights('./saved/VAE/ckpt/Model_Ckpts_2.h5')

# Print & Write model Parameters
parameters = (f'\nSelected model "{model_name}" with :\n - {batch_size}: Batche(s)\n - {numEpochs}: Epochs\n - {intermediate_dim}: Bottelneck size\n')
print(parameters)

 

# TRAIN 
print('\nTrain =>\n')
history = model.fit(x = data_gen,
                    steps_per_epoch = training_steps,
                    validation_data = val_gen,
                    validation_steps = validation_steps,
                    verbose = 0,
                    epochs = numEpochs,
                    callbacks = [calbks, tqdm_callback]
                    )
                          
# Get training and test loss histories                   
plot_history(history, path=fig_path)
plt.close()
time.sleep(2)



#-- Load Test-sets       
print('\nLoad Test-sets ===>\n')
oasis_test_2c = np.load(test_oasis_path)
mslub_test_2c = np.load(test_mslub_path)
brats_test_2c = np.load(test_brats_path)


#-- Predict MSLUB
print('\nPredict =====>\n')
steps = mslub_test_2c.shape[0]
batch_size = 1
predicted = model.predict(x=data_gen(mslub_test_2c, batch_size), steps=steps//batch_size)
np.save(predicted_mslub_path, predicted)
time.sleep(4)


#-- Predict BraTS
print('\nPredict =====>\n')
steps = brats_test_2c.shape[0]
batch_size = 1
predicted = model.predict(x=data_gen(brats_test_2c, batch_size), steps=steps//batch_size)
np.save(predicted_brats_path, predicted)
time.sleep(4)

