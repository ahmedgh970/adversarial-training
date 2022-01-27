# Imports

import json
import os

import cv2
from skimage import filters
import statistics
import scipy 
from sklearn import metrics

import numpy as np
from numpy import save
import matplotlib.pyplot as plt

from scripts.evalresults import *
from scripts.utils import *



list_ckpts_dir = ['./saved/DCAE/Predicted/brats/', './saved/VAE/Predicted/brats/']     


for l in list_ckpts_dir:

  ckpts_dir = l
  results_path = os.path.join(ckpts_dir, 'Results.txt')
  dice_plot_path = os.path.join(ckpts_dir, 'Dice_plot.png')
  predicted_path = os.path.join(ckpts_dir, 'Predicted_brats.npy')
  residual_path = os.path.join(ckpts_dir, 'Residuals.npy')
  residual_BP_path = os.path.join(ckpts_dir, 'Residuals_BP.npy')

  test_path = './data/BraTS/BraTS_Flair.npy'
  prior_path = './data/BraTS/BraTS_prior_52.npy'
  label_path = './data/BraTS/BraTS_GT.npy'
  brainmask_path = './data/BraTS/BraTS_Brainmask.npy'

  # Test       
  print('\nTest ===>\n')
  predicted = np.load(predicted_path)
  my_test = np.load(test_path)
  brainmask = np.load(brainmask_path)
  my_labels = np.load(label_path)
  prior = np.load(prior_path)

  #-- Calculate, Post-process and Save Residuals
  print('\nCalculate, Post-process and Save Residuals =====>\n')     
  residual_BP = calculate_residual_BP(my_test, predicted, brainmask)
  np.save(residual_BP_path, residual_BP)
        
  residual = calculate_residual(my_test, predicted, prior)
  np.save(residual_path, residual)
        

  #-- Evaluation
  print('\nEvaluate =========>\n')        
  [AUROC, AUPRC, AVG_DICE, MAD, STD], DICE = eval_residuals(my_labels, residual)     
  results = (f'\nResults after median_filter :\n - AUROC = {AUROC}\n - AUPRC = {AUPRC}\n - AVG_DICE = {AVG_DICE}\n - MEDIAN_ABSOLUTE_DEVIATION = {MAD}\n - STANDARD_DEVIATION = {STD}')
  print(results)
      
  len_testset = my_labels.shape[0]
               
  plt.figure()
  hor_axis = [x for x in range(len_testset)]
  plt.scatter(hor_axis, DICE, s = 5, marker = '.', c = 'blue')
  plt.ylabel('Dice Score')
  plt.xlabel('N° Samples')
  plt.title('Dice scores')
  plt.savefig(dice_plot_path)
  time.sleep(2)

  #-- Save
  print('\nSave Results and Parameters =============>\n')
  f = open(results_path, "w")
  f.write(results)       
  f.close()   
       
  #-- End
  print('\nEnd of evaluation !\n')

