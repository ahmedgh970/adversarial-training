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



list_ckpts_dir = ['./saved/DCAE/Predicted/mslub/', './saved/VAE/Predicted/mslub/'] 

for l in list_ckpts_dir:

  ckpts_dir = l
  results_path = os.path.join(ckpts_dir, 'Results.txt')
  dice_plot_path = os.path.join(ckpts_dir, 'Dice_plot.png')
  predicted_path = os.path.join(ckpts_dir, 'Predicted_mslub.npy')
  residual57_path = os.path.join(ckpts_dir, 'Residuals57.npy')
  residual84_path = os.path.join(ckpts_dir, 'Residuals84.npy')
  residual_BP_path = os.path.join(ckpts_dir, 'Residuals_BP.npy')

  test_path = './data/MSLUB/MSLUB_Flair.npy'
  prior57_path = './data/MSLUB/MSLUB_prior_57.npy'
  prior84_path = './data/MSLUB/MSLUB_prior_84.npy'
  label_path = './data/MSLUB/MSLUB_GT.npy'
  brainmask_path = './data/MSLUB/MSLUB_Brainmask.npy'

  # Test       
  print('\nTest ===>\n')
  predicted = np.load(predicted_path)
  my_test = np.load(test_path)
  brainmask = np.load(brainmask_path)
  my_labels = np.load(label_path)
  prior57 = np.load(prior57_path)
  prior84 = np.load(prior84_path)

  #-- Calculate, Post-process and Save Residuals
  print('\nCalculate, Post-process and Save Residuals =====>\n')     
  residual_BP = calculate_residual_BP(my_test, predicted, brainmask)
  np.save(residual_BP_path, residual_BP)
        
  residual_57 = calculate_residual(my_test, predicted, prior57)
  np.save(residual57_path, residual_57)

  residual_84 = calculate_residual(my_test, predicted, prior84)
  np.save(residual84_path, residual_84)
        

  #-- Evaluation
  print('\nEvaluate =========>\n')        
  [AUROC, AUPRC, AVG_DICE, MAD, STD], DICE = eval_residuals(my_labels, residual_57)     
  results_57 = (f'\nResults after median_filter and x_prior57 :\n - AUROC = {AUROC}\n - AUPRC = {AUPRC}\n - AVG_DICE = {AVG_DICE}\n - MEDIAN_ABSOLUTE_DEVIATION = {MAD}\n - STANDARD_DEVIATION = {STD}')
  print(results_57)
  
  [AUROC, AUPRC, AVG_DICE, MAD, STD], DICE = eval_residuals(my_labels, residual_84)     
  results_84 = (f'\nResults after median_filter and x_prior84 :\n - AUROC = {AUROC}\n - AUPRC = {AUPRC}\n - AVG_DICE = {AVG_DICE}\n - MEDIAN_ABSOLUTE_DEVIATION = {MAD}\n - STANDARD_DEVIATION = {STD}')
  print(results_84)
      
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
  f.write(results_57)
  f.write(results_84)       
  f.close()   
       
  print('\nEnd of evaluation !\n')

