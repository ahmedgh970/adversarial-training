from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import KerasClassifier
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



def mean_intensity_per_sample(arr):
    tot = np.float(np.sum(arr))
    return tot/arr.size

       
# load numpy
train = np.load('./OASIS/healthy_oasis/x_train.npy')

# test shapes
print(train.shape[0])



# create labels

label_train = np.zeros((train.shape[0],), dtype=int)

thresh1 = 0.0475
thresh2 = 0.1300

for i in range(train.shape[0]):
  if mean_intensity_per_sample(train[i])<thresh1 :
    label_train[i] = 0
  elif mean_intensity_per_sample(train[i])>=thresh1 and mean_intensity_per_sample(train[i])<thresh2:
    label_train[i] = 1
  elif mean_intensity_per_sample(train[i])>=thresh2:
    label_train[i] = 2

np.save('./OASIS/healthy_oasis/y_train.npy', label_train)


# to verify if the train-set is balanced
count0 = 0
count1 = 0
count2 = 0

for i in range(label_train.shape[0]):
  if label_train[i] == 0:
    count0 += 1
  elif label_train[i] == 1:
    count1 += 1
  else:
    count2 += 1

print(count0)
print(count1)
print(count2)

    