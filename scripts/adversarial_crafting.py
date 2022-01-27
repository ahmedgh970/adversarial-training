# Imports

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import KerasClassifier
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
from plot_keras_history import plot_history
import warnings
warnings.filterwarnings('ignore')


# Load my train and validation sets with labels
x_train = np.load('./data/OASIS/healthy_oasis/x_train.npy')
y_train = np.load('./data/OASIS/healthy_oasis/y_train.npy')


# All data shape: (H=256, W=256, C=1)

# Define the estimator (classifier) to generate the PGD attack 
model = tf.keras.applications.ResNet50(
  include_top=True,
  weights=None,
  input_tensor=None,
  input_shape=(256, 256,1),
  pooling=max,
  classes=3,
  classifier_activation='softmax'
)


# Compile
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']);  

#model.load_weights('./saved/ResNet50/ckpt/Model_Ckpts_train.h5')



#------------------- Train the classifier--------------------------------------
# Callbacks          
calbks = tf.keras.callbacks.ModelCheckpoint(filepath='./saved/ResNet50/ckpt/Model_Ckpts_train.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)
 

# Train
hist = model.fit(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[calbks]);
plot_history(hist, path='./saved/ResNet50/ckpt/History.png')
plt.close()


# Test
#loss_test, accuracy_test = model.evaluate(x_test, y_test)
#print('Accuracy on testing-set: {:4.2f}%'.format(accuracy_test * 100))
#-------------------------------------------------------------------------------




#------------------- Craft the adv samples -------------------------------------
# Create a ART keras classifier for the TensorFlow Keras model
classifier = KerasClassifier(model=model, clip_values=(0, 1))


# Get PGD attack on Guassian process classification
eps = 0.20
attack = ProjectedGradientDescent(classifier,
                                  eps=eps,
                                  eps_step=eps/10,
                                  targeted=True,
                                  batch_size=64)


# Generate adv examples
x_train_adv = attack.generate(x_train)
np.save('./data/OASIS/adv_oasis/x_train_adv.npy', x_train_adv)


# Evaluate on adversarial train data
loss_test, accuracy_test = model.evaluate(x_train_adv, y_train)
perturbation = np.mean(np.abs((x_train_adv - x_train)))
print(f'With eps=0.2 we have:\n')
print('Accuracy on adversarial train data: {:4.2f}%'.format(accuracy_test * 100))
print('Average perturbation: {:4.2f}\n'.format(perturbation))
#------------------------------------------------------------------------------


