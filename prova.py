import tensorflow as tf
import numpy as np
from keras.losses import losses_utils
import keras.backend as K

def Custom_Hamming_Loss(y_true, y_pred):
  return K.mean(y_true*(1-y_pred)+(1-y_true)*y_pred, axis=1)

y_pred, y_true = np.zeros((10, 5, 1)), np.zeros((10, 5, 1))
y_pred[0][0] = 1

y_true = tf.constant(y_true)
y_pred = tf.constant(y_pred)

print(Custom_Hamming_Loss( y_true, y_pred).numpy())

