# NOTE: loss function needs to be re-written to run with updated dqn.py

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.callbacks import History
from keras.regularizers import l2, activity_l2
from keras import optimizers

import theano.tensor as T
from theano import pp, printing
import theano

import numpy as np
import logging

def custom_loss(y_true, y_pred):
  """ 
  Custom loss funciton in Theano to correspond to the objective in 
  the paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

  """
  idx = T.argmax(y_true, axis=1)
  print(pp(idx))
  printing.Print(idx)
  printing.Print('idx')(idx)
  return T.sqr(y_true[idx] - y_pred[idx])
  #return T.sqr(T.max(y_true) - T.max(y_pred))

class CNN:
  """
  Convolutional Neural Network model.

  """

  def __init__(self, num_actions, observation_shape, params={}, verbose=False):
    """
    Initialize the CNN model with a set of parameters.

    Args:
      params: a dictionary containing values of the models' parameters.

    """
    self.verbose = verbose
    self.num_actions = num_actions
    self.observation_shape = observation_shape
    logging.info('Initialized with params: {}'.format(params))

    self.lr = params['lr']
    self.reg = params['reg']
    self.num_hidden = params['num_hidden']
    self.hidden_size = params['hidden_size']

    self.model = self.create_model()

  def create_model(self):
    """
    The model definition.

    """

    model = Sequential()
    for i in xrange(self.num_hidden):
      if i == 0:
        model.add(Dense(self.hidden_size, input_shape=self.observation_shape, W_regularizer=l2(self.reg), bias=True))
      else:
        model.add(Dense(self.hidden_size, W_regularizer=l2(self.reg), bias=True))
      model.add(Activation("relu"))

    if self.num_hidden == 0:
      model.add(Dense(self.num_actions, input_shape=self.observation_shape, W_regularizer=l2(self.reg), bias=True))
    else: 
      model.add(Dense(self.num_actions, W_regularizer=l2(self.reg), bias=True))
    model.add(Activation("relu"))

    sgd = SGD(lr=self.lr)

    model.compile(loss=custom_loss, optimizer=sgd)
    return model

  def train_step(self, Xs, ys):
    """
    Updates the CNN model with a mini batch of training examples.

    """
    print "IN TRAIN STEP"
    print Xs
    print ys
    hist = self.model.fit(Xs, ys, batch_size=len(Xs), nb_epoch=1, verbose=0)
    print(hist.history)['loss']
    print "DONE"

  def predict(self, observation):
    """
    Predicts the rewards for an input observation state. 

    Args:
      observation: a numpy array of a single observation state

    """
    # TODO: check dimensions for 2D case
    return self.model.predict(observation.reshape(1, len(observation)))






