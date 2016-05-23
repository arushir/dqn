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

import numpy as np
import logging

class CNN:
  """
  Convolutional Neural Network model.

  """

  def __init__(self, num_actions, observation_shape, params={}, verbose=True):
    """
    Initialize the CNN model with a set of parameters.

    Args:
      params: a dictionary containing values of the models' parameters.

    """
    self.verbose = verbose
    self.num_actions = num_actions
    self.observation_shape = observation_shape
    logging.info('Initialized with params: {}'.format(params))

    if 'lr' in params:
      self.lr = params['lr']
    else:
      self.lr = 0.01

    self.model = self.create_model()

  def create_model(self):
    """
    The model definition.

    """

    model = Sequential()
    #model.add(Convolution2D(num_filters, input_shape=(self.observation_shape)))
    model.add(Dense(self.num_actions, input_shape=self.observation_shape, bias=True))
    #model.add(Activation("relu"))
    model.add(Activation("linear"))
    sgd = SGD(lr=self.lr)

    #TODO: write custom loss function to match objective in paper
    model.compile(loss='mean_squared_error', optimizer = sgd)
    return model

  def train_step(self, Xs, ys):
    """
    Updates the CNN model with a mini batch of training examples.

    """
    self.model.fit(Xs, ys, batch_size = 1, nb_epoch=1)

  def predict(self, observation):
    """
    Predicts the rewards for an input observation state. 

    Args:
      observation: a numpy array of a single observation state

    """
    # TODO: check dimensions for 2D case
    return self.model.predict(observation.reshape(1, len(observation)))






