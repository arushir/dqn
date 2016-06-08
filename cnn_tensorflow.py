import tensorflow as tf
import numpy as np
import logging

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

    # TODO: observation shape will be a tuple
    self.observation_shape = observation_shape[0]
    logging.info('Initialized with params: {}'.format(params))

    self.lr = params['lr']
    self.reg = params['reg']
    self.num_hidden = params['num_hidden']
    self.hidden_size = params['hidden_size']
    #self.mini_batch_size = params['mini_batch_size']    

    self.session = self.create_model()


  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training. 

    Adds following nodes to the computational graph

    input_placeholder: Input placeholder tensor of shape
                       (None, window_size), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, label_size), type tf.float32
    dropout_placeholder: Dropout value placeholder (scalar),
                         type tf.float32

    Add these placeholders to self as the instance variables
  
      self.input_placeholder
      self.labels_placeholder
      self.dropout_placeholder

    (Don't change the variable names)
    """
    # TODO: How to define a 3d observation shape
    input_placeholder = tf.placeholder(tf.float32, shape=(None, self.observation_shape))
    #labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.num_actions))
    labels_placeholder = tf.placeholder(tf.float32, shape=(None,))
    actions_placeholder = tf.placeholder(tf.float32, shape=(None, self.num_actions))

    return input_placeholder, labels_placeholder, actions_placeholder


  def nn(self, input_obs):
    with tf.name_scope("Layer") as scope:
      Wshape = [self.observation_shape, self.hidden_size]
      W = tf.get_variable("W", shape=Wshape,)
      bshape = [1, self.hidden_size]
      b1 = tf.get_variable("b1", shape=bshape, initializer = tf.constant_initializer(0.0))

    with tf.name_scope("Softmax") as scope:
      Ushape = [self.hidden_size, self.num_actions]
      U = tf.get_variable("U", shape=Ushape)
      b2shape = [1, self.num_actions]
      b2 = tf.get_variable("b2", shape=b2shape, initializer = tf.constant_initializer(0.0))

    xW = tf.matmul(input_obs, W)
    h = tf.tanh(tf.add(xW, b1))
    hU = tf.matmul(h, U)    
    out = tf.add(hU, b2)
    return out


  def create_model(self):
    """
    The model definition.

    """
    self.input_placeholder, self.labels_placeholder, self.actions_placeholder = self.add_placeholders()
    outputs = self.nn(self.input_placeholder)

    self.predictions = outputs
    #self.predictions = tf.nn.softmax(outputs)

    self.q_vals = tf.reduce_sum(tf.mul(self.predictions, self.actions_placeholder), 1)


    #self.loss = tf.reduce_mean(tf.square(self.predictions - self.q_vals))
    self.loss = tf.reduce_sum(tf.square(self.labels_placeholder - self.q_vals))


    #optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.lr)

    self.train_op = optimizer.minimize(self.loss)
    init = tf.initialize_all_variables()
    #saver = tf.train.Saver()
    session = tf.Session()
    session.run(init)

    return session

  def train_step(self, Xs, ys, actions):
    """
    Updates the CNN model with a mini batch of training examples.

    """

    loss, _, prediction_probs, q_values = self.session.run(
      [self.loss, self.train_op, self.predictions, self.q_vals],
      feed_dict = {self.input_placeholder: Xs,
                  self.labels_placeholder: ys,
                  self.actions_placeholder: actions
                  })


    # print "loss: ", loss

    # print "prediction_probs: "
    # print prediction_probs

    # print "actions: "
    # print actions

    # print "q values: "
    # print q_values


  def predict(self, observation):
    """
    Predicts the rewards for an input observation state. 

    Args:
      observation: a numpy array of a single observation state

    """

    loss, prediction_probs = self.session.run(
      [self.loss, self.predictions],
      feed_dict = {self.input_placeholder: observation,
                  self.labels_placeholder: np.zeros(len(observation)),
                  self.actions_placeholder: np.zeros((len(observation), self.num_actions))
                  })

    # print "prediction probabilities: "
    # print prediction_probs

    return prediction_probs


