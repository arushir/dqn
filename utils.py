import os
import argparse

def parse_args():
  """
  Parses the command line input.

  """
  parser = argparse.ArgumentParser()
  parser.add_argument('-l', default = DEFAULT_LR, help = 'learning rate', type=float)
  parser.add_argument('-r', default = DEFAULT_REG, help = 'regularization', type=float)
  parser.add_argument('-e', default = DEFAULT_NB_EPOCH, help = 'number of epochs', type=int)

  args = parser.parse_args()
  params = {'lr': args.l, 'reg': args.r, 'nb_epoch': args.e}
  return params