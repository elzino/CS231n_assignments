import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  for i in range(num_train):
    Xi = np.reshape(X[i], (1, -1))
    score = np.dot(Xi, W)
    max_value = np.max(score)
    score -= max_value
    score = np.exp(score)
    score_sum = np.sum(score)
    loss -= (np.log(score[0, y[i]] / score_sum))
    dW += np.dot(Xi.T, (score / score_sum))
    dW[:, y[i]] -= Xi.flatten()
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(np.square(W))
  dW += (reg * 2 * W)

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_train = X.shape[0]
  scores = np.dot(X, W)
  max_values = np.max(scores, axis = 1, keepdims = True)
  scores -= max_values
  scores_exp = (np.exp(scores))
  scores_sum = np.sum(scores_exp, axis = 1)
  loss -= np.sum(np.log(scores_exp[np.arange(num_train), y] / scores_sum)) / num_train
  loss += reg * np.sum(np.square(W))
  y_onehot = np.zeros(scores_exp.shape)
  y_onehot[np.arange(num_train), y] = 1
  scores_sum = scores_sum.reshape(-1, 1) # to convert vector to array with dimension (num_train , 1)
  dW += np.dot(X.T, ((scores_exp/ scores_sum) - y_onehot))
  dW /= num_train
  dW += (reg * 2 * W)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

