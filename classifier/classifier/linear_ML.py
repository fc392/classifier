import numpy as np

def  ML_loss_vectorized(W, X, y, reg):
    """
      Structured MLE loss function, based on Guassian.

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_train = X.shape[0]
    temp = X.dot(W)
    temp[np.arange(num_train), y] -= 1;
    loss = 0.5 *np.sum(np.square(temp))
    dW += X.T.dot(temp)
    return loss, dW