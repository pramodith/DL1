import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  pass
  out=np.dot(x.reshape(-1,w.shape[0]),w)+b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  db=np.sum(dout,axis=0)
  dw=np.dot(dout.T,cache[0].reshape(-1,cache[1].shape[0]))
  dx=np.dot(dout,cache[1].T).reshape(cache[0].shape)
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw.T, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = x.copy()
  out[out<0]=0
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  grads=np.copy(x)
  grads[x<0]=0
  grads[x>0]=1
  dx=dout*grads
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  x_padded=np.pad(x,((0,0),(0,0),(conv_param['pad'],conv_param['pad']),(conv_param['pad'],conv_param['pad'])),mode='constant',constant_values=0)
  output_image = np.zeros((x_padded.shape[0],w.shape[0],1+(x.shape[2]+2*conv_param['pad']-w.shape[2])//conv_param['stride'],1+(x.shape[3]+2*conv_param['pad']-w.shape[3])//conv_param['stride']))
  ### END OF STUDENT CODE ####
  ############################
  for n in range(0,x.shape[0]):
    for k in range(0,w.shape[0]):
      for out_i,i in enumerate(range(w.shape[2] // 2, x.shape[2] + w.shape[2] // 2,conv_param['stride'])):
        for inp_i,j in enumerate(range(w.shape[3] // 2, x.shape[3] + w.shape[3] // 2,conv_param['stride'])):
            try:
              output_image[n,k,out_i,inp_i]= np.sum(x_padded[n,:,i - w.shape[2] // 2:i + int(np.ceil(w.shape[2]/ 2)),j - w.shape[3] // 2:j + int(np.ceil(w.shape[3]/2))]*w[k,:,:,:])+b[k]
            except Exception as e:
              print(e)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  out=output_image
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  padded_x=np.pad(cache[0],((0,0),(0,0),(cache[3]['pad'],cache[3]['pad']),(cache[3]['pad'],cache[3]['pad'])),'constant',constant_values=0)
  dw=np.zeros(cache[1].shape)
  dx=np.zeros(padded_x.shape)
  db=np.zeros(cache[1].shape[0])
  for n in range(cache[0].shape[0]):
    for k in range(cache[1].shape[0]):
      db[k] += np.sum(dout[n][k])
      for c in range(cache[1].shape[1]):
        for cnt_i,i in enumerate(range(0,padded_x.shape[2]-int(np.ceil(cache[1].shape[2]/2)),cache[3]['stride'])):
          for cnt_j,j in enumerate(range(0,padded_x.shape[3]-int(np.ceil(cache[1].shape[3]/2)),cache[3]['stride'])):
            dw[k][c]+=padded_x[n,c,i:i+cache[1].shape[2],j:j+cache[1].shape[3]]*dout[n][k][cnt_i][cnt_j]
            dx[n,c,i:i+cache[1].shape[2],j:j+cache[1].shape[3]]+=cache[1][k][c]*dout[n][k][cnt_i][cnt_j]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  dx=dx[:,:,cache[3]['pad']:dx.shape[2]-cache[3]['pad'],cache[3]['pad']:dx.shape[3]-cache[3]['pad']]
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  output=np.zeros((x.shape[0],x.shape[1],1+(x.shape[2]-pool_param['pool_height'])//pool_param['stride'],1+(x.shape[3]-pool_param['pool_width'])//pool_param['stride']))
  for n in range(0,x.shape[0]):
    for c in range(0,x.shape[1]):
      for cnt_i,i in enumerate(range(0,x.shape[2]-pool_param['pool_height']+1,pool_param['stride'])):
        for cnt_j,j in enumerate(range(0,x.shape[3]-pool_param['pool_width']+1,pool_param['stride'])):
          output[n][c][cnt_i][cnt_j]=np.max(x[n,c,i:i+pool_param['pool_height'],j:j+pool_param['pool_width']])
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  out=output.copy()
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  x=cache[0]
  dx = np.zeros(x.shape)
  pool_param=cache[1]
  for n in range(0,x.shape[0]):
    for c in range(0,x.shape[1]):
      for cnt_i,i in enumerate(range(0,x.shape[2]-pool_param['pool_height']+1,pool_param['stride'])):
        for cnt_j,j in enumerate(range(0,x.shape[3]-pool_param['pool_width']+1,pool_param['stride'])):
          loc=np.unravel_index(np.argmax(np.ravel(x[n,c,i:i+pool_param['pool_height'],j:j+pool_param['pool_width']])),(pool_param['pool_height'],pool_param['pool_width']))
          loc=list(loc)
          loc[0]+=i
          loc[1]+=j
          dx[n][c][loc[0]][loc[1]]+=dout[n,c,cnt_i,cnt_j]
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx


