import numpy as np
np.random.seed(1)

def sigmoid(z):
  """
  sigmoid function that maps inputs into the interval [0,1]
  Your implementation must be able to handle the case when z is a vector (see unit test)
  Inputs:
  - z: a scalar (real number) or a vector
  Outputs:
  - trans_z: the same shape as z, with sigmoid applied to each element of z
  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  trans_z = 1/(1 + np.exp(-z))
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return trans_z

def logistic_regression(X, w):
  """
  logistic regression model that outputs probabilities of positive examples
  Inputs:
  - X: an array of shape (num_sample, num_features)
  - w: an array of shape (num_features,)
  Outputs:
  - logits: a vector of shape (num_samples,)
  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  logits = sigmoid(X @ w)
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return logits

def logistic_loss(X, w, y):
  """
  a function that compute the loss value for the given dataset (X, y) and parameter w;
  It also returns the gradient of loss function w.r.t w
  Here (X, y) can be a set of examples, not just one example.
  Inputs:
  - X: an array of shape (num_sample, num_features)
  - w: an array of shape (num_features,)
  - y: an array of shape (num_sample,), it is the ground truth label of data X
  Output:
  - loss: a scalar which is the value of loss function for the given data and parameters
  - grad: an array of shape (num_featues,), the gradient of loss 
  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  logReg = logistic_regression(X, w)
  loss = np.sum(-y*np.log(logReg) - (1-y)*np.log(1-logReg))/len(y)

  grad = np.dot(X.T, (logReg-y)/y.shape[0])
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return loss, grad


def softmax(x):
  """
  Convert logits for each possible outcomes to probability values.
  In this function, we assume the input x is a 2D matrix of shape (num_sample, num_classes).
  So we need to normalize each row by applying the softmax function.
  Inputs:
  - x: an array of shape (num_sample, num_classse) which contains the logits for each input
  Outputs:
  - probability: an array of shape (num_sample, num_classes) which contains the
                 probability values of each class for each input
  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  probability = np.array([np.empty([x.shape[1]]) for i in range(x.shape[0])])
  for i in range(x.shape[0]): 
    probability[i] = np.exp(x[i]) / np.sum(np.exp(x[i])) 
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return probability

def MLR(X, W):
  """
  performs logistic regression on given inputs X
  Inputs:
  - X: an array of shape (num_sample, num_feature)
  - W: an array of shape (num_feature, num_class)
  Outputs:
  - probability: an array of shape (num_sample, num_classes)
  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  probability = softmax(X @ W)
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return probability

def cross_entropy_loss(X, W, y):
  """
  Inputs:
  - X: an array of shape (num_sample, num_feature)
  - W: an array of shape (num_feature, num_class)
  - y: an array of shape (num_sample,)
  Ouputs:
  - loss: a scalar which is the value of loss function for the given data and parameters
  - grad: an array of shape (num_featues, num_class), the gradient of the loss function 
  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  n = len(y)
  oneHot = np.zeros((n, W.shape[1]), dtype=int)
  for i, val in enumerate(y):
    oneHot[i][val] = 1

  mlr = MLR(X,W)
  loss = -np.sum(oneHot*(np.log(mlr)))/n

  grad = np.dot(X.T, (mlr - oneHot))/n

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return loss, grad


def gini_score(groups, classes):
  '''
  Inputs: 
  groups: 2 lists of examples. Each example is a list, where the last element is the label.
  classes: a list of different class labels (it's simply [0.0, 1.0] in this problem)
  Outputs:
  gini: gini score, a real number
  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  num_of_groups = 2;
  num_of_classes = len(classes);
  ginindex= np.zeros((2,1));
  #initialize w array
  firstgroupnumexamples = len(groups[0]);
  secondgroupnumexamples = len(groups[1]);
  datalistnumexamples = firstgroupnumexamples+secondgroupnumexamples;
  w = np.zeros((2, 1), dtype = np.float64);
  #calculate the w array
  w[0] = firstgroupnumexamples/datalistnumexamples;
  w[1] = secondgroupnumexamples/datalistnumexamples;
  #initialize the P_i_j array
  P_i_j = np.zeros((2,2));
  #calculate P_i_j
  for i in range(0, num_of_classes):
    for j in range(0, num_of_groups):
      examplesofi = 0;
      for k in range(0, len(groups[j])):
        if(groups[j][k][-1]==classes[i]):
          examplesofi+=1;
      if(len(groups[j]) == 0):
        P_i_j[i][j] = 0;
      else:
        P_i_j[i][j] = examplesofi/len(groups[j]);
  #count all samples at split point
  #calculate gini indexes
  gini = 0;
  for j in range(0, num_of_groups):
    #calculate sum for each group
    sum = 0;
    for i in range(0, num_of_classes):
      sum += np.power(P_i_j[i][j], 2);
    ginindex[j] = 1-sum;
    #sum weighted Gini index for each group
    gini+=float(w[j])*float(ginindex[j]);

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return gini

def create_split(index, threshold, datalist):
  '''
  Inputs:
  index: The index of the feature used to split data. It starts from 0.
  threshold: The threshold for the given feature based on which to split the data.
        If an example's feature value is < threshold, then it goes to the left group.
        Otherwise (>= threshold), it goes to the right group.
  datalist: A list of samples. 
  Outputs:
  left: List of samples
  right: List of samples
  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  left = list();
  right = list();
  for i in range(0, len(datalist)):
    if(datalist[i][index] < threshold):
      left.append(datalist[i]);
    else:
      right.append(datalist[i]);
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return left, right

def get_best_split(datalist):
  '''
  Inputs:
  datalist: A list of samples. Each sample is a list, the last element is the label.
  Outputs:
  node: A dictionary contains 3 key value pairs, such as: node = {'index': integer, 'value': float, 'groups': a tuple contains two lists of examples}
  Pseudo-code:
  for index in range(#feature): # index is the feature index
    for example in datalist:
      use create_split with (index, example[index]) to divide datalist into two groups
      compute the Gini index for this division
  construct a node with the (index, example[index], groups) that corresponds to the lowest Gini index
  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  best_giniscore = 1000000;
  classes = [0,1]
  for index in range(0, len(datalist[0])-1): # index is the feature index
    for example in datalist:
      result = create_split(index, example[index], datalist);
      giniscore = gini_score(result, classes);
      if(giniscore < best_giniscore):
        best_giniscore = giniscore;
        node = dict(index=index, 
                value=example[index], 
                groups=result)
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return node

def to_terminal(group):
  '''
  Input:
    group: A list of examples. Each example is a list, whose last element is the label.
  Output:
    label: the label indicating the most common class value in the group
  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  classes = np.zeros((2,1));
  classes[0] = 0;
  classes[1] = 0;
  for i in range(0, len(group)):
    if(group[i][-1] == 0):
      classes[0]+=1;
    else:
      classes[1]+=1;
  if(classes[0]>classes[1]):
    label = 0;
  else:
    label = 1;
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return label


def recursive_split(node, max_depth, min_size, depth):
  '''
  Inputs:
  node:  A dictionary contains 3 key value pairs, node = 
         {'index': integer, 'value': float, 'groups': a tuple contains two lists fo samples}
  max_depth: maximum depth of the tree, an integer
  min_size: minimum size of a group, an integer
  depth: tree depth for current node
  Output:
  no need to output anything, the input node should carry its own subtree once this function ternimate
  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  left_g, right_g = node['groups'];
  node.pop('groups', None)
  # check for a no split
  if(len(left_g) == 0):
    node['left'] = to_terminal(right_g);
    node['right'] = to_terminal(right_g);
    return
  elif(len(right_g) == 0):
    node['left'] = to_terminal(left_g);
    node['right'] = to_terminal(left_g);
    return
 
  # check for max depth
  if (depth >= max_depth - 1):   # use >= instead of == in case max_depth = 1
    node['left'] = to_terminal(left_g);
    node['right'] = to_terminal(right_g);
    return
  # process left child
  if len(left_g) <= min_size:
    node['left'] = to_terminal(left_g);
  else:
    node['left'] = get_best_split(left_g);
    recursive_split(node['left'], max_depth, min_size, depth+1)
  # process right child
  if len(right_g) <= min_size:
    node['right'] = to_terminal(right_g);
  else:
    node['right'] = get_best_split(right_g)
    recursive_split(node['right'], max_depth, min_size, depth+1)

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


def build_tree(train, max_depth, min_size):
  '''
  Inputs:
    - train: Training set, a list of examples. Each example is a list, whose last element is the label.
    - max_depth: maximum depth of the tree, an integer (root has depth 1)
    - min_size: minimum size of a group, an integer
  Output:
    - root: The root node, a recursive dictionary that should carry the whole tree
  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  root = get_best_split(train);
  recursive_split(root, max_depth, min_size, 1);
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return root

def predict(root, sample):
  '''
  Inputs:
  root: the root node of the tree. a recursive dictionary that carries the whole tree.
  sample: a list
  Outputs:
  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  while(isinstance(root,int)==False):
    if(sample[root['index']] < root['value']):
      root = root['left'];
    else:
      root = root['right'];
  return root;
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
