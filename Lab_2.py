import numpy as np
import string
from collections import Counter


np.random.seed(1)

# Problem 1

def count_frequency(documents):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    wordBucket = []
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    if type(documents) is list:
        for message in documents:
            msg = message.lower()
            for char in msg:
                if char in punc:
                    msg = msg.replace(char, " ")
            msgSplit = msg.split(' ')
            wordBucket.extend(msgSplit)
    elif type(documents) is np.str_ or type(documents) is str:
        message = [documents.lower()]
        for msg in message:
            for char in msg:
                if char in punc:
                    msg = msg.replace(char, " ")
            msgSplit = msg.split(' ')
            wordBucket.extend(msgSplit)
    frequency = Counter(wordBucket)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return frequency

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
def prior_prob(y_train):
    labelCount = Counter(y_train)
    prior = {}
    for key in labelCount:
        prior[key] = labelCount[key]/np.prod(y_train.shape)

    return prior


def conditional_prob(X_train, y_train):
    condProb = {}
    labels = Counter(y_train)
    msgDict = {}

    for key in labels:
        condProb[key] = {}
        msgDict[key] = []

    for i, key in enumerate(y_train):
        msgDict[key].append(X_train[i])

    for key in labels:
        x = count_frequency(msgDict[key])
        condProb[key] = Counter(condProb[key]) + x

    for label in condProb:
        tempDict = condProb[label]
        #print(tempDict)
        totWords = sum(tempDict.values())
        for word in tempDict:
            condProb[label][word] = (tempDict[word] + 1.0) / (totWords+20000.0)
            #print(condProb[label][word])
    cond_prob = condProb


    return cond_prob

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


def predict_label(X_test, prior_prob, cond_prob):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    maxProb = float('-inf')
    predict = []
    test2D = np.empty(X_test.shape[0] * len(prior_prob))
    test_prob = test2D.reshape(X_test.shape[0], len(prior_prob))
    probArray = []
    maxLabel = None
    labelsArray = []
    for i, msg in enumerate(X_test):
        countFreq = count_frequency(msg)

        #computes all g_y(X)
        for label in prior_prob:
            labelsArray.append(label)
            labelProb = compute_test_prob(countFreq, prior_prob[label], cond_prob[label])     
            probArray.append(labelProb)
        
        #computes m = max_y g_y(X)
        m = max(probArray)
        sumProb = 0

        #computes sum_y(exp(g_y(X)-m))
        for j in probArray:
            sumProb += np.exp(j - m)
            
        temp = []
        #computes exp(g_y(X)-m) / sum_y(exp(g_y(X)-m))
        for k in probArray:
            predProb = np.exp(k - m) / sumProb
            temp.append(predProb)
        test_prob[i] = np.array(temp)
        probArray = []
        predict.append([labelsArray[np.argmax(temp)]])
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return predict, test_prob

def compute_test_prob(word_count, prior_cat, cond_cat):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    logSum = 0.0
    wordSum = 0.0
    totWords = sum(cond_cat.values())
    
    #computes sum(logP(Xi|Y))
    for wd in word_count:  
        if cond_cat[wd]:
            wordSum += np.log(cond_cat[wd]) * cond_cat[wd]
        else:
            wordSum += np.log(1/(totWords + 20000))
        logSum += wordSum
        wordSum = 0.0

    #computes gy(X) = logP(Y) + sum(logP(Xi|Y))
    prob = np.log(prior_cat) + logSum
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return prob

def compute_metrics(y_pred, y_true):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average = 'binary')
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return acc, cm, f1


# Problem 2

def featureNormalization(X):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X_mean = np.mean(X, axis = 0)
    X_std = np.std(X, axis = 0)
    X_normalized = (X-X_mean)/X_std
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return X_normalized, X_mean, X_std


def applyNormalization(X, X_mean, X_std):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X_normalized = (X-X_mean)/X_std

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return X_normalized

def computeMSE(X, y, theta):

  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  m = len(y)
  pred = np.dot(X, theta)
  error = (m/16) * np.sum(np.square(pred-y))
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return error[0]

def computeGradient(X, y, theta):

  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  num_instances = X.shape[0]
  pred = np.dot(X, theta)
  gradient = pred-y
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

  return gradient

def gradientDescent(X, y, theta, alpha, num_iters):

  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  m = len(y)
  mse = np.zeros(num_iters)
  for i in range(num_iters):
    pred = np.dot(X,theta)
    Loss_record = computeMSE(X,y,theta)
    theta = theta - (1/m)*alpha*(X.T.dot((pred-y)))
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return theta, Loss_record

def closeForm(X, y):

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  theta = np.dot(y,np.linalg.inv(X))
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return theta
