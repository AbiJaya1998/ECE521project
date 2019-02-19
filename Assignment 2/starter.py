import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def relu(x):
    '''Calculates the RELU max value of X'''

    return x *(x > 0) # or can be done using np.maximum(x,0) 

def softmax(x):
    '''Calculates Softmax of x'''

    exp = np.exp(x)
    exp_sum = np.sum(np.exp(x),axis=0)
    soft_max=exp/exp_sum
    return soft_max



'''def computeLayer(X, W, b):
    # TODO


def CE(target, prediction):
    # TODO


def gradCE(target, prediction):
    # TODO'''

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
print(np.shape(trainData))
print(np.shape(validData))
print(np.shape(testData))
print(np.shape(trainTarget))
print(np.shape(validTarget))
print(np.shape(testTarget))


