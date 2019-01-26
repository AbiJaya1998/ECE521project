import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
print("Data type: ", type(Data))
print("Target type: ", type(Target))


def MSE(W, b, x, y, reg):  # Your implementation here
    size=y.size# Your implementation here
    y_pred=x.dot(W.transpose()).flatten()
    error=power((y_pred+b-y.flatten()),2)
    estimate=(1/size)*error.sum()
    decay=LA.norm(W)*LA.norm(W)*0.5*reg
    return


def gradMSE(W, b, x, y, reg):  # Your implementation here

    return


def crossEntropyLoss(W, b, x, y, reg):  # Your implementation here

    return


def gradCE(W, b, x, y, reg): # Your implementation here

    return


def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS): # Your implementation here

    return


def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None): # Your implementation here
    return

