import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math
# from ..utils.extmath import (log_logistic, safe_sparse_dot, softmax, squared_norm)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
'''
Train Data -(10000,28, 28)
Valid Data -(6000,28, 28)
Test Data -(2724,28, 28)

Train Target -(10000,)
valid Target -(10000,)
Test Target -(10000,)
'''

"""
For sanity: shapes into functions:

general variables:
n = number of examples
k = number of classes

our images are 784 pixels long after flattening

shapes after one-hot encoding
TrainTarget - (10000, 10)
ValidTarget - (6000, 10)
TestTarget -  (2724, 10)

gradCE(targets, predictions)
    target = k x 1 or 1 x k (fla
    predictions = k x n
    
    <target, predictions> = 1 x n
    sum of <target, predictions> = scalar, 1 x 1
"""


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


def flattenData(data):
    for item in data:
        item = item.flatten()
    return data


def relu(x):
    '''Calculates the RELU max value of X'''
    return np.maximum(x,0) 


def softmax(x):
    # Calculates Softmax of x
    exp = np.exp(x)
    exp_sum = np.reshape(np.sum(np.exp(x),axis=1), (x.shape[0], 1))
    soft_max=exp/exp_sum
    return soft_max


def computeLayer(X, W, b):
    # TODO
    W_transpose = np.transpose(W)
    return np.dot(X,W) + b      # Open to changes


def averageCE(target, prediction):
    print('target: ', target[1])
    print('prediction: ', prediction[1])
    CE_vector = target * np.log(prediction)
    print('CE_VEC: ', CE_vector.shape[0])
    CE_error = -1*(CE_vector.sum((0, 1)) / CE_vector.shape[0])
    print('Size: ', CE_vector.shape[0])
    print('Vector shape: ', np.shape(CE_vector))
    print('Error shape: ', np.shape(CE_error))
    return CE_error
    # TODO


def accuracy(target, prediction):
    target_index = np.argmax(target, axis=1)
    prediction_index = np.argmax(prediction, axis=1)
    
    accuracy = np.equal(target_index, prediction_index).astype(np.float32).sum() / len(target_index)
    print(accuracy)
    return accuracy


def gradCE(target, prediction):
    # softpreds = softmax(prediction)
    # print('softpreds: ', np.max(softpreds))
    # print('Prediction: ', np.amax(prediction, (0, 1)))
    Grad = (prediction - target)
    #print Grad
    #totalGrad = (softpreds - target) 
    return np.transpose(Grad)#, totalGrad
    # TODO


def forward_prop(W_in_h,W_out_h,b_in_h,b_out_h,trainDB):
    bias_h = np.ones((trainDB.shape[0], 1), dtype=np.float32)
    Sval_h = computeLayer(trainDB, W_in_h, np.dot(bias_h,b_in_h))
    Xval_h = relu(Sval_h)
    # print ("Hi",np.amin(Xval_h, (0, 1)))    
    Sval_o = computeLayer(Xval_h,W_out_h,np.dot(bias_h,b_out_h))  
    Xval_o= softmax(Sval_o)
    # print ("Hello",np.amin(Xval_o, (0, 1))) 
    return Sval_h, Xval_h, Sval_o, Xval_o      


def Back_prop(Sval_h, Xval_h,W_out_h, Sval_o, Xval_o,trainTarget):
    #print("Svalh-",(np.shape(Sval_h)))
    #print("Xvalh-",(np.shape(Xval_h)))
    #print("Svalo-",(np.shape(Sval_o)))
    #print("Xvalo-",(np.shape(Xval_o)))
    delta_o = gradCE(trainTarget,Xval_o)
    
    unit_step = np.greater_equal(Sval_h, 0).astype(np.float32)
    #print('Unit_step: ', unit_step)
    delta_h = np.transpose(unit_step) * np.dot(W_out_h,delta_o)
    # print("delta_o",np.shape(delta_o))
    # print("delta_h",np.shape(delta_h))
    return delta_o,delta_h


def learning(W_in_h,W_out_h,b_in_h,b_out_h,trainDB, alpha, gamma):
    num_samples = len(trainDB)
    bias = np.ones((len(trainDB), 1), dtype=np.float32)
    iterations = 0

    tr_err_list = []
    va_err_list = []
    te_err_list = []    
    
    tr_acc_list = []
    va_acc_list = []
    te_acc_list = []

    v_W_out_old = np.full(np.shape(W_out_h), 1e-5, dtype=np.float32)
    v_W_in_old = np.full(np.shape(W_in_h), 1e-5, dtype=np.float32)
    v_b_out_old = np.full(np.shape(b_out_h), 1e-5, dtype=np.float32)
    v_b_in_old = np.full(np.shape(b_in_h), 1e-5, dtype=np.float32)

    G_W_in_h=np.zeros((784,1000))
    G_W_out_h=np.zeros((1000,10))
    G_b_in_h=np.zeros((1,1000))
    G_b_out_h=np.zeros((1,10))

    while(iterations<=200):
        print(iterations)
        # print(W_out_h)
        Sval_h, Xval_h, Sval_o, Xval_o = forward_prop(W_in_h,W_out_h,b_in_h,b_out_h,trainDB)
        delta_o,delta_h = Back_prop(Sval_h, Xval_h,W_out_h, Sval_o, Xval_o,trainTarget)
        
        G_W_out_h = np.dot(np.transpose(Xval_h), np.transpose(delta_o)) / num_samples
        G_W_in_h = np.dot(np.transpose(trainDB), np.transpose(delta_h)) / num_samples
        # print(G_W_out_h)
        # print("G_W_in_h")
        v_W_out_new = gamma*v_W_out_old + alpha*G_W_out_h
        v_W_in_new = gamma*v_W_in_old + alpha*G_W_in_h
        # print("v_W_out_new",np.shape(v_W_out_new))
        # print("v_W_in_new",np.shape(v_W_in_new))
        W_out_h -= v_W_out_new
        W_in_h -= v_W_in_new

        G_b_out_h = np.dot(np.transpose(bias), np.transpose(delta_o)) / num_samples
        G_b_in_h = np.dot(np.transpose(bias), np.transpose(delta_h)) / num_samples
        
        v_b_out_new = gamma*v_b_out_old + alpha*G_b_out_h
        v_b_in_new = gamma*v_b_in_old + alpha*G_b_in_h
        
        b_out_h -= v_b_out_new
        b_in_h -= v_b_in_new
        
        # print('num samples: ', num_samples)
        print('Train Error: ', averageCE(trainTarget, Xval_o))
        print('Train Accuracy: ', accuracy(trainTarget, Xval_o))
        tr_acc_list.append(accuracy(trainTarget, Xval_o))
        tr_err_list.append(averageCE(trainTarget, Xval_o))

        Sval_h_va, Xval_h_va, Sval_o_va, Xval_o_va = forward_prop(W_in_h,W_out_h,b_in_h,b_out_h,validData)
        print('Valid Error: ', averageCE(validTarget, Xval_o_va))
        print('Valid Accuracy: ', accuracy(validTarget, Xval_o_va))
        va_acc_list.append(accuracy(validTarget, Xval_o_va))
        va_err_list.append(averageCE(validTarget, Xval_o_va))

        Sval_h_te, Xval_h_te, Sval_o_te, Xval_o_te = forward_prop(W_in_h,W_out_h,b_in_h,b_out_h,testData)
        print('Test Error: ', averageCE(testTarget, Xval_o_te))
        print('Test Accuracy: ', accuracy(testTarget, Xval_o_te))
        te_acc_list.append(accuracy(testTarget, Xval_o_te))
        te_err_list.append(averageCE(testTarget, Xval_o_te))     

        iterations += 1
        
        v_W_out_old = v_W_out_new
        v_W_in_old = v_W_in_new
        
        v_b_out_old = v_b_out_new
        v_b_in_old = v_b_in_new
    
    fig, ax = plt.subplots()
    iterations_list = np.arange(0, 201)
    # ax.plot(iterations_list, tr_err_list, label='Training Error')
    # ax.plot(iterations_list, va_err_list, label='Validation Error')
    # ax.plot(iterations_list, te_err_list, label='Testing Error')
    
    ax.plot(iterations_list, tr_acc_list, label='Training Accuracy')
    ax.plot(iterations_list, va_acc_list, label='Validation Accuracy')
    ax.plot(iterations_list, te_acc_list, label='Testing Accuracy')
    ax.legend()
    plt.show()
    return W_out_h, W_in_h, b_out_h, b_in_h
    #TODO
   

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

trainData = trainData.reshape(10000, 784)
validData = validData.reshape(6000, 784)
testData = testData.reshape(2724, 784)

bias_h = np.ones((10000, 1), dtype=np.float32)
trainDB = trainData  #np.concatenate((bias_h, trainData), axis=1)

W_hidden=np.random.normal(0,math.sqrt(float(2./1784)),(784, 1000))
print(np.max(W_hidden))

trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)

W_output=np.random.normal(0,math.sqrt(float(2./1010)),(1000, 10))
print(np.max(W_output))

b_in_h=np.random.normal(0,math.sqrt(float(2./1784)),(1, 1000))
b_out_h=np.random.normal(0,math.sqrt(float(2./1010)),(1, 10))

W_out_h, W_in_h, b_out_h, b_in_h = learning(W_hidden, W_output,b_in_h,b_out_h,trainDB, 0.001, 0.95)


"""
print(np.shape(trainData))
print(np.shape(validData))
print(np.shape(testData))
print(np.shape(trainTarget))
print(np.shape(validTarget))
print(np.shape(testTarget))
"""

