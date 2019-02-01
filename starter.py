import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import linalg as LA
from numpy.linalg import inv
import time

img_h = img_w = 28             # MNIST images are 28x28
img_size_flat = img_h * img_w  # 28x28=784, the total number of pixels


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


def preproc_new(trainData, validData=None, testData=None):
    shape_train = np.shape(trainData)
    shape_valid = np.shape(validData)
    shape_test = np.shape(testData)

    num_samples_train = shape_train[0]
    num_samples_valid = shape_valid[0]
    num_samples_test = shape_test[0]

    pic_flattened = shape_train[1]*shape_train[2]
    trainData = np.reshape(trainData, (num_samples_train, pic_flattened))
    validData = np.reshape(validData, (num_samples_valid, pic_flattened))
    testData = np.reshape(testData, (num_samples_test, pic_flattened))
    return trainData, validData, testData


def preproc(trainData, validData=None, testData=None):

    symm_train = np.zeros((len(trainData), 2))
    for i in range(0, len(trainData)):
        flipped_train = np.flipud(trainData[i])
        square_diff_train = np.square(trainData[i] - flipped_train).sum()
        symm_train[i]=np.array([1, square_diff_train])

    returned_vals = (symm_train,)

    if validData is not None and testData is not None:
        symm_valid = np.zeros((len(validData), 2))
        symm_test = np.zeros((len(testData), 2))

        for i in range(0, len(validData)):
            flipped_valid = np.flipud(validData[i])
            square_diff_valid = np.square(validData[i] - flipped_valid).sum()
            symm_valid[i] = np.array([1, square_diff_valid])

        for j in range(0, len(testData)):
            flipped_test = np.flipud(testData[j])
            square_diff_test = np.square(testData[j] - flipped_test).sum()
            symm_test[j] = np.array([1, square_diff_test])

        returned_vals = (symm_train, symm_valid, symm_test)

    return returned_vals


def MSE(W, b, x, y, reg):
    """
    :param W: Weight Matrix for which error will be calculated
    :param b: Bias scalar added to the weight matrix
    :param x: Dataset, matrix multiplied by W to produce predicted results, y_pred
    :param y: Ground truth classification values
    :param reg: Regularization value to be applied to the weight matrix W
    :return: Error value returned using Mean Square Error method to calculate

    Calculates the Mean Square Error of the weight matrix W in predicting the class of dataset x, based on ground
    truths y. Determines as follows:

    MSE = (1/2N)*sum(norm((W_transpose * x(n) + b - y(n)))^2 + 0.5*reg(norm(W))^2
    """
    size = y.size  # Your implementation here
    y_pred = x.dot(W.transpose()).flatten()
    error = np.power((y_pred + b - y.flatten()), 2)
    # print(error.sum())
    # print(size)
    estimate = error.sum() / (2*size)
    # print(estimate)
    decay = 0.5 * LA.norm(W) * LA.norm(W) * reg
    return estimate + decay
   

def gradMSE(W, b, x, y, reg):
    # Your implementation here
    size = y.size   
    y_pred = x.dot(W.transpose()).flatten()
    weight_error = (y_pred + b - y.flatten()) + reg*W[:-1].sum()
    bias_error = (y_pred + b -y.flatten())

     
    return weight_error.dot(x) / size,bias_error.sum()/size


def grad_loop(W, b, trainingData, trainingLabels, reg, alpha):
    errorW,errorB = gradMSE(W, b, trainingData, trainingLabels, reg)
    
    norm = np.linalg.norm(errorW)
    #errorG /= norm

    # print W
    W += -(errorW*alpha)
    b += -(errorB*alpha)
    return norm,W,b

    # Your implementation here
    # E=MSE(W,b,trainingData,trainingLabels,reg) if close to zero don';t run gradient descent


def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    mse_list = []
    iterations_list = []
    V_list=[]
    T_list=[]
    while iterations < 500:
        g_val, W, b= grad_loop(W, b, trainingData, trainingLabels, reg, alpha)
        mse_training = MSE(W, b, trainingData, trainingLabels, reg)
        iterations += 1
        Error_V=MSE(W, b, validData, validTarget, reg=0.0)
        Error_T=MSE(W, b, testData, testTarget, reg=0.0)
        V_list.append(Error_V)
        T_list.append(Error_T)
        iterations_list.append(iterations)
        mse_list.append(mse_training)

        # print("W: ", W)
        if g_val < 0.001:
            print("done")
            break
    # print("W: ", W)
    fig, ax = plt.subplots()
    ax.plot(iterations_list, mse_list)
    ax.plot(iterations_list, V_list)
    ax.plot(iterations_list, T_list)
    ax.legend((mse_list, V_list, T_list), ('Training Error', 'Validation Error', 'Testing Error'))
    plt.show()
    return W, b


def sigmoid(W, b, x):
    # print('sigmoid: ')
    # print(np.shape(W))
    # print(np.shape(x))
    # print(np.shape(b))
    z = x.dot(W.transpose()).flatten()+b
    return 1/(1+np.exp(z))


def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    size = y.size
    """
    print('size of y: ', np.shape(y.flatten()))
    print('crossEntropy: ')
    print(np.shape(W))
    print(np.shape(x))
    print(np.shape(b))
    print('hello: ')
    z2 = np.log(sigmoid(W, b, x))
    print(np.shape(z2))
    print('end: ', z2)
    # z1 = -(y.dot(np.log(sigmoid(W, b, x))))
    # print(z1)
    """
    y_flat = y.flatten()
    z = -(y_flat.dot(np.log(sigmoid(W, b, x)))) - np.dot((1 - y_flat), np.log(1 - sigmoid(W, b, x)))
    z = z.sum()/size + 0.5*reg*LA.norm(W)*LA.norm(W)
    print(z, np.shape(z))
    return z


def gradCE(W, b, x, y, reg):
    # Your implementation here
    return


def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    # Your implementation here
    return


def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    return


trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData, validData, testData = preproc_new(trainData, validData, testData)
print('shape', np.shape(trainData))
# all_data = preproc(trainData, validData, testData)
# trainData = all_data[0]
# validData = all_data[1]
# testData = all_data[2]
np.random.seed(3)
W = np.random.rand(784)
crossEntropyLoss(W, 0, trainData, trainTarget, reg=1)

'''
atStart = MSE(W, 0, trainData, trainTarget, reg=0.0)

# W, b = grad_descent(W, 0, trainData, trainTarget, alpha=0.0003, iterations=0.0, reg=0.0, EPS=1e-7)

atEnd = MSE(W, b, trainData, trainTarget, reg=0.0)
print(atStart, atEnd)

print('validation error: ', MSE(W, b, validData, validTarget, reg=0.0))
print('test error: ', MSE(W, b, testData, testTarget, reg=0.0))
# print("W", np.matmul(np.matmul(inv(np.matmul(trainData.transpose(), trainData)), trainData.transpose()), trainTarget))
'''

