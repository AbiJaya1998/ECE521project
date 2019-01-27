import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy.linalg import inv

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


def preproc(trainData, validData=None, testData=None):

	symm_train = np.zeros((len(trainData), 2))
	for i in range(0, len(trainData)):
		flipped_train = np.flipud(trainData[i])
		square_diff_train = np.square(trainData[i] - flipped_train).sum()
		symm_train[i]=np.array([1, square_diff_train])
	
	returned_vals = (symm_train)

	if validData is not None and testData is not None:
		symm_valid = np.zeros((len(validData), 2))
		symm_test = np.zeros((len(testData), 2))

		for i in range(0, len(validData)):
			flipped_valid = np.flipud(validData[i])
			square_diff_valid = np.square(validData[i] - flipped_valid).sum()
			symm_valid[i] = np.array([1, square_diff_valid])

		for j in range(0, len(testData)):
			flipped_test = np.flipud(testData[i])
			square_diff_test = np.square(testData[i] - flipped_test).sum()
			symm_test[i] = np.array([1, square_diff_test])
		
		returned_vals = (symm_train, symm_valid, symm_test)
	
	return returned_vals


def MSE(W, b, x, y, reg):
    size=y.size# Your implementation here
    y_pred=x.dot(W.transpose()).flatten()
    error=np.power((y_pred+b-y.flatten()),2)
    print error.sum()
    print size
    estimate=error.sum()/(2*size)
    print estimate
    decay=LA.norm(W)*LA.norm(W)*0.5*reg
    return (estimate + decay)
   

def gradMSE(W, b, x, y, reg):
    # Your implementation here
     size=y.size
     y_pred=x.dot(W.transpose()).flatten()
     error = (y_pred + b - y.flatten()) + reg*W[:-1].sum()
     return (error.dot(x)/size)


def grad_loop(W, b, trainingData,trainingLabels, reg,alpha):
    error=gradMSE(W,b,trainingData,trainingLabels,reg)
    
    norm=np.linalg.norm(error) 
    error/=norm

    #print W
    W+=-(error)*(alpha)
    return norm,W


    # Your implementation here
    #E=MSE(W,b,trainingData,trainingLabels,reg) if close to zero don';t run gradient descent
def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
   
    while iterations<100000:
        g_val,W=grad_loop(W, b, trainingData,trainingLabels, reg,alpha)
        iterations+=1
        print("W: ", W)
    	if g_val<0.001:
             print"done"
             break
    print("W: ", W)
'''def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here

def gradCE(W, b, x, y, reg):
    # Your implementation here

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    # Your implementation here

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
'''

trainData, validData, testData, trainTarget, validTarget, testTarget=loadData()
all_data = preproc(trainData, validData, testData)
trainData = all_data[0]
validData = all_data[1]
testData = all_data[2]

W = np.linspace(0.0, 1.0, num=2)
atStart = MSE(W, 0, trainData, trainTarget, reg=0.0)

grad_descent(W, 0 ,trainData,trainTarget,alpha=0.0003,iterations=0.0,reg=0.0,EPS=1e-7)

atEnd = MSE(W, 0, trainData, trainTarget, reg=0.0)
print(atStart,atEnd)
print "W",np.matmul(np.matmul(inv(np.matmul(trainData.transpose(),trainData)),trainData.transpose()),trainTarget)


