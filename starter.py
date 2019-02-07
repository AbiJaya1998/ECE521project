import tensorflow as tf
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
    """
    :param trainData: Training data used for training
    :param validData: Validation data
    :param testData: Testing data
    :return: Training, validation, and testing data to be used in training

    Preprocesses the data to be used. Data is assumed to be of shape [n, z, z],
    where n is the number of data samples and z is the side length of the data
    The the image is flattened such that the shape becomes [n, z*z]
    """
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
    """
    :param trainData: Training data
    :param validData: Validation data
    :param testData: Testing data
    :return tuple: trainData, validData, testData 
    
    Old preprocessing function that uses symmetry as a feature to reduce the
    size of both the data and the weight matrix to be used
    """
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
    :param x: Dataset, matrix multiplied by W to produce predicted results, 
        y_pred
    :param y: Ground truth classification values
    :param reg: Regularization value to be applied to the weight matrix W
    :return: Error value returned using Mean Square Error method to calculate

    Calculates the Mean Square Error of the weight matrix W in predicting the 
    class of dataset x, based on ground
    truths y. Determines as follows:

    MSE=(1/2N)*sum(norm((W_transpose * x(n) + b - y(n)))^2 + 0.5*reg(norm(W))^2
    """
    # Your implementation here
    size = y.size  
    y_pred = x.dot(W.transpose()).flatten()
    error = np.power((y_pred + b - y.flatten()), 2)
    estimate = error.sum() / (2*size)
    decay = 0.5 * LA.norm(W) * LA.norm(W) * reg
    return estimate + decay
   

def gradMSE(W, b, x, y, reg):
    """
    :param W: Weight matrix to calculate gradient of
    :param b: Bias scalar added to the weight matrix
    :param x: Dataset, matrix multiplied by W to produce predicted results, y_pred
    :param y: Ground truth classification values
    :param reg: Regularization value to be applied to the weight matrix W
    :return: Gradient of both the weights and the bias

    Calculates the gradient of both the weights W and bias b for the MSE error
    """
    # Your implementation here
    size = y.size   
    y_pred = x.dot(W.transpose()).flatten()
    weight_error = (y_pred + b - y.flatten()).dot(x)/size + reg*W
    bias_error = (y_pred + b - y.flatten())
    # print('MSE: ', np.shape(bias_error))
    return weight_error, bias_error.sum()/size


def grad_loop(W, b, trainingData, trainingLabels, reg, alpha):
    """
    :param W: Weight Matrix for which error will be calculated
    :param b: Bias scalar added to the weight matrix
    :param x: Dataset, matrix multiplied by W to produce predicted results, 
        y_pred
    :param y: Ground truth classification values
    :param reg: Regularization value to be applied to the weight matrix W
    :return: Error value returned using Mean Square Error method to calculate
    
    Helper function used in grad_descent. Calculates the gradient in W, b using
    gradMSE, then sends that to grad_descent for further work
    """
    errorW,errorB = gradMSE(W, b, trainingData, trainingLabels, reg)
    
    norm = np.linalg.norm(errorW)
    W += -(errorW*alpha)
    b += -(errorB*alpha)
    return norm,W,b


def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS=1e-7):
    """
    Computes the gradient descent for MSE loss. Note that iterations
    parameter should be set to 0
    """
    mse_list = []
    tr_list_acc = []
    iterations_list = list(range(1, iterations+1))
    V_list=[]
    T_list=[]
    V_list_acc = []
    T_list_acc = []
    W_old = W
    while iterations > 0:
        g_val, W, b = grad_loop(W, b, trainingData, trainingLabels, reg, alpha)
        mse_training = MSE(W, b, trainingData, trainingLabels, reg)
        iterations -= 1
        Error_V=MSE(W, b, validData, validTarget, reg=reg)
        Error_T=MSE(W, b, testData, testTarget, reg=reg)
        
        tr_acc = accuracy(W, b, trainData, trainTarget)
        V_acc = accuracy(W, b, validData, validTarget)
        T_acc = accuracy(W, b, testData, testTarget)
        
        tr_list_acc.append(tr_acc)
        V_list_acc.append(V_acc)
        T_list_acc.append(T_acc)
        
        V_list.append(Error_V)
        T_list.append(Error_T)
        mse_list.append(mse_training)
        if g_val < 0.0001:
            print("done")
            break
        W_old = W
    return W, b, mse_list, V_list, T_list, iterations_list, tr_list_acc, V_list_acc, T_list_acc


def sigmoid(W, b, x):
    """
    :param W: Weight matrix to be trained
    :param b: Bias to be added
    :param x: Dataset to be trained

    Computes the sigmoid function, defined as follows:
        1 / (1 + exp(-z)), where z = W_tx + b
    """
    z = x.dot(W.transpose()).flatten()+b
    
    return 1/(1+np.exp(-z))


def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    size = y.size
    y_flat = y.flatten()
    z = -(y_flat.dot(np.log(sigmoid(W, b, x)))) - np.dot((1 - y_flat), np.log(1 - sigmoid(W, b, x)))
    z = z.sum() / size + 0.5 * reg * LA.norm(W) * LA.norm(W)
    return z


def gradCE(W, b, x, y, reg):
    # Your implementation here
    size = y.size

    grad_W = -((y.flatten() - sigmoid(W, b, x)).dot(x))/size + reg*W
    grad_b = (-(y.flatten() - sigmoid(W, b, x)))
    grad_b = np.sum(grad_b)/size
    return grad_W, grad_b


def grad_loop_CE(W, b, trainingData, trainingLabels, reg, alpha):
    """
    :param W: Weight matrix to train
    :param b: bias to be added to the weight matrix
    :param trainingData: training data to be trained on
    :param trainingLabels: training labels to be trained on
    :param reg: regularization parameter
    :param alpha: learning rate
    """
    errorW, errorB = gradCE(W, b, trainingData, trainingLabels, reg)
    norm = np.linalg.norm(errorW)

    W += -(errorW*alpha)
    b += -(errorB*alpha)
    return norm, W, b


def grad_descent_CE(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS=1e-7):
    """
    Computes the gradient descent for cross-entropy loss. Note that iterations
    parameter should be set to 0
    """
    mse_list = []
    iterations_list = iterations_list = list(range(1, iterations+1))
    V_list=[]
    T_list=[]

    train_acc_list = []
    v_acc_list = []
    test_acc_list = []

    while iterations > 0:
        g_val, W, b = grad_loop_CE(W, b, trainingData, trainingLabels, reg, alpha)
        mse_training = crossEntropyLoss(W, b, trainingData, trainingLabels, reg)
        iterations -= 1
        Error_V = crossEntropyLoss(W, b, validData, validTarget, reg=reg)
        Error_T = crossEntropyLoss(W, b, testData, testTarget, reg=reg)
        V_list.append(Error_V)
        T_list.append(Error_T)
        
        mse_list.append(mse_training)
        train_acc = accuracy(W, b, trainingData, trainingLabels, sig=True)
        v_acc = accuracy(W, b, trainingData, trainingLabels, sig=True)
        test_acc = accuracy(W, b, trainingData, trainingLabels, sig=True)

        train_acc_list.append(train_acc)
        v_acc_list.append(v_acc)
        test_acc_list.append(test_acc)

        if g_val < 0.001:
            print("done")
            break
    return W, b, mse_list, V_list, T_list, train_acc_list, v_acc_list, test_acc_list, iterations_list


def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None, batch_size=1):
    # Your implementation here
    tf.set_random_seed(421)
    reg=0.0
    n_epochs = 700
    batch_size = batch_size
    n_batches = int(trainData.shape[0])//batch_size

    # Creation of all required data tensors
    weights = tf.Variable(tf.truncated_normal([784, 1], stddev=0.5, name='Weights'))
    bias = tf.Variable(0, dtype=tf.float32, name='Biases')
    x_data = tf.placeholder(dtype=tf.float32, name='x_data')
    y_labels = tf.placeholder(dtype=tf.float32, name='y_data')
    
    combinedDatLab = np.concatenate((trainData.transpose(), trainTarget.transpose()), axis=0).transpose()
    print(np.shape(combinedDatLab))
    half_reg = tf.constant(0.5*reg, dtype=tf.float32, shape=[1], name='regularization')
    half = tf.constant(0.5, dtype=tf.float32, shape=[1], name='half')
    
    error = None
    acc = None
    y_pred = None
    best_optimizer = None

    if lossType == 'MSE':
        y_pred_MSE = tf.add(tf.matmul(x_data, weights), bias)
        mse1 = tf.reduce_mean(tf.square(y_labels - y_pred_MSE))
        mse2 = tf.multiply(half_reg, tf.norm(weights))
        error = tf.add(mse1, mse2)
        
        diff = tf.abs(tf.subtract(y_labels, y_pred_MSE))
        pred_labels = tf.cast(tf.less(diff, half), tf.float32)
        acc = tf.reduce_mean(pred_labels)
        optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)

    elif lossType == 'CE':
        y_pred_CE = tf.add(tf.matmul(x_data, weights),bias)
        y_pred_CE_sig = tf.sigmoid(y_pred_CE)
        ce1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred_CE,labels=y_labels))
        ce2 = tf.multiply(half_reg, tf.norm(weights))
        error = tf.add(ce1, ce2)
        optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon).minimize(error)

    init = tf.global_variables_initializer()
    
    train_err_list = []
    valid_err_list = []
    test_err_list = []
    train_acc_list = []
    valid_acc_list = []
    test_acc_list = []
    start = time.time()
    with tf.Session() as sess:
        sess.run(init)
        # before = crossEntropyLoss(weights.eval().flatten(), bias.eval(), trainData, trainTarget, reg=0)
        for epoch in range(n_epochs):
            # Every epoch, shuffle
            np.random.shuffle(combinedDatLab)  # shuffled data
            new_x = combinedDatLab[:, :784]
            new_y = combinedDatLab[:, 784:785]
            print(epoch)
            for i in range(n_batches):
                xvals = new_x[i*batch_size:(i+1)*batch_size, :]            
                yvals = new_y[i*batch_size:(i+1)*batch_size]  
                # Now input into the session
                if lossType == "MSE":
                    sess.run([y_pred_MSE, optimizer1], feed_dict={x_data: xvals, y_labels: yvals})
                    err_v = sess.run(error, feed_dict={x_data: validData, y_labels: validTarget})
                    err_te = sess.run(error, feed_dict={x_data: testData, y_labels: testTarget})
                    err_tr = sess.run(error, feed_dict={x_data: trainData, y_labels: trainTarget})
                
                    acc_v = sess.run(acc, feed_dict={x_data: validData, y_labels: validTarget})
                    acc_te = sess.run(acc, feed_dict={x_data: testData, y_labels: testTarget})
                    acc_tr = sess.run(acc, feed_dict={x_data: trainData, y_labels: trainTarget})
                    train_err_list.append(err_tr)
                    valid_err_list.append(err_v)
                    test_err_list.append(err_te)
                    train_acc_list.append(acc_tr)
                    valid_acc_list.append(acc_v)
                    test_acc_list.append(acc_te)
                    
                elif lossType=="CE":
                    y_sig, opt = sess.run([y_pred_CE_sig, optimizer2], feed_dict={x_data: xvals, y_labels: yvals})
                    err_tr = sess.run(error, feed_dict={x_data: trainData, y_labels: trainTarget})
                    err_v = sess.run(error, feed_dict={x_data: validData, y_labels: validTarget})
                    err_te = sess.run(error, feed_dict={x_data: testData, y_labels: testTarget})
                    acc_tr = accuracy(weights.eval().flatten(),bias.eval(),trainData,trainTarget, sig=True)
                    acc_v = accuracy(weights.eval().flatten(),bias.eval(),validData,validTarget, sig=True)
                    acc_te = accuracy(weights.eval().flatten(),bias.eval(),testData,testTarget, sig=True)
                     
                    train_err_list.append(err_tr)
                    valid_err_list.append(err_v)
                    test_err_list.append(err_te)
                    train_acc_list.append(acc_tr)
                    valid_acc_list.append(acc_v)
                    test_acc_list.append(acc_te)
                    
                    print(accuracy(weights.eval().flatten(),bias.eval(),trainData,trainTarget, sig=True))
        end = time.time()
        print('Time elapsed: ', end - start)
        if lossType=="MSE":
            y_MSE, tot_v, acc_v = sess.run([y_pred_MSE, pred_labels, acc], feed_dict={x_data: validData, y_labels: validTarget})
            err_te, acc_te = sess.run([pred_labels, acc], feed_dict={x_data: testData, y_labels: testTarget})
            err_v, acc_v = sess.run([pred_labels, acc], feed_dict={x_data: validData, y_labels: validTarget})
            err_tr, acc_tr = sess.run([pred_labels, acc], feed_dict={x_data: trainData, y_labels: trainTarget})
            print("Training accuracy for " + str(batch_size) + " batch size: ", acc_tr)
            print("Validation accuracy for " + str(batch_size) + " batch size: ", acc_v)
            print("Testing accuracy for " + str(batch_size) + " batch size: ", acc_te)
        if lossType=="CE":
            acc_tr = accuracy(weights.eval().flatten(), bias.eval(), trainData, trainTarget, sig=True)
            acc_v = accuracy(weights.eval().flatten(), bias.eval(), validData, validTarget, sig=True)
            acc_te = accuracy(weights.eval().flatten(), bias.eval(), testData, testTarget, sig=True)
            print("Training accuracy for " + str(batch_size) + " batch size: ", acc_tr)
            print("Validation accuracy for " + str(batch_size) + " batch size: ", acc_v)
            print("Testing accuracy for " + str(batch_size) + " batch size: ", acc_te)
            
        fig, ax = plt.subplots()
        plt.subplot(121)
        n_iterations = n_batches*n_epochs
        iterations_list = np.arange(0, n_iterations)
        print('shape: ', np.shape(train_err_list))
        plt.plot(iterations_list, train_err_list, label='training error')
        plt.plot(iterations_list, valid_err_list, label='validation error')
        plt.plot(iterations_list, test_err_list, label='testing error')
        plt.xlabel('Total Number of Iterations')
        plt.ylabel('MSE Error Value')
        plt.legend(loc='upper right')
        
        plt.subplot(122)
        plt.plot(iterations_list, train_acc_list, label='training accuracy')
        plt.plot(iterations_list, valid_acc_list, label='validation accuracy')
        plt.plot(iterations_list, test_acc_list, label='testing accuracy')
        plt.xlabel('Total Number of Iterations')
        plt.ylabel('Accuracy Value')
        plt.legend(loc='upper right')
        plt.show()
        
        best_weights = weights.eval()
        best_bias = bias.eval()
        if lossType == "MSE":
            best_optimizer = optimizer1
            predicted_labels = sess.run(y_pred_MSE, feed_dict={x_data: xvals, y_labels: yvals})
        elif lossType == 'CE':
            best_optimizer = optimizer2
            predicted_labels = sess.run(tf.sigmoid(y_pred_CE), feed_dict={x_data: xvals, y_labels: yvals})
    
        final_error = sess.run(error, feed_dict={x_data: testData, y_labels: testTarget})
        real_labels = testTarget
        
    return best_weights, best_bias, predicted_labels, real_labels, final_error, best_optimizer, reg


def accuracy(W, b, x, y, sig=False):
    y_pred = x.dot(W.transpose()).flatten() + b
    if sig == True:
        y_pred = sigmoid(W, b, x)

    diff = np.abs(y.flatten() - y_pred)
    diff = diff.flatten()
    correct = 0
    for item in diff:
        if item < 0.5:
            correct += 1
    return correct / len(diff)


trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData, validData, testData = preproc_new(trainData, validData, testData)
np.random.seed(3)

W = np.zeros((784,))
b = 0
"""
W_2 = W
W_1_log, b_1_log, ce_list, v_list, t_list, train_acc_list, v_acc_list, test_acc_list, iterations_list = grad_descent_CE(W, 0, trainData, trainTarget, alpha=0.005, iterations=5000, reg=0.1, EPS=1e-7)
fig = plt.figure()
plt.subplot(121)
plt.plot(iterations_list, ce_list, label='Training Error')
plt.plot(iterations_list, v_list, label='Validation Error')
plt.plot(iterations_list, t_list, label='Testing Error')
plt.xlabel('Number of Iterations')
plt.ylabel('Cross-Entropy Error Value')
plt.legend(loc='upper right')

plt.subplot(122)
plt.plot(iterations_list, train_acc_list, label='Training Accuracy')
plt.plot(iterations_list, v_acc_list, label='Validation Accuracy')
plt.plot(iterations_list, test_acc_list, label='Testing Accuracy')
plt.xlabel('Number of Iterations')
plt.ylabel('Accuracy Value')
plt.legend(loc='upper right')

plt.suptitle('Training Accuracy when Regularization = 0.1, Alpha=0.005')
plt.show()
"""
batch_size=500
buildGraph(beta1=0.95, lossType='MSE', learning_rate=0.001, batch_size=batch_size)


