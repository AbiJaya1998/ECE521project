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
    weight_error = (y_pred + b - y.flatten()).dot(x)/size + reg*W
    bias_error = (y_pred + b - y.flatten())
    # print('MSE: ', np.shape(bias_error))
    return weight_error, bias_error.sum()/size


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


def grad_loop_CE(W, b, trainingData, trainingLabels, reg, alpha):
    errorW, errorB = gradCE(W, b, trainingData, trainingLabels, reg)
    norm = np.linalg.norm(errorW)
    #errorG /= norm
    # print(norm)
    # print W
    W += -(errorW*alpha)
    b += -(errorB*alpha)
    # b += -(errorB*alpha)
    return norm, W, b


def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS=1e-7):
    mse_list = []
    iterations_list = []
    V_list=[]
    T_list=[]
    W_old = W
    while iterations < 5000:
        g_val, W, b = grad_loop(W, b, trainingData, trainingLabels, reg, alpha)
        mse_training = MSE(W, b, trainingData, trainingLabels, reg)
        iterations += 1
        Error_V=MSE(W, b, validData, validTarget, reg=reg)
        Error_T=MSE(W, b, testData, testTarget, reg=reg)
        V_list.append(Error_V)
        T_list.append(Error_T)
        iterations_list.append(iterations)
        mse_list.append(mse_training)
        # print("W: ", W)
        if g_val < 0.0001:
            print("done")
            break
        W_old = W
    # print("W: ", W)
    """
    fig, ax = plt.subplots()
    ax.plot(iterations_list, mse_list)
    ax.plot(iterations_list, V_list)
    ax.plot(iterations_list, T_list)
    ax.legend((mse_list, V_list, T_list), ('Training Error', 'Validation Error', 'Testing Error'))
    plt.show()
    """
    return W, b, mse_list, V_list, T_list, iterations_list


def sigmoid(W, b, x):
    # print('sigmoid: ')
    # print(np.shape(W))
    # print(np.shape(x))
    # print(np.shape(b))
    z = x.dot(W.transpose()).flatten()+b
    # print('z: ', z)
    return 1/(1+np.exp(-z))


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
    # print("sig:",np.log(sigmoid(W, b, x)))
    # print(1 - y_flat)
    # print('logsig:',(np.log(1 - sigmoid(W, b, x))))
    z = -(y_flat.dot(np.log(sigmoid(W, b, x)))) - np.dot((1 - y_flat), np.log(1 - sigmoid(W, b, x)))
    z = z.sum() / size + 0.5 * reg * LA.norm(W) * LA.norm(W)
    # print(z, np.shape(z))
    return z


def gradCE(W, b, x, y, reg):
    # Your implementation here
    size = y.size
    '''print('Y shape: ', np.shape(y))
    print('W shape: ', np.shape(W))
    print('X shape: ', np.shape(x))
    print('Sigmoid Shape: ', np.shape(sigmoid(W, b, x)))'''
    grad_W = -((y.flatten() - sigmoid(W, b, x)).dot(x))/size + reg*W
    grad_b = (-(y.flatten() - sigmoid(W, b, x)))
    grad_b = np.sum(grad_b)/size
    # print(np.shape(grad_b))
    return grad_W, grad_b


def grad_descent_CE(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    # Your implementation here
    mse_list = []
    iterations_list = []
    V_list=[]
    T_list=[]

    train_acc_list = []
    v_acc_list = []
    test_acc_list = []

    while iterations < 5000:
        g_val, W, b = grad_loop_CE(W, b, trainingData, trainingLabels, reg, alpha)
        mse_training = crossEntropyLoss(W, b, trainingData, trainingLabels, reg)
        iterations += 1
        Error_V = crossEntropyLoss(W, b, validData, validTarget, reg=reg)
        Error_T = crossEntropyLoss(W, b, testData, testTarget, reg=reg)
        V_list.append(Error_V)
        T_list.append(Error_T)
        iterations_list.append(iterations)
        mse_list.append(mse_training)
        train_acc = accuracy(W, b, trainingData, trainingLabels, sig=True)
        v_acc = accuracy(W, b, trainingData, trainingLabels, sig=True)
        test_acc = accuracy(W, b, trainingData, trainingLabels, sig=True)

        train_acc_list.append(train_acc)
        v_acc_list.append(v_acc)
        test_acc_list.append(test_acc)

        # print("W: ", W)
        if g_val < 0.001:
            print("done")
            break
            # print("W: ", W)
    """
    fig, ax = plt.subplots()
    ax.plot(iterations_list, mse_list, label='Training')
    ax.plot(iterations_list, V_list, label='Validation')
    ax.plot(iterations_list, T_list, label='Testing')
    ax.legend(loc='upper right')
    plt.show()
    """
    return W, b, mse_list, V_list, T_list, train_acc_list, v_acc_list, test_acc_list, iterations_list


def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    tf.set_random_seed(421)
    reg=0.0
    n_epochs = 700
    batch_size = 500
    n_batches = int(trainData.shape[0])//batch_size

    # Creation of all required data tensors
    weights = tf.Variable(tf.truncated_normal([784, 1], stddev=0.5, name='Weights'))
    bias = tf.Variable(0, dtype=tf.float32, name='Biases')
    x_data = tf.placeholder(dtype=tf.float32, name='x_data')
    y_labels = tf.placeholder(dtype=tf.float32, name='y_data')
    
    combinedDatLab = np.concatenate((trainData.transpose(), trainTarget.transpose()), axis=0).transpose()
    print(np.shape(combinedDatLab))
    half_reg = tf.constant(0.5*reg, dtype=tf.float32, shape=[1], name='regularization')
    if lossType == 'MSE':
        y_pred_MSE = tf.matmul(x_data, weights) + bias
        mse1 = tf.reduce_mean(tf.square(y_labels - y_pred_MSE))
        # mse1 = tf.losses.mean_squared_error(y_labels, y_pred_MSE)
        mse2 = tf.multiply(half_reg, tf.norm(weights))
        error = tf.add(mse1, mse2)
    
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(mse1)

    init = tf.global_variables_initializer()
    
    train_err_list = []
    valid_err_list = []
    test_err_list = []
    iterations_list = []
    with tf.Session() as sess:
        sess.run(init)
        before = MSE(weights.eval().flatten(), bias.eval(), trainData, trainTarget, reg=0)
        
        for epoch in range(n_epochs):
            # Every epoch, shuffle
            np.random.shuffle(combinedDatLab)  # shuffled data
            new_x = combinedDatLab[:, :784]
            new_y = combinedDatLab[:, 784:785]
            # print(np.shape(new_y))
            # print(np.shape(new_x))
            # print(epoch)
            for i in range(n_batches):
                xvals = new_x[i*batch_size:(i+1)*batch_size, :]
                # print(xvals.eval())
                yvals = new_y[i*batch_size:(i+1)*batch_size]  
                # Now input into the session
                sess.run([y_pred_MSE, optimizer], feed_dict={x_data: xvals, y_labels: yvals})
                err_v = sess.run(error, feed_dict={x_data: validData, y_labels: validTarget})
                err_te = sess.run(error, feed_dict={x_data: testData, y_labels: testTarget})
                # print(np.shape(y_pred))
                # check cost
                err_tr = sess.run(error, feed_dict={x_data: xvals, y_labels:yvals})
                print('error test: ', err_te)
            # print('Error: ', err)
        after = MSE(weights.eval().flatten(), bias.eval(), trainData, trainTarget, reg=reg)
        after_v = MSE(weights.eval().flatten(), bias.eval(), validData, validTarget, reg=reg)
        after_te = MSE(weights.eval().flatten(), bias.eval(), testData, testTarget, reg=reg)
        print('Error before training: ', before)
        print('Error after training: ', after)
        print('Error after valid: ', after_v)
        print("Error after testing: ", after_te)
        
        # get predicted labels
        # y_pred, err_final = sess.run([y_pred_MSE, error], feed_dict={x_data: trainData, y_labels: trainTarget})
        # print('Predicted y: ', y_pred)
        # print('Err: ', err_final)
        
        best_weights = weights.eval()
        best_bias = bias.eval()
    return best_weights, best_bias, optimizer, reg


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
print('shape', np.shape(trainData))
print('shape y', np.shape(trainTarget))
# all_data = preproc(trainData, validData, testData)
# trainData = all_data[0]
# validData = all_data[1]
# testData = all_data[2]
np.random.seed(3)
W = np.zeros((784,))  # np.random.rand(784)
W_init1 = W
W_init2 = W
W_init3 = W
b = 0
"""
CE = crossEntropyLoss(W, 0, trainData, trainTarget, reg=0.0)
gCE = gradCE(W, 0, trainData, trainTarget, reg=0.0)
z = sigmoid(W, 0, trainData)
print('CE: ', CE)
print('shape of gCE: ', np.shape(gCE))
print('gCE: ', gCE)
print('shape of z: ', np.shape(z))
print(z)
# W, b = grad_descent_CE(W, 0, trainData, trainTarget, alpha=0.0003, iterations=0.0, reg=0.0, EPS=1e-7)
"""

"""
start = time.time()
W_1, b_1, mse_list_1, V_list_1, T_list_1, iterations_list_1 = grad_descent(W_init1, 0, trainData, trainTarget, alpha=0.005, iterations=0, reg=0.01, EPS=1e-7)
# W_1_log, b_1_log, ce_list, v_list, t_list, train_acc_list, v_acc_list, test_acc_list, iterations_list = grad_descent_CE(W_init2, 0, trainData, trainTarget, alpha=0.005, iterations=0, reg=0.01, EPS=1e-7)
end = time.time()
print('Test Accuracy for Alpha=0.005: ', accuracy(W_1, b_1, testData, testTarget, sig=False))
# print('Test Accuracy for Alpha=0.005: ', accuracy(W_1_log, b_1_log, testData, testTarget, sig=True))
"""
"""
W_2, b_2, mse_list_2, V_list_2, T_list_2, iterations_list_2 = grad_descent(W_init2, 0, trainData, trainTarget, alpha=0.005, iterations=0, reg=0.1, EPS=1e-7)
print('Test Accuracy for Alpha=0.001: ', accuracy(W_2, b_2, testData, testTarget))
W_3, b_3, mse_list_3, V_list_3, T_list_3, iterations_list_3 = grad_descent(W_init3, 0, trainData, trainTarget, alpha=0.005, iterations=0, reg=0.1, EPS=1e-7)
print('Test Accuracy for Lambda=0.0001: ', accuracy(W_3, b_3, testData, testTarget))
"""

# print('Accuracy for Lambda=0.001: ', accuracy(W_1, b_1, testData, testTarget))
# print('Accuracy for Lambda=0.1: ', accuracy(W_2, b_2, testData, testTarget))
# print('Accuracy for Lambda=0.5: ', accuracy(W_3, b_3, testData, testTarget))
'''
fig = plt.figure()

plt.subplot(121)
# plt.plot(iterations_list_1, mse_list_1, label='Training')
# plt.plot(iterations_list_1, V_list_1, label='Validation')
plt.plot(iterations_list_1, T_list_1, label='Testing')
plt.xlabel('Number of Iterations')
plt.ylabel('MSE Error Value')
plt.legend(loc='upper right')
plt.title('MSE Test Error when Learning Rate = 0.005, Reg = 0.01')

# plt.subplot(122)
# plt.plot(iterations_list, t_list, label='Testing')
# plt.xlabel('Number of Iterations')
# plt.ylabel('Cross-Entropy Error Value')
# plt.legend(loc='upper right')
# plt.title('CE Test Error when Learing Rate = 0.005, Reg = 0.01')
'''

"""
plt.subplot(132)
plt.plot(iterations_list_2, mse_list_2, label='Training')
plt.plot(iterations_list_2, V_list_2, label='Validation')
plt.plot(iterations_list_2, T_list_2, label='Testing')
plt.legend(loc='upper right')
plt.xlabel('Number of Iterations')
plt.ylabel('MSE Error Value')
plt.title('Error when Learning Rate = 0.001')

plt.subplot(133)
plt.plot(iterations_list_3, mse_list_3, label='Training')
plt.plot(iterations_list_3, V_list_3, label='Validation')
plt.plot(iterations_list_3, T_list_3, label='Testing')
plt.legend(loc='upper right')
plt.xlabel('Number of Iterations')
plt.ylabel('MSE Error Value')
plt.title('Error when Learning Rate = 0.0001')

plt.suptitle("Error for All Datasets for Learning Rate = 0.005, 0.001, 0.0001", fontsize=14)
"""
# plt.show()


"""
start_nideal = time.time()
W_1, b_1, mse_list_1, V_list_1, T_list_1, iterations_list_1 = grad_descent(W_init1, 0, trainData, trainTarget, alpha=0.005, iterations=0, reg=0.0001, EPS=1e-7)
end_nideal = time.time()
error_nideal = mse_list_1[len(mse_list_1) - 1]
start_acc = time.time()
acc_nideal = accuracy(W_1, b_1, trainData, trainTarget)
end_acc = time.time()

print("time of acc func: ", end_acc - start_acc)

start_ideal = time.time()
W_ideal = np.matmul(np.matmul(inv(np.matmul(trainData.transpose(), trainData)), trainData.transpose()), trainTarget)
end_ideal = time.time()
W_ideal = W_ideal.flatten()
error_ideal = MSE(W_ideal, 0, trainData, trainTarget, reg=0.0)
acc_ideal = accuracy(W_ideal, 0, trainData, trainTarget)

print('===Comparison of Optimal Solution to Trained Solution===')
print('Time of Trained solution: ', end_nideal - start_nideal)
print('Time of Optimal solution: ', end_ideal - start_ideal)
print('Error of Trained solution: ', error_nideal)
print('Error of Optimal solution: ', error_ideal)
print('Accuracy of Trained solution: ', acc_nideal)
print('Accuracy of Optimal solution: ', acc_ideal)
"""

"""
# gradCE(W, b, trainData, trainTarget, reg=0.0)
# gradMSE(W, b, trainData, trainTarget, reg=0.0)
# print(trainTarget.flatten())
print(np.shape(validData))
print(trainData)
W_1_log, b_1_log, ce_list, v_list, t_list, train_acc_list, v_acc_list, test_acc_list, iterations_list = grad_descent_CE(W, 0, trainData, trainTarget, alpha=0.005, iterations=0, reg=0.1, EPS=0)

fig = plt.figure()
plt.subplot(221)
plt.plot(iterations_list, ce_list, label='Training Error')
plt.xlabel('Number of Iterations')
plt.ylabel('Cross-Entropy Error Value')
plt.legend(loc='upper right')
plt.title('Training Error when Regularization=0.1, Alpha=0.005')

plt.subplot(222)
plt.plot(iterations_list, train_acc_list, label='Training Accuracy')
plt.xlabel('Number of Iterations')
plt.ylabel('Accuracy Value')
plt.legend(loc='upper right')
plt.title('Training Accuracy when Regularization = 0.1, Alpha=0.005')

plt.subplot(223)
plt.plot(iterations_list, t_list, label='Test Error')
plt.xlabel('Number of Iterations')
plt.ylabel('Cross-Entropy Error Value')
plt.legend(loc='upper right')
plt.title('Testing Error when Regularization = 0.1, Alpha=0.005')

plt.subplot(224)
plt.plot(iterations_list, test_acc_list, label='Test Accuracy')
plt.xlabel('Number of Iterations')
plt.ylabel('Accuracy Error Value')
plt.legend(loc='upper right')
plt.title('Testing Accuracy when Regularization = 0.1, Alpha=0.005')

print('Final Validation Accuracy: ', v_acc_list[len(v_acc_list) - 1])
print('Final Testing Accuracy: ', test_acc_list[len(test_acc_list) - 1])
print('Final Validation Error: ', v_list[len(v_list) - 1])
print('Final Testing Error: ', t_list[len(t_list) - 1])

plt.show()
"""

buildGraph(lossType='MSE', learning_rate=0.0001)

'''
atStart = MSE(W, 0, trainData, trainTarget, reg=0.0)



atEnd = MSE(W, b, trainData, trainTarget, reg=0.0)
print(atStart, atEnd)

print('validation error: ', MSE(W, b, validData, validTarget, reg=0.0))
print('test error: ', MSE(W, b, testData, testTarget, reg=0.0))
# print("W", np.matmul(np.matmul(inv(np.matmul(trainData.transpose(), trainData)), trainData.transpose()), trainTarget))
'''

