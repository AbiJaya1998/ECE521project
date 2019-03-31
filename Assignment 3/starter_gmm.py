import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
from math import pi
from math import sqrt
from helper import reduce_logsumexp
from helper import logsoftmax

# Loading data
data = np.load('data100D.npy')
#data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)
num_class = 20
# For Validation set
is_valid = False
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    N = num_pts
    D = dim
    K = num_class

    X_expand = tf.expand_dims(X,0)
    MU_expand = tf.expand_dims(MU,1)
    distances = tf.reduce_sum(tf.square(tf.subtract( X_expand , MU_expand )),2)

    return tf.transpose(distances)
    # TODO


def log_GaussPDF(X, mu, var):
    # We should have a D x D covariance matrix for each of the K
    # clusters. But each covariance matrix is a diagonal, and the
    # variance along the diagonal is the same for each cluster. So,
    # we can represent this with just a column vector of variances,
    # one for each cluster.
    #
    # X   is N X D
    # mu  is K x D
    # var is 1 x K
    N = num_pts
    D = dim
    K = num_class

    var = tf.transpose(var) # 1 x K
    X_expand = tf.expand_dims(X,0) # 1 x N x D
    MU_expand = tf.expand_dims(mu,1) # K x 1 x D
    distances = tf.transpose(tf.reduce_sum(tf.subtract(X_expand , MU_expand), axis=2)) # K x N
    
    inv_var = 1 / var # 1 x K
    dist_inv_var = distances*tf.broadcast_to((inv_var),[N,K]) # N x K
    
    log_two_pi_variance = D*tf.log(2 * np.pi) + tf.log(var)  # 1 x K
    
    return -0.5 * (log_two_pi_variance + dist_inv_var) # N x K


'''
def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1
    # log_pi: K X 1
    N = num_pts
    D = dim
    K = num_class
    
    X_expand = tf.expand_dims(X,0) # 1 x N x D
    MU_expand = tf.expand_dims(mu, 1) # K x 1 x D
    X_centered = X_expand - MU_expand # K x N x D
    

    sigma = 
    X_centered = X - ith_vec # Broadcasted to 1 x D x N
    var_covar = (1/N)*tf.matmul(tf.transpose(X_centered, perm=[0, 2, 1]), X_centered) # D x D x N
    
    inv_var_covar = tf.linalg.inv(var_covar) # D x D
    det_var_covar = tf.linalg.det(var_covar) # 1 x 1
    
    log_exp_1 = tf.tensordot(X_centered, inv_var_covar, axes=[[0], [1]])
    log_exp_2 = tf.tensordot(log_exp_1, tf.transpose(X_centered, perm=[0, 2, 1]))

    log_exp = -0.5*tf.log(tf.tensordot(X_centered, inv_var_covar, axes=[[0], [1]]))
    log_pi_cov = -0.5*(D * tf.log(2*pi) + tf.log(det_var_covar))
    log_total = log_exp + log_pi_cov
    list_tens.append() # 1 x N, appended to list
        

    stacked = tf.stack(list_tens, axis=-1) # 1 x N x K
    squeezed = tf.squeeze(stacked) # N x K
    
    

    mean_X_t = distances_t # K x N
    mean_X = distances # N x K

    var_covar = (1/N)*tf.matmul(distances_t, distances) # K x K
    print(tf.shape(var_covar))
    inv_var_covar = tf.linalg.inv(var_covar) # K x K
    det_var_covar = tf.linalg.det(var_covar) # 1 x 1

    logGaussPDF = -0.5*tf.matmul(tf.matmul(distances, inv_var_covar), distances_t) - 0.5*(D*tf.log(2*pi) + tf.log(det_var_covar))
    return tf.transpose(logGaussPDF)
    # Outputs:
    # log Gaussian PDF N X K
    
    # TODO
'''


def find_p_zgx(X,mu,sigma,pi):
        log_sum = log_GaussPDF(X,mu,sigma) + tf.log(tf.transpose(pi))# N X K
        log_sum_pi_gauss = tf.reshape(reduce_logsumexp(log_sum, 1), [-1, 1])# N X 1

        return log_sum- log_sum_pi_gauss


def neg_log(X,mu,sigma,pi):
        log_sum = log_GaussPDF(X,mu,sigma*sigma) + tf.log(tf.transpose(pi))#N X K
        log_pi_gauss = tf.reshape(reduce_logsumexp(log_sum, 1), [-1, 1])# N X 1
    
        return -1*tf.reduce_sum(log_pi_gauss)


def learning(data,learning_rate = 0.5,epsilon=1e-5, epochs = 50):
    N = num_pts
    D = dim
    K = num_class
    tf.set_random_seed(421)

    X = tf.placeholder(dtype=tf.float32, name="data") # N x D
    sigma = tf.exp(tf.Variable(tf.truncated_normal([K, 1], mean=0, stddev=0.5)))
    pi = tf.exp(logsoftmax(tf.Variable(tf.truncated_normal([K,1], mean=0, stddev=0.5))))
    MU = tf.Variable(tf.random_normal([K, D]), name="mu")
    log_probs = find_p_zgx(X, MU, sigma, pi)
    sols = tf.argmax(log_probs,axis =1)
    error = neg_log(X,MU,sigma,pi)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1=0.9, beta2=0.99, epsilon = 1e-5).minimize(error)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs):
            logs, m,er,op = sess.run([log_probs, MU,error,optimizer] , feed_dict={X: data})
            print er
        cluster_assignments = sess.run(sols, feed_dict={X: data}) 
        print cluster_assignments
        M = MU.eval()
    
    list_clusters = groupClusters(data, logs)
    return list_clusters, M


def groupClusters(X, log_probs):
    """
    X: N x D
    log_probs: N x K
    
    Make K arrays, each array having the coordinates for the points in the
    cluster
    -Find N x K matrix (where each row represents the distances to each of the
     K means)
    -Use arg_min to find the indices that have minimum K distance to means
    """
    N = num_pts
    D = dim
    K = num_class
    
    list_clusters = []
    max_logs = np.argmax(log_probs, axis=1)
    print('max_logs: ', (log_probs[0]))
    for i in range(K): 
        locations = max_logs == i
        cluster = X[locations,:]
        print(np.shape(cluster))
        list_clusters.append(cluster)
    return list_clusters
    

def dispGraphs(means, list_clusters):
    m1 = means[:, 0]
    m2 = means[:, 1]
    
    fig, ax = plt.subplots()
    #print list_clusters
    for item in list_clusters:
        print('hi')
        item1 = item[:, 0]
        item2 = item[:, 1]
        ax.scatter(item1, item2, s=5)
    # ax.scatter(x1, x2)
    #print(m1,m2)
    means = ax.scatter(m1, m2, c="black", marker="^")
    plt.legend((means,), ('Cluster Centres',))
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("K-means clustering for K = " + str(num_class))
    plt.show()
    return


'''def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # TODO'''

list_clusters, m = learning(data)
dispGraphs(m, list_clusters)
