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
num_class = 5
# For Validation set
is_valid = True
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
    X_expand = tf.expand_dims(X,1) # N x 1 x D
    MU_expand = tf.expand_dims(mu,0) # 1 x K x D
    distances = tf.reduce_sum(tf.square(tf.subtract(X_expand , MU_expand)), axis=2) # N x K
    
    inv_var = 1 / var # 1 x K
    dist_inv_var = distances*(inv_var) # N x K
    
    log_two_pi_variance = D*tf.log(2 * np.pi) + D*tf.log(var)  # 1 x K
    
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
        log_sum = log_GaussPDF(X,mu,sigma) + tf.transpose(pi) # N X K
        log_sum_pi_gauss = tf.reshape(reduce_logsumexp(log_sum, 1), [-1, 1])# N X 1

        return log_sum - log_sum_pi_gauss


def neg_log(X,mu,sigma,pi):
        log_sum = log_GaussPDF(X,mu,sigma) + (tf.transpose(pi))#N X K
        log_pi_gauss = tf.reshape(reduce_logsumexp(log_sum, 1), [-1, 1])# N X 1
    
        return -1*tf.reduce_sum(log_pi_gauss),log_sum - log_pi_gauss


def learning(data, val_data = None, learning_rate = 0.01,epsilon=1e-5, epochs = 600):
    N = num_pts
    D = dim
    K = num_class
    Y = []
    Y_v = []
    Xv = []
    tf.set_random_seed(421)
    # tf.random_normal([K,1], mean = 0, stddev=0.5)
    X = tf.placeholder(dtype=tf.float32, name="data") # N x D
    sigma = tf.Variable(tf.exp(tf.random_normal([K], mean = 0.0, stddev=0.5)))
    pi = logsoftmax(tf.Variable(tf.random_normal([K,1], mean = 0.0, stddev=0.5)))
    MU = tf.Variable(tf.random_normal([K, D],mean = 0.0 , stddev = 1.0 , dtype = tf.float32), name="mu")
    
    error ,log_probs = neg_log(X,MU,sigma,pi)
    sols = tf.argmax(log_probs,axis =1)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1=0.9, beta2=0.99, epsilon = 1e-5).minimize(error)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs):
            logs,m,er,op = sess.run([log_probs, MU,error,optimizer] , feed_dict={X: data})
            
            if is_valid == True:
                logs_v, m_v, er_v, op_v = sess.run([log_probs, MU,error,optimizer] , feed_dict={X: val_data})
                Y_v.append(er_v)
            
            Y.append(er)
            Xv.append(i)

        cluster_assignments = sess.run(sols, feed_dict={X: data}) 
        M = MU.eval()
        P = pi.eval()
        S = sigma.eval()
        # print 'Training Error for K = ' + str(num_class) + ': ', Y[-1]
        # print 'Validation Error for K = ' + str(num_class) + ': ', Y_v[-1]
    list_clusters = groupClusters(data, logs)
    return list_clusters, M, P, S, Xv, Y, Y_v


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
        num_clusts, dims = cluster.shape
        print 'Number of Points in Cluster ' + str(i + 1) + ': ', num_clusts
        list_clusters.append(cluster)
    return list_clusters
    

def dispGraphs(means, list_clusters):
    m1 = means[:, 0]
    m2 = means[:, 1]
    fig, ax = plt.subplots()
    #print list_clusters
    counter = 0
    for item in list_clusters:
        counter += 1
        print('hi')
        item1 = item[:, 0]
        item2 = item[:, 1]
    
        num, dim = item.shape
        if num != 0: 
            scatter = ax.scatter(item1, item2, s=5, alpha=0.5, label='Data Cluster ' + str(counter))

        # ax.scatter(x1, x2)
        #print(m1,m2)

    means = ax.scatter(m1, m2, c="black", marker="^")
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Gaussian Mixture Model for K = " + str(num_class) + ', Learning Rate = 0.01')
    plt.show()
    return



'''def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # TODO'''


#data = (data - mean_data) / stddev_data
list_clusters, m, p, s, X, Y, Y_v = learning(data, val_data)

print 'Final Training Error: ', Y[-1]
print 'Final Validation Error: ', Y_v[-1]

# print "Trained Pi values for K = " + str(num_class) + " clusters: ", p
# print "Trained Standard Deviations for K = " + str(num_class) + " clusters: ", s
# print "Trained Cluster centres for K = " + str(num_class) + " clusters: ", m

dispGraphs(m, list_clusters)
plt.figure(2)
loss = plt.plot(np.asarray(X),np.asarray(Y))
plt.xlabel('Number of Updates')
plt.grid(b=True)
plt.ylabel('Loss')
plt.title('Gaussian Mixture Model Loss vs. Number of Updates for K = ' + str(num_class))
plt.show()
