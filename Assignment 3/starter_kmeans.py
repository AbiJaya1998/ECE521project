import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)
num_class = 1
is_valid = 	False
# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]


def errorFunc(pair_dist):
    minval = tf.reduce_min(pair_dist , 1)
    return tf.reduce_mean(minval)

"""
# Distance function for K-means
def distanceFunc(X, MU):
    N = num_pts
    D = dim
    K = num_class
    #pair_dist = []
    ''' X_s =  X*X
    X_norm = tf.reduce_sum(X_s,axis=1,keep_dims = True)
  
    MU_s = MU*MU
    MU_norm = tf.reduce_sum(MU_s,axis=1,keep_dims = True)

    dot = tf.matmul(MU,tf.transpose(X))
    outer = tf.transpose(X_norm)+MU_norm
    pair_dist = outer - 2*dot
    error = errorFunc(pair_dist)
    return error '''
  
    for i in range(N):
        for j in range(MU):
           curr =  mew
           diff = (row - mew)
           cur_error =np.linalg.norm(diff)
           row_dist.append(cur_error)

        #pair_dist.append(row_dist) #np.concatenate((np.array(pair_dist),np.array(row_dist)),axis=0)
        #print(np.shape(np.array(pair_dist)))
             
          
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO
"""

def distance(X, MU):
    N = num_pts
    D = dim
    K = num_class
    X_expand = tf.expand_dims(X,0)
    MU_expand = tf.expand_dims(MU,1)
    distances = tf.reduce_sum(tf.square(tf.subtract( X_expand , MU_expand )),2)
    return errorN(tf.transpose(distances)) 


def errorN(pair_dist):
    N = num_pts
    D = dim
    K = num_class
   
    xvals = np.linspace(0,9999,num = 10000)
    X = tf.reshape(tf.convert_to_tensor(xvals,np.int64),[10000,1])
    assign = tf.reshape(tf.argmin(pair_dist,axis = 1),[10000,1])
    mean = tf.concat([X,assign], axis = 1)
    means = tf.reduce_sum(tf.gather_nd(pair_dist,mean))
    ''' for c in xrange(K):
       means.append(tf.reduce_mean(tf.gather(pair_dist,tf.reshape(tf.where(tf.equal(assign, c) ),[1,-1])),reduction_indices = [1]))


    #tf.concat([X,tf.argmin(pair_dist,axis = 1)],axis = 1)
    #assign = tf.gather_nd(pair_dist,assign)
    #for x in range(N):
    print np.shape(means)'''
    return assign ,X,means,pair_dist


def learning(data,learning_rate = 0.1,epochs = 200):
     tf.set_random_seed(421)
     N = num_pts
     D = dim
     K = num_class
    
     x_data = tf.placeholder(dtype=tf.float32, name='x_data')
     MU = tf.Variable(tf.random_normal([K, D]), name="mu")
     assign,X,M, error = distance(x_data,MU)
     #update_centroids = tf.assign(MU, M)

     optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9, beta2=0.99, epsilon=1e-6).minimize(M)

     init = tf.global_variables_initializer()
     with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs):
             mu,err,ass,m,x,op = sess.run([MU,error,assign,M,X,optimizer1] , feed_dict={x_data: data})
             #print((m))
        M = MU.eval()
        
     list_clusters = groupClusters(data, M)
     return list_clusters, M


def groupClusters(X, MU):
    """
    X: N x D
    MU: K x D
    
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
    
    X_ex = np.expand_dims(X, axis=1)
    MU_ex = np.expand_dims(MU, axis=0)
    mean_dist = np.linalg.norm(X_ex - MU_ex, axis=-1) #N x K matrix
    min_ax = np.argmin(mean_dist, axis = 1) # N x 1
    for i in range(K): 
        locations = min_ax == i
        cluster = X[locations,:]
        print(np.shape(cluster))
        list_clusters.append(cluster)
    return list_clusters
    

def dispGraphs(means, list_clusters):
    m1 = means[:, 0]
    m2 = means[:, 1]
    
    fig, ax = plt.subplots()
    for item in list_clusters:
        item1 = item[:, 0]
        item2 = item[:, 1]
        ax.scatter(item1, item2, s=5)
    # ax.scatter(x1, x2)
    means = ax.scatter(m1, m2, c="black", marker="^")
    plt.legend((means,), ('Cluster Centres',))
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("K-means clustering for K = " + str(num_class))
    plt.show()
    return

"""
min_val = np.amin(data)
max_val = np.amax(data)
data = (data - min_val) / (max_val - min_val)
"""

print (num_pts)
print (dim)
print (np.shape(data))
list_clusters, m = learning(data)
print(np.shape(m))
dispGraphs(m, list_clusters)



