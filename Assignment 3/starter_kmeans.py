import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)
num_class = 3
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


# Distance function for K-means
def distanceFunc(X, MU):
    N = num_pts
    D = dim
    K = num_class
    #pair_dist = []
    X_s =  X*X
    X_norm = tf.reduce_sum(X_s,axis=1,keep_dims = True)
  
    MU_s = MU*MU
    MU_norm = tf.reduce_sum(MU_s,axis=1,keep_dims = True)

    dot = tf.matmul(MU,tf.transpose(X))
    outer = tf.transpose(X_norm)+MU_norm
    pair_dist = outer - 2*dot
    error = errorFunc(pair_dist)
    return error 
  
    '''for row in X:
        row_dist = []
        for mew in MU:
           curr =  mew
           diff = (row - mew)
           cur_error =np.linalg.norm(diff)
           row_dist.append(cur_error)'''
        #pair_dist.append(row_dist) #np.concatenate((np.array(pair_dist),np.array(row_dist)),axis=0)
        #print(np.shape(np.array(pair_dist)))
             
          
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO



def learning(data,learning_rate = 0.1,epochs = 200):
     tf.set_random_seed(421)
     N = num_pts
     D = dim
     K = num_class
    
     x_data = tf.placeholder(dtype=tf.float32, name='x_data')
     MU = tf.Variable(tf.random_normal([K, D]), name="mu")
     error = distanceFunc(x_data,MU)
     optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9, beta2=0.99,
epsilon=1e-6).minimize(error)

     init = tf.global_variables_initializer()
     with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs):
             mu,err, op = sess.run([MU,error ,optimizer1] , feed_dict={x_data: data})
             print(err)
       
   


        M = MU.eval()
     return M


"""
min_val = np.amin(data)
max_val = np.amax(data)
data = (data - min_val) / (max_val - min_val)
"""

print (num_pts)
print (dim)
print (np.shape(data))
m = learning(data)
print(np.shape(m))
m1 = m[:, 0]
m2 = m[:, 1]

x1 = data[:,0]
x2 = data[:,1]
fig, ax = plt.subplots()
ax.scatter(x1, x2)
ax.scatter(m1, m2)
plt.show()

