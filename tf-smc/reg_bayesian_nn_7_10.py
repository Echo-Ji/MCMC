#-*- coding:utf8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, warnings, pickle
import pandas as pd
import edward as ed
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from edward.models import Normal, Dirichlet, InverseGamma, ParamMixture, Bernoulli
warnings.simplefilter('ignore')

'''
Definition of some helper functions.
'''
def build_dataset(data_file):
    # Just support csv data.
    df = pd.read_csv(data_file)
    X = df.iloc[:,:-1].astype(np.float32)
    y = df.iloc[:,-1].astype(np.float32)
    return X, y

def neural_network(X, W_0, W_1, W_2, b_0, b_1, b_2):
    # Neural Network with two hidden layers.
    h = tf.tanh(tf.matmul(X, W_0) + b_0)
    h = tf.tanh(tf.matmul(h, W_1) + b_1)
    h = tf.matmul(h, W_2) + b_2
    return tf.reshape(h, [-1])

def pack_theta( W_0, W_1, W_2, b_0, b_1, b_2):
    pW_0 = np.reshape(W_0, [-1])
    pW_1 = np.reshape(W_1, [-1])
    pW_2 = np.reshape(W_2, [-1])
    pb_0 = np.reshape(b_0, [-1])
    pb_1 = np.reshape(b_1, [-1])
    pb_2 = np.reshape(b_2, [-1])
    return np.concatenate((pW_0, pW_1, pW_2, pb_0, pb_1, pb_2), axis=0)

'''
Program starts from here.
'''
ed.set_seed(42)
alls = time.time()

# DATA
reg_pre = 'CO_'
train_file = './data/'+reg_pre+'train.csv'
X_train, Y_train = build_dataset(train_file) #真实数据
N, D = X_train.shape
layer_size = [D, 7, 10, 1]

# MODEL
model_start = time.time()
C = 4  # expected number of modes in posterior distribution
with tf.variable_scope("W_0"):
    probs = Dirichlet(tf.ones([D, 7, C]))
    mu = Normal(tf.zeros([D, 7]), 10*tf.ones([D, 7]), sample_shape=C)
    sigmasq = InverseGamma(tf.ones([D, 7]), tf.ones([D, 7]), sample_shape=C)
    W_0 = ParamMixture(probs, {'loc': mu, 'scale': 10*tf.sqrt(sigmasq)}, Normal)
with tf.variable_scope("W_1"):
    probs = Dirichlet(tf.ones([7, 10, C]))
    mu = Normal(tf.zeros([7, 10]), 10*tf.ones([7, 10]), sample_shape=C)
    sigmasq = InverseGamma(tf.ones([7, 10]), tf.ones([7, 10]), sample_shape=C)
    W_1 = ParamMixture(probs, {'loc': mu, 'scale': 10*tf.sqrt(sigmasq)}, Normal)
with tf.variable_scope("W_2"):
    probs = Dirichlet(tf.ones([10, 1, C]))
    mu = Normal(tf.zeros([10, 1]), 10*tf.ones([10, 1]), sample_shape=C)
    sigmasq = InverseGamma(tf.ones([10, 1]), tf.ones([10, 1]), sample_shape=C)
    W_2 = ParamMixture(probs, {'loc': mu, 'scale': 10*tf.sqrt(sigmasq)}, Normal)
with tf.variable_scope("b_0"):
    probs = Dirichlet(tf.ones([7, C]))
    mu = Normal(tf.zeros(7), 10*tf.ones(7), sample_shape=C)
    sigmasq = InverseGamma(tf.ones(7), tf.ones(7), sample_shape=C)
    b_0 = ParamMixture(probs, {'loc': mu, 'scale': 10*tf.sqrt(sigmasq)}, Normal)
with tf.variable_scope("b_1"):
    probs = Dirichlet(tf.ones([10, C]))
    mu = Normal(tf.zeros(10), 10*tf.ones(10), sample_shape=C)
    sigmasq = InverseGamma(tf.ones(10), tf.ones(10), sample_shape=C)
    b_1 = ParamMixture(probs, {'loc': mu, 'scale': 10*tf.sqrt(sigmasq)}, Normal)
with tf.variable_scope("b_2"):
    probs = Dirichlet(tf.ones([1, C]))
    mu = Normal(tf.zeros(1), 10*tf.ones(1), sample_shape=C)
    sigmasq = InverseGamma(tf.ones(1), tf.ones(1), sample_shape=C)
    b_2 = ParamMixture(probs, {'loc': mu, 'scale': 10*tf.sqrt(sigmasq)}, Normal)
X = tf.placeholder(tf.float32, [None, D], name="X")
# regression
y_ = tf.placeholder(tf.float32, [None], name = "y_placeholder")
Y = Normal(loc=neural_network(X, W_0, W_1, W_2, b_0, b_1, b_2), scale=0.1, name="Y") #broadcast the scale
# classification
#y_ = tf.placeholder(tf.int32, [None], name = "y_placeholder")
#Y = Bernoulli(probs=neural_network(X, W_0, W_1, W_2, b_0, b_1, b_2), name="Y")
print('Model Building time: ', time.time()-model_start)

# INFERENCE FRAMEWORK
frm_start = time.time()
with tf.variable_scope("qW_0"):
    probs = tf.get_variable("probs", [D, 7, C], initializer=tf.constant_initializer(1.0 / C))
    mu = tf.get_variable("loc", [C, D, 7])
    sigma = tf.nn.softplus(tf.get_variable("scale", [C, D, 7]))
    qW_0 = ParamMixture(probs, {'loc': mu, 'scale': sigma}, Normal)
with tf.variable_scope("qW_1"):
    probs = tf.get_variable("probs", [7, 10, C], initializer=tf.constant_initializer(1.0 / C))
    mu = tf.get_variable("loc", [C, 7, 10])
    sigma = tf.nn.softplus(tf.get_variable("scale", [C, 7, 10]))
    qW_1 = ParamMixture(probs, {'loc': mu, 'scale': sigma}, Normal)
with tf.variable_scope("qW_2"):
    probs = tf.get_variable("probs", [10, 1, C], initializer=tf.constant_initializer(1.0 / C))
    mu = tf.get_variable("loc", [C, 10, 1])
    sigma = tf.nn.softplus(tf.get_variable("scale", [C, 10, 1]))
    qW_2 = ParamMixture(probs, {'loc': mu, 'scale': sigma}, Normal)
with tf.variable_scope("qb_0"):
    probs = tf.get_variable("probs", [7, C], initializer=tf.constant_initializer(1.0 / C))
    mu = tf.get_variable("loc", [C, 7])
    sigma = tf.nn.softplus(tf.get_variable("scale", [C, 7]))
    qb_0 = ParamMixture(probs, {'loc': mu, 'scale': sigma}, Normal)
with tf.variable_scope("qb_1"):
    probs = tf.get_variable("probs", [10, C], initializer=tf.constant_initializer(1.0 / C))
    mu = tf.get_variable("loc", [C, 10])
    sigma = tf.nn.softplus(tf.get_variable("scale", [C, 10]))
    qb_1 = ParamMixture(probs, {'loc': mu, 'scale': sigma}, Normal)
with tf.variable_scope("qb_2"):
    probs = tf.get_variable("probs", [1, C], initializer=tf.constant_initializer(1.0 / C))
    mu = tf.get_variable("loc", [C, 1])
    sigma = tf.nn.softplus(tf.get_variable("scale", [C, 1]))
    qb_2 = ParamMixture(probs, {'loc': mu, 'scale': sigma}, Normal)
print('Inference Framework Building time: ', time.time() - frm_start)

# Variational Inference
inf_time = time.time()
B = 5000
K = 20
epochs = 2000 # change it to get more iteration
batchsz = 10000
n_batches = len(X_train) // batchsz
J = n_batches*epochs
sess = tf.InteractiveSession()
inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                     W_1: qW_1, b_1: qb_1,
                     W_2: qW_2, b_2: qb_2}, data={Y: y_})
inference.initialize(n_samples=K, n_iter=J)
sess.run(tf.global_variables_initializer())
for i in range(epochs):
    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
    for j in range(n_batches):
        X_batch = X_train[j*batchsz:(j+1)*batchsz]
        Y_batch = Y_train[j*batchsz:(j+1)*batchsz]
        info_dict = inference.update(feed_dict= {X: X_batch,  y_: Y_batch})
        inference.print_progress(info_dict)
inference.finalize()
print('Inference(via VI) time: ', time.time() - inf_time)

# Samples from Posterior Distribution
smpl_time = time.time()
qw0 = qW_0.sample(B)
qw1 = qW_1.sample(B)
qw2 = qW_2.sample(B)
qb0 = qb_0.sample(B)
qb1 = qb_1.sample(B)
qb2 = qb_2.sample(B)
qw0, qw1, qw2, qb0, qb1, qb2 = sess.run([qw0, qw1, qw2, qb0, qb1, qb2])
outputs = np.stack(
    [pack_theta(w0, w1, w2, b0, b1, b2)
    for w0, w1, w2, b0, b1, b2 in zip(qw0, qw1, qw2, qb0, qb1, qb2)])
print('Sample time: ', time.time() - smpl_time)

# Samples Dumping
dump_time = time.time()
with open('./theta/2h-7-10-10v/'+reg_pre+'fb'+str(C)+'-'+str(J)+'-'+str(B)+'-2h-10v','wb') as f:
    pickle.dump(outputs, f)
print('Dump time: ', time.time() - dump_time)

# Evaluation
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

def np_neural_network(X, W_0, W_1, W_2, b_0, b_1, b_2):
    h = np.tanh(np.matmul(X, W_0) + b_0)
    h = np.tanh(np.matmul(h, W_1) + b_1)
    h = np.matmul(h, W_2) + b_2
    return np.reshape(h, [-1])

evl_time = time.time()
test_file = './data/'+reg_pre+'test.csv'
X_test, Y_test = build_dataset(test_file)

res = []
for W_0, W_1, W_2, b_0, b_1, b_2 in zip(qw0, qw1, qw2, qb0, qb1, qb2):
    res.append(np_neural_network(X_test, W_0, W_1, W_2, b_0, b_1, b_2))
y_pred = np.mean(res, axis=0)

r_val = rmsle(Y_test, y_pred)
print('Mode Number: ', C)
print('Rmsle Value: ', r_val)
print('Rmsle Time: ', time.time() - evl_time)
print('Program time: ', time.time() - alls)
