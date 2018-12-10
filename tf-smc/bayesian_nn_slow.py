#-*- coding:utf8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, warnings, pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, Dirichlet, InverseGamma, ParamMixture, Bernoulli
warnings.simplefilter('ignore')

def build_toy_dataset():
    features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
    df = pd.read_csv('./train-winequality-white.csv', sep=';')
    X = df[features].values
    y = (df.quality >= 7).values.astype(np.float32)

    # 标准化处理
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    return X, y

def neural_network(X, W_0, W_1, W_2, b_0, b_1, b_2):
    h = tf.tanh(tf.matmul(X, W_0) + b_0)
    h = tf.tanh(tf.matmul(h, W_1) + b_1)
    #h = tf.matmul(h, W_2) + b_2
    h = tf.sigmoid(tf.matmul(h, W_2) + b_2)
    return tf.reshape(h, [-1])

def pack_theta( W_0, W_1, W_2, b_0, b_1, b_2):
    pW_0 = tf.reshape(W_0, [-1])
    pW_1 = tf.reshape(W_1, [-1])
    pW_2 = tf.reshape(W_2, [-1])
    pb_0 = tf.reshape(b_0, [-1])
    pb_1 = tf.reshape(b_1, [-1])
    pb_2 = tf.reshape(b_2, [-1])
    return tf.concat(values=[pW_0, pW_1, pW_2, pb_0, pb_1, pb_2], axis=0)

ed.set_seed(42)
layer_size = [11, 7, 10, 1]
J = 20000
B = 5000
K = 20

alls = time.time()
# DATA
X_train, y_train = build_toy_dataset() #真实数据
N, D = X_train.shape
#print(N, D)

# MODEL
model_start = time.time()
C = 5
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
X = tf.placeholder(tf.float32, [N, D], name="X")
#y = Normal(loc=neural_network(X, W_0, W_1, W_2, b_0, b_1, b_2), scale=0.1 * tf.ones(N), name="y")
y = Bernoulli(probs=neural_network(X, W_0, W_1, W_2, b_0, b_1, b_2), name="y")
print('Model Building time: ', time.time()-model_start)

# INFERENCE FRAMEWORK
frm_start = time.time()
# INFERENCE FRAMEWORK
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

# Sample functions from variational model to visualize fits.
'''mus = tf.stack(
    [neural_network(X_train, qW_0.sample(), qW_1.sample(),
              qb_0.sample(), qb_1.sample())
    for _ in range(B)])
'''
mus = tf.stack(
    [pack_theta(qW_0.sample(), qW_1.sample(), qW_2.sample(),
              qb_0.sample(), qb_1.sample(), qb_2.sample())
    for _ in range(B)])
print('Inference Framework Building time: ', time.time() - frm_start)

inf_time = time.time()
sess = ed.get_session()
tf.global_variables_initializer().run()
#print("Initializing the graph...")

inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                W_1: qW_1, b_1: qb_1,
                W_2: qW_2, b_2: qb_2}, data={X: X_train, y: y_train})
inference.run(n_iter=J, n_samples=K)
print('Inference(via VI) time: ', time.time() - inf_time)

smpl_time = time.time()
outputs = mus.eval()
print('Sample time: ', time.time() - smpl_time)
dump_time = time.time()
with open('../smc2/theta'+'-'+str(J)+'-'+str(B),'wb') as f:
    pickle.dump(outputs, f)
print('Dump time: ', time.time() - dump_time)
print('Program time: ', time.time() - alls)
