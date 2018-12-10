#-*- coding:utf8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

import tensorflow as tf
import edward as ed
import edward.models as edm

#a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
#b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
#c = a + b
C = 2
with tf.variable_scope("W_1"):
    probs = edm.Dirichlet(tf.ones([7, 10, C]))
    mu = edm.Normal(tf.zeros([7, 10]), tf.ones([7, 10]), sample_shape=C)
    sigmasq = edm.InverseGamma(tf.ones([7, 10]), tf.ones([7, 10]), sample_shape=C)
    W_1 = edm.ParamMixture(probs, {'loc': mu, 'scale': 10*tf.sqrt(sigmasq)}, edm.Normal)
#dis = edm.Normal(loc=0.0, scale=1.0)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
for i in range(100000):
    print(sess.run(W_1.sample()))
#samples = W_1.sample(900000)
# 通过log_device_placement参数来输出运行每一个运算的设备。
print sess.run(samples)
#print(samples.eval(session=sess))
