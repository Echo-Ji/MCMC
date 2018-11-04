#-*- coding:utf8 -*-

import os, sys, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import itertools
from copy import deepcopy
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

import matplotlib
matplotlib.use('agg')


import matplotlib.pyplot as plt


features = ['fixed acidity','volatile acidity','residual sugar','chlorides','free sulfur dioxide','density','pH','sulphates']
df = pd.read_csv('winequality-white.csv', sep=';')
X = df[features].values
y = (df.quality >= 7).values

layer_size = [8, 15, 10, 1]
act_func = np.tanh

# 置信度
# alpha = 0.9

class Theta(object):
    def __init__(self):
        self.w = []
        self.b = []
        for s1,s2 in zip(layer_size[:-1], layer_size[1:]):
            self.w.append(np.random.normal(0, 10, [s1,s2]))
            self.b.append(np.random.normal(0, 10, [s2]))
    
    def random_walk(self):
        i = 0
        for s1,s2 in zip(layer_size[:-1], layer_size[1:]):
            self.w[i] += np.random.normal(0, 1, [s1,s2])
            self.b[i] += np.random.normal(0, 1, [s2])
            i += 1


def predict(X, theta):
    pred = X
    for w,b in zip(theta.w[:-1],theta.b[:-1]):
        pred = act_func(np.matmul(pred, w) + b)
    pred = 1 / (1 + np.exp(-np.matmul(pred, theta.w[-1]) - theta.b[-1]))
    return pred.reshape([-1])


def Ln(theta):
    pred = predict(X,theta)
    pred[pred==0] = 1e-6
    pred[pred==1] = 1 - 1e-6
    return np.mean(np.log(pred)*y + np.log(1-pred)*(1-y))


def gradient(theta):
    result = []
    for i in range(len(features)):
        X1 = deepcopy(X)
        X1[:,i] += 1e-6
        X2 = deepcopy(X)
        X2[:,i] -= 1e-6
        pred1 = predict(X1, theta)
        pred2 = predict(X2, theta)
        g = (pred1 - pred2) / 2e-6    # 梯度
        result.append(np.mean(g))
    return result


with open('./wine_theta_samples','rb') as f:
    samples = np.array(pickle.load(f))

'''
# 求alpha%置信区间
likelihood = []
for theta in samples:
    likelihood.append(Ln(theta))
sorted_index = np.argsort(likelihood)
samples = samples[sorted_index[int(-alpha*len(samples)):]]
'''

# AUC
s = []
for theta in samples:
    pred = predict(X, theta)
    # s.append(accuracy_score(y, pred>=0.5))
    s.append(roc_auc_score(y, pred))

print ('AUC mean:', np.mean(s), 'AUC std:', np.std(s))
print ('max:', np.max(s), 'min:', np.min(s))

plt.hist(s, bins=100)
plt.savefig('auc.png')
plt.close()


# 计算每个样本点位置的偏导数（对于每个参数）
G = []
pool = Pool(30)
results = []
for theta in samples:
    results.append(pool.apply_async(gradient, (theta,)))
pool.close()
for r in results:
    G.append(r.get())
G = np.array(G)

IF = IsolationForest(n_estimators=20, n_jobs=30)
IF.fit(G)
result = IF.predict(G)
G = G[np.where(result>0)]

print ('最终剩余样本量', len(G))

# 均值
for i in range(len(features)):
    plt.title(features[i])
    plt.boxplot(G[:,i])
    plt.savefig('./box/' + features[i] + '.png')
    plt.close()
    plt.title(features[i])
    plt.hist(G[:,i])
    plt.savefig('./hist/' + features[i] + '.png')
    plt.close()

# 散点图
for i in range(len(features)):
    for j in range(i, len(features)):
        plt.title(str(np.cov(np.stack((G[:,i], G[:,j]), axis=0))[0,1]))
        plt.scatter(G[:,i], G[:,j])
        plt.xlabel(features[i])
        plt.ylabel(features[j])
        plt.savefig('./cov/' + features[i] + '-' + features[j] + '.png')
        plt.close()
