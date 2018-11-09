#-*- coding:utf8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings, pickle, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt

warnings.simplefilter('ignore')

def build_toy_dataset():
    features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
    df = pd.read_csv('../winequality-white.csv', sep=';')
    X = df[features].values
    y = (df.quality >= 7).values.astype(np.float32)

    # 标准化处理
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    return X, y

def np_neural_network(X, W_0, W_1, b_0, b_1):
    h = np.tanh(np.matmul(X, W_0) + b_0)
    h = np.matmul(h, W_1) + b_1
    return np.reshape(h, [-1])

def unpack_thete_get_outputs(X, theta):
    W_0 = np.reshape(theta[0:11*14], [11, 14])
    W_1 = np.reshape(theta[11*14: 11*14+14*1], [14, 1])
    b_0 = np.reshape(theta[11*14+14*1:11*14+14*1+14], [14])
    b_1 = np.reshape(theta[11*14+14*1+14:], [1])
    return np_neural_network(X, W_0, W_1, b_0, b_1)

alls = time.time()
J = 10000
B = 5000
#burnin = 500
X_train, y_train = build_toy_dataset()

with open('../smc2/theta'+'-'+str(J)+'-'+str(B),'rb') as f:
    outputs = np.array(pickle.load(f))
sp = []
for theta in outputs:
    sp.append(roc_auc_score(y_train,unpack_thete_get_outputs(X_train, theta)))
plt.hist(sp, bins=100)
plt.savefig('../aly_out/qauc'+'-'+str(J)+'-'+str(B)+'.png')
print('Elapse: ', time.time() - alls)
print('Please change to dir ../aly_out/ to get the result.')
