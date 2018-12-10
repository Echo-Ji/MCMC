#-*- coding:utf8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings, pickle, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from matplotlib import pyplot as plt

warnings.simplefilter('ignore')

def build_toy_dataset():
    features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
    df = pd.read_csv('./test-winequality-white.csv', sep=';')
    X = df[features].values
    y = (df.quality >= 7).values.astype(np.float32)

    # 标准化处理
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    return X, y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def np_neural_network(X, W_0, W_1, W_2, b_0, b_1, b_2):
    h = np.tanh(np.matmul(X, W_0) + b_0)
    h = np.tanh(np.matmul(h, W_1) + b_1)
    h = sigmoid(np.matmul(h, W_2) + b_2)
    return np.reshape(h, [-1])

def unpack_thete_get_outputs(X, theta):
    W_0 = np.reshape(theta[0:11*7], [11, 7])
    W_1 = np.reshape(theta[11*7: 11*7+7*10], [7, 10])
    W_2 = np.reshape(theta[11*7+7*10: 11*7+7*10+10*1], [10, 1])
    bs = 11*7+7*10+10*1
    b_0 = np.reshape(theta[bs:bs+7], [7])
    b_1 = np.reshape(theta[bs+7:bs+7+10], [10])
    b_2 = np.reshape(theta[bs+7+10:], [1])
    return np_neural_network(X, W_0, W_1, W_2, b_0, b_1, b_2)

def plot_hist(data, ax, bins=50, title=''):
    ax.hist(data, bins=bins)
    ax.set_title(title+': '+str("%.3f" % np.mean(data))+','+str("%.3f" % np.std(data)))
    return

alls = time.time()
J = 10000
B = 5000
threshold = 0.5
#burnin = 500
print('Reading test data...')
X_test, y_test = build_toy_dataset()
print('Reading sampled weights...')
with open('../smc2/fastbatch-theta'+'-'+str(J)+'-'+str(B),'rb') as f:
    outputs = np.array(pickle.load(f))
aucs = []
accs = []
f1s = []
print('Calculating results...')
for theta in outputs:
    y_pred = unpack_thete_get_outputs(X_test, theta)
    aucs.append(roc_auc_score(y_test, y_pred))
    y_pred[y_pred>=threshold] = 1
    y_pred[y_pred<threshold] = 0
    accs.append(accuracy_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))
#for theta in outputs:
#    aucs.append(roc_auc_score(y_test,unpack_thete_get_outputs(X_test, theta)))
#    accs.append(accuracy_score(y_test,unpack_thete_get_outputs(X_test, theta),normalize=True))
'''plt.hist(aucs, bins=100)
plt.title('auc: '+str("%.3f" % np.mean(aucs))+','+str("%.3f" % np.std(aucs)))
plt.savefig('../aly_out/qauc'+'-'+str(J)+'-'+str(B)+'.png')
plt.close()
plt.hist(accs, bins=50)
plt.title('acc: '+str("%.3f" % np.mean(accs))+','+str("%.3f" % np.std(accs)))
plt.savefig('../aly_out/qacc'+'-'+str(J)+'-'+str(B)+'.png')
plt.close()
plt.hist(f1s, bins=50)
plt.title('f1: '+str("%.3f" % np.mean(f1s))+','+str("%.3f" % np.std(f1s)))
plt.savefig('../aly_out/qf1'+'-'+str(J)+'-'+str(B)+'.png')
plt.close()
'''

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 3))
plot_hist(aucs, axes[0], 100, 'auc')
plot_hist(accs, axes[1], 50, 'acc')
plot_hist(f1s, axes[2], 50, 'f1')
plt.savefig('../aly_out/qeval'+'-'+str(J)+'-'+str(B)+'.png')
plt.close()
print('Elapse: ', time.time() - alls)
print('Please change to dir ../aly_out/ to get the result.')
