#-*- coding:utf8 -*-

import os, sys, pickle, math, warnings, time
import numpy as np
import pandas as pd
#from scipy.stats import multivariate_normal
from copy import deepcopy
from multiprocessing import Pool
#from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
#import tensorflow as tf
#import pycuda.autoinit
#import pycuda.driver as drv
#from pycuda.compiler import SourceModule

warnings.simplefilter('ignore')

# 原始数据
#features = ['fixed acidity','volatile acidity','residual sugar','chlorides','free sulfur dioxide','density','pH','sulphates']
features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
df = pd.read_csv('winequality-white.csv', sep=';')
X = df[features].values
y = (df.quality >= 7).values


# 标准化处理
scaler = StandardScaler()
X = scaler.fit_transform(X)
#X = X.astype(np.float32)
#y = y.astype(np.float32)

#print( 'X.shape: ', X.shape)
#print( 'y.shape: ', y.shape)
# SMC参数
J = 100
B = 10**4
K = 16


# 交叉验证调参
'''
model = GridSearchCV(
    estimator = MLPClassifier(activation='relu', alpha=0.0003, solver='adam'),
    param_grid = {
        'hidden_layer_sizes': [(7, h2) for h2 in range(10,21,2)],
    },
    n_jobs = 30,
    cv = 6,
    refit = True,
    scoring = 'roc_auc',
)
model.fit(X, y)
print (model.best_score_, model.best_params_)
'''

# 超参数
'''
layer_size = [X.shape[1]] + list(model.best_params_['hidden_layer_sizes']) + [1]

if model.best_params_['activation']=='logistic':
    act_func = lambda x: 1 / (1 + np.exp(-x))
elif model.best_params_['activation']=='tanh':
    act_func = np.tanh
else:
    act_func = lambda x: (np.abs(x) + x) / 2

l2_norm = model.best_params_['alpha']
'''

layer_size = [11, 22, 1]
#act_func = lambda x: (np.abs(x)+x)/2 #ReLU activation func
act_func = lambda x: 1.0/(1+np.exp(-x))
l2_norm = 0.0001



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


def Ln(theta):
    # 预测
#    st = time.time()
    pred = X
    for w,b in zip(theta.w[:-1],theta.b[:-1]):
        pred = act_func(np.matmul(pred, w) + b)
    pred = 1 / (1 + np.exp(-np.matmul(pred, theta.w[-1]) - theta.b[-1]))
    pred = pred.reshape([-1])
    pred[pred==0] = 1e-6
    pred[pred==1] = 1 - 1e-6
#    print('ln time:', time.time() - st)
    return np.mean(np.log(pred)*y + np.log(1-pred)*(1-y))

def MH(theta0, phi_j):
    def log_prior_ratio(theta1, theta2):
        t1,t2 = 0,0
        np_sum = np.sum
        for w in theta1.w:
            t1 += np_sum(w*w)
        for b in theta1.b:
            t1 += np_sum(b*b)
        for w in theta2.w:
            t2 += np_sum(w*w)
        for b in theta2.b:
            t2 += np_sum(b*b)
        return (t1 - t2) * l2_norm
    
    def log_prob(theta):
        p1 = X
        for w,b in zip(theta.w[:-1],theta.b[:-1]):
            p1 = act_func(np.matmul(p1, w) + b)
        p1 = 1 / (1 + np.exp(-np.matmul(p1, theta.w[-1]) - theta.b[-1]))
        p1 = p1.reshape([-1])
        return np.sum(np.log(np.abs(p1 - 1 + y)))
    
    #start = time.time()
    # 每次决定转移与否的均匀分布随机变量
    thresholds = np.log(np.random.uniform(0, 1, size=K))
    # random walk
    theta = deepcopy(theta0)
    for i in range(K):
        _theta = deepcopy(theta)
        _theta.random_walk()
        
        if phi_j * (log_prob(_theta) - log_prob(theta)) + log_prior_ratio(_theta,theta) > thresholds[i]:
            theta = _theta
    return theta#, time.time() - start


def draw_from_multinomial(support, weight):
    # weight归一化
    weight /= np.sum(weight)
    # weight累加
    t = 0
    cum = [0]
    for w in weight:
        t += w
        cum.append(t)
    cum[-1] = 1.1
    # 抽样
    n = len(weight)
    U = np.random.uniform(0, 1, n)
    result = []
    for u in U:
        low = 0
        high = len(cum) - 1
        while high-low>1:
            mid = int((high + low) / 2)
            if cum[mid] < u:
                low = mid
            else:
                high = mid
        result.append(support[low])
    return np.array(result)


def SMC(samples):
    # 数据样本总条数
    n = len(X)
    w = np.ones(shape=B)
    phi = np.linspace(0, 1, J)**2
    #samples = deepcopy(samples)
    # draw form prior distribution
    for j in range(1,J):
        start = time.time()
        #v = np.array([np.exp((phi[j]-phi[j-1]) * n * Ln(samples[b])) for b in range(B)]) #for loop -> lambda
        #v = np.exp((phi[j]-phi[j-1]) * n * np.array(list(map(lambda x:Ln(x), samples))))
        pool = Pool(32)
        pool_apply_async = pool.apply_async
        vs = []
        vs_append = vs.append
        for theta0 in samples:
            vs_append(pool_apply_async(Ln, (theta0,)))
        pool.close()
        pool.join()
        v = []
        v_append = v.append
        for r in vs:
            v_append(r.get())
        v = np.exp((phi[j]-phi[j-1]) * n * np.array(v))
        print('v time:', time.time()-start)
        
        #----------------
        startphi = time.time()
        w = B * v * w / np.sum(v*w)
        ESS = B**2 / np.sum(w**2)
        #startphi = time.time()
        if ESS > 0.5*B:
            varphi = deepcopy(samples)  #此处赋值即可，因为之后samples会新建
            print('phi time(not sample):', time.time() - startphi)
        else:
            varphi = draw_from_multinomial(samples, w)
            w = np.ones(shape=B)
            print('phi time(do sample):', time.time() - startphi)
        
        #------------------
        startpool = time.time()
        pool = Pool(32)
        pool_apply_async = pool.apply_async
        results = []
        results_append = results.append
        #samples = []
        for theta0 in varphi:
            results_append(pool_apply_async(MH, (theta0,phi[j],)))
            #samples.append(pool.apply_async(MH, (theta0,)).get())
        pool.close()
        pool.join()
        #startres = time.time()
        samples = []
        samples_append = samples.append
        times = []
        times_append = times.append
        #samples = list(map(lambda r:r.get(), results))
        for r in results:
            #s,t = r.get()
            samples_append(r.get())
            #times_append(t)
        #print('startres: ', time.time() - startres)
        print('pooltime:',time.time()-startpool)
        #print ('轮数：',j, '总时间：', time.time()-start, '进程平均时间：', np.mean(times), '标准差：', np.std(times))
        print ('轮数：',j, '总时间：', time.time()-start)
        sys.stdout.flush()
        #for r in results:
        #    samples_append(r.get())
        stw = time.time()
        with open('./smc2/'+str(j),'wb') as f:
            pickle.dump(samples, f)
        print('write time: ', time.time() - stw)
        print()
    return samples



samples = []
for i in range(B):
    samples.append(Theta())

samples = SMC(samples)

'''import profile
profile.run("SMC(samples)", "prof.txt")
import pstats
p = pstats.Stats("prof.txt")
p.sort_stats("time").print_stats()
'''
'''
with open('wine_samples','wb') as f:
    pickle.dump(samples, f)
'''

