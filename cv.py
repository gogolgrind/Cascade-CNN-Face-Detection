# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:56:17 2016

@author: Kostya
"""
from util import Util as util
import scipy as sp
from sklearn.metrics import accuracy_score, f1_score
from cnn_cascade_lasagne import Cnn as Cnnl
import seaborn as sns
from scipy import interpolate

def cv(nn_name,d_num = 10000,k_fold = 7,score_metrics = 'accuracy',verbose = 0):
    suff = str(nn_name[:2])
    if nn_name.find('calib') > 0:
        X_data_name = 'train_data_icalib_'+ suff +  '.npy'
        y_data_name = 'labels_icalib_'+ suff + '.npy'
    else:
        X_data_name = 'train_data_'+ suff +  '.npy'
        y_data_name = 'labels_'+ suff + '.npy'
    X,y = sp.load(X_data_name),sp.load(y_data_name)
    d_num = min(len(X),d_num)        
    X = X[:d_num]
    y = y[:d_num] 
    rates12 = sp.hstack((0.05 * sp.ones(25,dtype=sp.float32),0.005*sp.ones(15,dtype=sp.float32),0.0005*sp.ones(10,dtype=sp.float32)))
    rates24 = sp.hstack((0.01 * sp.ones(25,dtype=sp.float32),0.0001*sp.ones(15,dtype=sp.float32)))
    rates48 = sp.hstack ([0.05 * sp.ones(15,dtype=sp.float32),0.005*sp.ones(10,dtype=sp.float32) ])
    if nn_name == '48-net':
        X12 = sp.load('train_data_12.npy')[:d_num]
        X24 = sp.load('train_data_24.npy')[:d_num]
    elif nn_name == '24-net':
        X12 = sp.load('train_data_12.npy')[:d_num]
        
    if score_metrics == 'accuracy':
        score_fn = accuracy_score
    else:
        score_fn = f1_score 
    scores = []
    iteration = 0
    for t_indx,v_indx in util.kfold(X,y,k_fold=k_fold):
        nn = None
        X_train,X_test,y_train,y_test = X[t_indx], X[v_indx], y[t_indx], y[v_indx]
        
        #print('\t \t',str(iteration+1),'fold out of ',str(k_fold),'\t \t' )
        if nn_name == '24-net':
            nn = Cnnl(nn_name = nn_name,l_rates=rates24,subnet=Cnnl(nn_name = '12-net',l_rates=rates12).load_model(
            '12-net_lasagne_.pickle'))
            nn.fit(X = X_train,y = y_train,X12 = X12[t_indx])
        elif nn_name == '48-net':
            nn = Cnnl(nn_name = nn_name,l_rates=rates48,subnet=Cnnl(nn_name = '24-net',l_rates=rates24,subnet=Cnnl(nn_name = '12-net',l_rates=rates12).load_model(
            '12-net_lasagne_.pickle')).load_model('24-net_lasagne_.pickle'))
            nn.fit(X = X_train,y = y_train,X12 = X12[t_indx],X24 = X24[t_indx])
        else:
            
            nn = Cnnl(nn_name = nn_name,l_rates=rates12,verbose=verbose)
            nn.fit(X = X_train,y = y_train)
    
        if nn_name == '24-net':  
            y_pred = nn.predict(X_test,X12=X12[v_indx])
        elif nn_name == '48-net':
            y_pred = nn.predict(X_test,X12=X12[v_indx],X24=X24[v_indx])
        else:
            y_pred = nn.predict(X_test)
        score = score_fn(y_test,y_pred)
        
        #print(iteration,'fold score',score)
        scores.append(score)
        iteration += 1
    score_mean = sp.array(scores).mean()
    print(d_num,'mean score',score)
    return score_mean

def main():
    s = []
    num = []
    for d in [10 ** i for i in range(1,6)]:
        score_mean = cv('12-net',d_num=d)
        num.append(sp.log10(d))
        s.append(score_mean)
        pass
    sns.plt.title('12-net  accuracy lerning curve')
    sns.plt.xlabel('num samples on 7 fold cv')
    sns.plt.ylabel('mean accuracy')
    sns.plt.xlim(1,5)
    sns.plt.ylim(0,1.0)
    sns.plt.plot(num,s)
    sns.plt.plot(num,s,'ro')
    sns.plt.show()
if __name__ == '__main__':
    main()