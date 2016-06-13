# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:53:50 2016
@author: Kostya S. , k.sozykin@innopolis.ru, gogolgrind@gmail.com,
                    gogolgrind@yandex.ru
This is an implementation of the algorithm described in the following paper:
    http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf
"""
import scipy as sp
from datasets import Datasets as ds
import os
from cnn_cascade_lasagne import Cnn as Cnnl


def train(nn_name = '12-net',k = 12):
    """
    Fucntion for traning 12-net with testing on part of data
    using cross validation
    """
    suff = str(k)
    if nn_name.find('calib') > 0:
        X_data_name = 'train_data_icalib_'+ suff +  '.npy'
        y_data_name = 'labels_icalib_'+ suff + '.npy'
    else:
        X_data_name = 'train_data_'+ suff +  '.npy'
        y_data_name = 'labels_'+ suff + '.npy'
    
    rates12 = sp.hstack((0.05 * sp.ones(25,dtype=sp.float32),0.005*sp.ones(15,dtype=sp.float32),0.0005*sp.ones(10,dtype=sp.float32)))
    rates24 = sp.hstack((0.01 * sp.ones(25,dtype=sp.float32),0.0001*sp.ones(15,dtype=sp.float32)))
    rates48 = sp.hstack ([0.05 * sp.ones(15,dtype=sp.float32),0.005*sp.ones(10,dtype=sp.float32) ])
    if nn_name == '24-net':
        nn = Cnnl(nn_name = nn_name,l_rates=rates24,subnet=Cnnl(nn_name = '12-net',l_rates=rates12).load_model(
            '12-net_lasagne_.pickle'))
    elif nn_name == '48-net':    
        nn = Cnnl(nn_name = nn_name,l_rates=rates48,subnet=Cnnl(nn_name = '24-net',l_rates=rates24,subnet=Cnnl(nn_name = '12-net',l_rates=rates12).load_model(
            '12-net_lasagne_.pickle')).load_model('24-net_lasagne_.pickle'))     
    else:
        nn = Cnnl(nn_name = nn_name,l_rates=rates12)
    if not os.path.exists(nn_name   + '_lasagne_.pickle'): 
        if nn_name.find('calib') > 0:
            ds.get_train_wider_calib_data(k=k)  
        else:
            ds.get_train_data(k=k)
    X,y = sp.load(X_data_name),sp.load(y_data_name)
        
    X_train,y_train = X,y
    
    if not os.path.exists(nn_name   + '_lasagne_.pickle'):
        if nn_name == '24-net':
            X_sub_train12 = sp.load('train_data_12.npy')
            nn.fit(X = X_train,y = y_train,X12 = X_sub_train12)
        elif nn_name == '48-net':
            X_sub_train12 = sp.load('train_data_12.npy')
            X_sub_train24 = sp.load('train_data_24.npy')
            nn.fit(X = X_train,y = y_train,X12 = X_sub_train12,X24 = X_sub_train24)
        else:
            nn.fit(X = X_train,y = y_train)
        nn.save_model(nn_name   + '_lasagne_.pickle')
        
        
def main(): 
    train(nn_name='12-net',k=12)
    #train(nn_name='24-net',k=24)
    #train(nn_name='48-net',k=48)
    #train(nn_name='12-calib_net',k=12)
    #train(nn_name='24-calib_net',k=24)
    #train(nn_name='48-calib_net',k=48)
if __name__ == '__main__':
    main()