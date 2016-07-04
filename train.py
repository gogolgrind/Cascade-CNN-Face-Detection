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


def load_from_npz(data_name):
    with sp.load(data_name) as f:
        values = [f['arr_%d' % i] for i in range(len(f.files))][0]
        return values


def train(nn_name = '12-net',k = 12):
    """
    Fucntion for traning 12-net with testing on part of data
    using cross validation
    """
    suff = str(k)
    if nn_name.find('calib') > 0:
        X_data_name = 'train_data_icalib_'+ suff +  '.npz'
        y_data_name = 'labels_icalib_'+ suff + '.npz'
    else:
        X_data_name = 'train_data_'+ suff +  '.npz'
        y_data_name = 'labels_'+ suff + '.npz'
    
    rates12 = sp.hstack((0.001 * sp.ones(70,dtype=sp.float32),0.0001*sp.ones(20,dtype=sp.float32),0.00001*sp.ones(5,dtype=sp.float32)))
    rates12_c =  sp.hstack((0.05 * sp.ones(25,dtype=sp.float32),0.005*sp.ones(15,dtype=sp.float32),0.0005*sp.ones(7,dtype=sp.float32)))
    rates24 = sp.hstack((0.01 * sp.ones(25,dtype=sp.float32),0.0001*sp.ones(15,dtype=sp.float32)))
    rates48 = sp.hstack ([0.05 * sp.ones(15,dtype=sp.float32),0.005*sp.ones(10,dtype=sp.float32) ])
    if nn_name == '24-net':
        nn = Cnnl(nn_name = nn_name,l_rates=rates24,subnet=Cnnl(nn_name = '12-net',l_rates=rates12).__load_model_old__(
            '12-net_lasagne_.pickle'))
    elif nn_name == '48-net':    
        nn = Cnnl(nn_name = nn_name,l_rates=rates48,subnet=Cnnl(nn_name = '24-net',l_rates=rates24,
                                                                subnet=Cnnl(nn_name = '12-net',l_rates=rates12).__load_model_old__(
            '12-net_lasagne_.pickle')).__load_model_old__('24-net_lasagne_.pickle'))     
    else:
        if nn_name.find('calib') > 0:
            nn = Cnnl(nn_name = nn_name,l_rates=rates12_c)
        else:
            nn = Cnnl(nn_name = nn_name,l_rates=rates12)
    if not os.path.exists(nn_name   + '_lasagne_.npz') or not os.path.exists(nn_name   + '_lasagne_.pickle'): 
        if nn_name.find('calib') > 0:
            ds.get_train_wider_calib_data(k=k)  
        else:
            ds.get_train_data(k=k)
            

    X_train = load_from_npz(X_data_name)
    y_train = load_from_npz(y_data_name)
    print("Done!\n","Positive examples count, Negative exapmples count:",len(y_train[y_train==1]),len(y_train[y_train==0]))
    
    
    if not os.path.exists(nn_name   + '_lasagne_.npz') or not os.path.exists(nn_name   + '_lasagne_.pickle'):
        if nn_name == '24-net':
            X_sub_train12 = load_from_npz('train_data_12.npz')
            nn.fit(X = X_train,y = y_train,X12 = X_sub_train12)
        elif nn_name == '48-net':
            X_sub_train12 = load_from_npz('train_data_12.npz')
            X_sub_train24 = load_from_npz('train_data_24.npz')
            nn.fit(X = X_train,y = y_train,X12 = X_sub_train12,X24 = X_sub_train24)
        else:
            nn.fit(X = X_train,y = y_train)
        nn.__save_model_old__(nn_name   + '_lasagne_.pickle')
        
        
def main(): 
    train(nn_name='12-net',k=12)
    #train(nn_name='24-net',k=24)
    #train(nn_name='48-net',k=48)
    #train(nn_name='12-calib_net',k=12)
    #train(nn_name='24-calib_net',k=24)
    #train(nn_name='48-calib_net',k=48)
if __name__ == '__main__':
    main()