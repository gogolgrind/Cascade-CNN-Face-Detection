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
import sys
from cnn_cascade_lasagne import Cnn as Cnnl
from util import Util as util


def train(nn_name = '12-net',rootdir = 'F:\\'):
    """
    Fucntion for traning 12-net with testing on part of data
    using cross validation
    """
    is_calib = nn_name.find('calib') > 0
    
    X_name = 'train_data_'+('icalib_' if is_calib else '' )+ nn_name[:2] + '.npz'
    y_name = 'labels_'+('icalib_' if is_calib else '' )+ nn_name[:2] + '.npz'
    is_calib = nn_name.find('calib') > 0
    rates = sp.hstack((0.05 * sp.ones(333,dtype=sp.float32),0.005*sp.ones(333,dtype=sp.float32),0.0005*sp.ones(333,dtype=sp.float32),
                         0.00005*sp.ones(5,dtype=sp.float32)))
    rates12 = rates
    rates24 = rates
    rates48 = rates
    rates12_c =  rates12[333:666]
    rates48_c = rates12_c
    rates24_c = rates12_c
    if nn_name == '24-net':
        if is_calib:
            nn = Cnnl(nn_name = nn_name,l_rates=rates24_c)
        else:
            nn = Cnnl(nn_name = nn_name,l_rates=rates24)
    elif nn_name == '48-net':    
        if is_calib:
            nn = Cnnl(nn_name = nn_name,l_rates=rates48_c)
        else:
            nn = Cnnl(nn_name = nn_name,l_rates=rates48)
    else:
        if is_calib:
            nn = Cnnl(nn_name = nn_name,l_rates=rates12_c)
        else:
            nn = Cnnl(nn_name = nn_name,l_rates = rates12)
    
    if is_calib:
        X = util.load_from_npz(X_name)
        y = util.load_from_npz(y_name)
        nn.fit(X,y)
    else:
        rpath = os.path.join(rootdir,'traindata',nn.nn_name)
        nn.train_on_hdd(rootdir = rpath)
    nn.__save_model_old__(nn_name   + '_lasagne_.pickle')
        
        
def main(): 
    nn_name = sys.argv[1]
    train(nn_name=nn_name)
if __name__ == '__main__':
    main()