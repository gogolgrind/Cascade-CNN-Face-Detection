# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:22:35 2016

@author: Kostya
"""

from cnn_cascade_lasagne import Cnn as Cnnl
import scipy as sp

def pickle2npz():
    nn12 = Cnnl('12-net').__load_model_old__('12-net_lasagne_.pickle')
    nn12.save_model('12-net_lasagne_.npz')
    
#    nn_calib12 =  Cnnl('12-calib_net').__load_model_old__('12-calib_net_lasagne_.pickle')
#    nn_calib12.save_model('12-calib_net_lasagne_.npz')
#    
#    nn24 = Cnnl(nn_name = '24-net',subnet=nn12).__load_model_old__('24-net_lasagne_.pickle')
#    nn_calib24 =  Cnnl('24-calib_net').__load_model_old__('24-calib_net_lasagne_.pickle')
#    nn48 = Cnnl(nn_name = '48-net',subnet=nn24).__load_model_old__('48-net_lasagne_.pickle')
#    nn_calib48 =  Cnnl('48-calib_net').__load_model_old__('48-calib_net_lasagne_.pickle')
#    
#    
#    nn24.save_model('24-net_lasagne_.npz')
#    nn48.save_model('48-net_lasagne_.npz')
#
#    nn_calib24.save_model('24-calib_net_lasagne_.npz')
#    nn_calib48.save_model('48-calib_net_lasagne_.npz')

def mat2npz():
    struct = sp.io.loadmat('f12net.mat')
    w = []
    for indx,item in enumerate(struct['layers'][0]):
        layer = item[0][0]
        l_type = layer[0]
        if l_type == 'relu' or l_type == 'pool':
            continue
        weights_cell = layer[1]
        weights = weights_cell[0][0].transpose()
       
        bias = weights_cell[0][1][0]
        print(weights.shape)
        print(bias.shape)
        
        
def npy2npz():
     sp.savez('train_data_12.npz',sp.load('train_data_12.npy'))
     
     sp.savez('labels_12.npz',sp.load('labels_12.npy'))

def main():
    #npy2npz()

    pickle2npz()
    
if __name__ == '__main__':
    main()