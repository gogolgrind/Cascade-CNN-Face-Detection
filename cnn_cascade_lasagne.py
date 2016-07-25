# -*- coding: utf-8 -*-
"""
Created on Tue May 31 20:20:22 2016

@author: Kostya
"""
from lasagne.nonlinearities import softmax, rectify as relu 
from lasagne import layers
from lasagne import updates
from lasagne import regularization
from lasagne import objectives
from time import time
from six.moves import cPickle as pickle
from util import Util as util
from sklearn.cross_validation import train_test_split

import theano
import theano.tensor as T
import scipy as sp

import sys
sys.setrecursionlimit(10000)

class Cnn(object):
    net = None
    subnet = None
    nn_name = ''
    l_rates = []
    max_epochs = 120
    batch_size = 256
    verbose = 0
    eta = None
    
    __train_fn__ = None
    # create classifcation nets
    def __build_12_net__(self):

        network = layers.InputLayer((None, 3, 12, 12), input_var=self.__input_var__)
        network = layers.dropout(network, p=0.1)
        network = layers.Conv2DLayer(network,num_filters=16,filter_size=(3,3),stride=1,nonlinearity=relu)
        network = layers.batch_norm(network)
        network = layers.MaxPool2DLayer(network, pool_size = (3,3),stride = 2)
        network = layers.DropoutLayer(network,p=0.3)        
        network = layers.DenseLayer(network,num_units = 16,nonlinearity = relu)
        network = layers.batch_norm(network)
        network = layers.DropoutLayer(network,p=0.3)
        network = layers.DenseLayer(network,num_units = 2, nonlinearity = softmax)
        return network
    
    def __build_24_net__(self):
       
        network = layers.InputLayer((None, 3, 24, 24), input_var=self.__input_var__)
        network = layers.dropout(network, p=0.1)
        network = layers.Conv2DLayer(network,num_filters=64,filter_size=(5,5),stride=1,nonlinearity=relu)
        network = layers.batch_norm(network)
        network = layers.MaxPool2DLayer(network, pool_size = (3,3),stride = 2)
        network = layers.DropoutLayer(network,p=0.5)
        network = layers.batch_norm(network)
        network = layers.DenseLayer(network,num_units = 64,nonlinearity = relu)
        network = layers.DropoutLayer(network,p=0.5)
        network = layers.DenseLayer(network,num_units = 2, nonlinearity = softmax)
        return network
    
    def __build_48_net__(self):
        network = layers.InputLayer((None, 3, 48, 48), input_var=self.__input_var__)
       
        network = layers.Conv2DLayer(network,num_filters=64,filter_size=(5,5),stride=1,nonlinearity=relu)
        network = layers.MaxPool2DLayer(network, pool_size = (3,3),stride = 2)        
        network = layers.batch_norm(network)

        network = layers.Conv2DLayer(network,num_filters=64,filter_size=(5,5),stride=1,nonlinearity=relu)
        network = layers.batch_norm(network)
        network = layers.MaxPool2DLayer(network, pool_size = (3,3),stride = 2)
        
        network = layers.Conv2DLayer(network,num_filters=64,filter_size=(3,3),stride=1,nonlinearity=relu)
        network = layers.batch_norm(network)
        network = layers.MaxPool2DLayer(network, pool_size = (3,3),stride = 2)
        
        network = layers.DenseLayer(network,num_units = 256,nonlinearity = relu)
        network = layers.DenseLayer(network,num_units = 2, nonlinearity = softmax)
        return network
        
    def __build_12_calib_net__(self):
        network = layers.InputLayer((None, 3, 12, 12), input_var=self.__input_var__)
        network = layers.Conv2DLayer(network,num_filters=16,filter_size=(3,3),stride=1,nonlinearity=relu)
        network = layers.MaxPool2DLayer(network, pool_size = (3,3),stride = 2)
        network = layers.DenseLayer(network,num_units = 128,nonlinearity = relu)
        network = layers.DenseLayer(network,num_units = 45, nonlinearity = softmax)
        return network
        
    def __build_24_calib_net__(self):
        network = layers.InputLayer((None, 3, 24, 24), input_var=self.__input_var__)
        network = layers.Conv2DLayer(network,num_filters=32,filter_size=(5,5),stride=1,nonlinearity=relu)
        network = layers.MaxPool2DLayer(network, pool_size = (3,3),stride = 2)
        network = layers.DenseLayer(network,num_units = 64,nonlinearity = relu)
        network = layers.DenseLayer(network,num_units = 45, nonlinearity = softmax)
        return network
        
    def __build_48_calib_net__(self):
        network = layers.InputLayer((None, 3, 48, 48), input_var=self.__input_var__)
        network = layers.Conv2DLayer(network,num_filters=64,filter_size=(5,5),stride=1,nonlinearity=relu)
        network = layers.batch_norm(layers.MaxPool2DLayer(network, pool_size = (3,3),stride = 2))
        network = layers.Conv2DLayer(network,num_filters=64,filter_size=(5,5),stride=1,nonlinearity=relu)
        network = layers.batch_norm(layers.MaxPool2DLayer(network, pool_size = (3,3),stride = 2))
        network = layers.DenseLayer(network,num_units = 256,nonlinearity = relu)
        network = layers.DenseLayer(network,num_units = 45, nonlinearity = softmax)
        return network
    
    def __build_loss_train__fn__(self):
        # create loss function
        prediction = layers.get_output(self.net)
        loss = objectives.categorical_crossentropy(prediction, self.__target_var__)
        loss = loss.mean() + 1e-4 * regularization.regularize_network_params(self.net, regularization.l2)
        
        val_acc = T.mean(T.eq(T.argmax(prediction, axis=1), self.__target_var__),dtype=theano.config.floatX)
        
        # create parameter update expressions
        params = layers.get_all_params(self.net, trainable=True)
        self.eta = theano.shared(sp.array(sp.float32(0.05), dtype=sp.float32))
        update_rule = updates.nesterov_momentum(loss, params, learning_rate=self.eta,
                                                    momentum=0.9)
        
        # compile training function that updates parameters and returns training loss
        self.__train_fn__ = theano.function([self.__input_var__,self.__target_var__], loss, updates=update_rule)
        self.__predict_fn__ = theano.function([self.__input_var__], layers.get_output(self.net,deterministic=True))
        self.__val_fn__ = theano.function([self.__input_var__,self.__target_var__], [loss,val_acc])
    
    def __init__(self,nn_name,batch_size=1024,freeze=1,l_rates = sp.float32(0.05)*sp.ones(512,dtype=sp.float32),verbose = 1,subnet= None):
        self.nn_name = nn_name
        self.subnet = subnet
        if subnet != None and freeze:
            self.subnet.__freeze__()
        self.batch_size = batch_size
        self.verbose = verbose
        self.l_rates = l_rates
        self.__input_var__ = T.tensor4('X'+self.nn_name[:2])
        self.__target_var__ = T.ivector('y+'+self.nn_name[:2])
        self.max_epochs = self.l_rates.shape[0]
        if self.nn_name == '12-net':
            self.net = self.__build_12_net__()
        elif self.nn_name == '24-net':
            self.net = self.__build_24_net__()
        elif self.nn_name == '48-net':
            self.net = self.__build_48_net__()
        elif self.nn_name =='12-calib_net':
            self.net = self.__build_12_calib_net__()
        elif self.nn_name =='24-calib_net':
            self.net = self.__build_24_calib_net__()
        elif self.nn_name =='48-calib_net':
            self.net = self.__build_48_calib_net__()
        self.__build_loss_train__fn__()
        
    def iterate_minibatches(self,X, y, batchsize, shuffle=False):
        assert len(X) == len(y)
        if shuffle:
            indices = sp.arange(len(X))
            sp.random.shuffle(indices)
        for start_idx in range(0, len(X) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield X[excerpt], y[excerpt]
    
    def __freeze__(self):
        for layer in layers.get_all_layers(self.net):
            for param in layer.params:
                layer.params[param].discard('trainable')
            
    def train_on_hdd(self,rootdir = '12-net/'):
        print(self.nn_name,'training  start...','data folder',rootdir)
        mean_acc = 0
        total_time = 0
        bpaths = util.get_files(rootdir = rootdir,fexpr = '*.npz')
        m = len(bpaths)
        r = len(util.load_from_npz(bpaths [-1]))
        total_len = m * len(util.load_from_npz(bpaths [0]))
        print('data input size is around',total_len)
        for epoch in range(self.max_epochs):
            self.eta.set_value(self.l_rates[epoch])
            t_loss = 0
            start = time()
            for bpath in bpaths:
                batch = util.load_from_npz(bpath)
                items,labels = batch[:,0],batch[:,1]
                items = sp.array([e.astype(sp.float32) for e in items])
                labels = labels.astype(sp.int32)
				
                X_train, X_val, y_train, y_val = train_test_split(items,labels,test_size = 0.25)
				
                t_loss += self.__train_fn__ (X_train,y_train)
                val_acc = 0
                val_batches = 0 
                for xval,yval  in self.iterate_minibatches(X_val,y_val,16):
                    err, acc = self.__val_fn__(xval, yval) 
                    val_acc += acc
                    val_batches += 1
					
            if self.verbose:
                dur = time() - start
                a0 = 100*(val_acc/val_batches)
                mean_acc += a0
                total_time += dur
                print("epoch %d out of %d \t loss %g \t  acсuracy  %g \t time %d s \t" % (epoch + 1,self.max_epochs, t_loss / (total_len),a0,dur))
        m = (total_time)//60
        s = total_time - 60 * m 
        h =  m//60
        m = m - 60 * h
        mean_acc = mean_acc / self.max_epochs
        print('Training  end with total time %d h %d m %d s and mean accouracy over epochs %g' % (h,m,s,mean_acc))
        
        
    def fit(self,X,y):
        X = X.astype(sp.float32)
        y = y.astype(sp.int32)
        total_time = 0
        mean_acc = 0
        print(self.nn_name,'training  start...')
        for epoch in range(self.max_epochs):
            self.eta.set_value(self.l_rates[epoch])
            t_loss = 0
            start = time()
            for input_batch, target in self.iterate_minibatches(X,y,self.batch_size):
                
                X_train, X_val, y_train, y_val = train_test_split(input_batch, target,test_size = 0.1)
				
                t_loss += self.__train_fn__ (X_train,y_train)
                val_acc = 0
                val_batches = 0 
                for xval,yval  in self.iterate_minibatches(X_val,y_val,16):
                    err, acc = self.__val_fn__(xval, yval) 
                    val_acc += acc
                    val_batches += 1
					
            if self.verbose:

                dur = time() - start
                a0 = 100*(val_acc/val_batches)
                mean_acc += a0
                total_time += dur
            
                print("epoch %d  out of %d \t loss %g \t  acсuracy  %g \t time %d s \t" % (epoch + 1,self.max_epochs, t_loss / (len(X)),100*(val_acc/val_batches),dur))
            
        m = (total_time)//60
        s = total_time - 60 * m 
        h =  m//60
        m = m - 60 * h
        mean_acc = mean_acc / self.max_epochs
        print('Training  end with total time %d h %d m %d s and mean accouracy over epochs %g' % (h,m,s,mean_acc))
        
    def predict(self,X):
        proba = self.predict_proba(X=X)
        y_pred = sp.argmax(proba,axis=1)
        return sp.array(y_pred)
        
    def predict_proba(self,X,X12 = None,X24 = None):
        proba = []
        N = max(1,self.batch_size)
        for x_chunk in [X[i:i + N] for i in range(0, len(X), N)]:
            chunk_proba = self.__predict_fn__(x_chunk)
            for p in chunk_proba:
                proba.append(p)
        return sp.array(proba)
    
    def __save_model_old__(self,model_name = nn_name+'.pickle'):
        with open(model_name, 'wb') as f:
            pickle.dump(self, f, -1)
            
    def __load_model_old__(self,model_name = nn_name+'.pickle'):
        with open(model_name, 'rb') as f:
            model = pickle.load(f)
            f.close()
        return model

    def save_model(self,model_name = nn_name+'.npz'):
        sp.savez(model_name, *layers.get_all_param_values(self.net))
            
    def load_model(self,model_name = nn_name+'.npz'):
        print(model_name,'is loaded')
        with sp.load(model_name) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            layers.set_all_param_values(self.net, param_values)
        return self            