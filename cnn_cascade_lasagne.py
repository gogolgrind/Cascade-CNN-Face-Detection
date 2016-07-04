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
#        
#        network = layers.InputLayer((None, 3, 12, 12), input_var=self.__input_var__)
#        network = layers.Conv2DLayer(network,num_filters=16,filter_size=(3,3),stride=1,nonlinearity=relu)    
#        network = layers.MaxPool2DLayer(network, pool_size = (3,3),stride = 2)    
#        network = layers.Conv2DLayer(network,num_filters=16,filter_size=(4,4),stride=1,nonlinearity=relu)    
#        network = layers.Conv2DLayer(network,num_filters=2,filter_size=(1,1),stride=1,nonlinearity=relu)    
#        network = layers.DenseLayer(network,num_units = 2, nonlinearity = softmax)


        network = layers.InputLayer((None, 3, 12, 12), input_var=self.__input_var__)
        network = layers.Conv2DLayer(network,num_filters=16,filter_size=(3,3),stride=1,nonlinearity=relu)
        network = layers.MaxPool2DLayer(network, pool_size = (3,3),stride = 2)
        network = layers.DropoutLayer(network)        
        network = layers.DenseLayer(network,num_units = 16,nonlinearity = relu)
        network = layers.DenseLayer(network,num_units = 2, nonlinearity = softmax)
        return network
    
    def __build_24_net__(self):
        model12 = self.subnet
        network = layers.InputLayer((None, 3, 24, 24), input_var=self.__input_var__)
        network = layers.Conv2DLayer(network,num_filters=16,filter_size=(5,5),stride=1,nonlinearity=relu)
        network = layers.MaxPool2DLayer(network, pool_size = (3,3),stride = 2)
        network = layers.DropoutLayer(network)
        network = layers.DenseLayer(network,num_units = 128,nonlinearity = relu)
        denselayer12 = model12.net.input_layer  
        network = layers.ConcatLayer([network, denselayer12]) 
        network = layers.DenseLayer(network,num_units = 2, nonlinearity = softmax)
        return network
    
    def __build_48_net__(self):
       
        model24 = self.subnet
        network = layers.InputLayer((None, 3, 48, 48), input_var=self.__input_var__)
        network = layers.Conv2DLayer(network,num_filters=64,filter_size=(5,5),stride=1,nonlinearity=relu)
        network = layers.batch_norm(layers.MaxPool2DLayer(network, pool_size = (3,3),stride = 2))
        network = layers.Conv2DLayer(network,num_filters=64,filter_size=(5,5),stride=1,nonlinearity=relu)
        network = layers.BatchNormLayer(network)
        network = layers.MaxPool2DLayer(network, pool_size = (3,3),stride = 2)
        network = layers.DenseLayer(network,num_units = 256,nonlinearity = relu)
        #network = layers.Conv2DLayer(network,num_filters=256,filter_size=(1,1),stride=1,nonlinearity=relu)
        denselayer24 = model24.net.input_layer
        network = layers.ConcatLayer([network, denselayer24])  
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
        loss = loss.mean() + 1e-4 * regularization.regularize_network_params(
                self.net, regularization.l2)
        
        # create parameter update expressions
        params = layers.get_all_params(self.net, trainable=True)
        self.eta = theano.shared(sp.array(sp.float32(0.05), dtype=sp.float32))
        update_rule = updates.nesterov_momentum(loss, params, learning_rate=self.eta,
                                                    momentum=0.9)
        
        # compile training function that updates parameters and returns training loss
        if  self.nn_name == '24-net':
            self.__train_fn__ = theano.function([self.__input_var__,self.subnet.__input_var__, self.__target_var__], loss, updates=update_rule)
            self.__predict_fn__ = theano.function([self.__input_var__,self.subnet.__input_var__], layers.get_output(self.net,deterministic=True))
        elif self.nn_name == '48-net':
            self.__train_fn__ = theano.function([self.__input_var__,self.subnet.__input_var__,self.subnet.subnet.__input_var__,self.__target_var__], loss, updates=update_rule)
            self.__predict_fn__ = theano.function([self.__input_var__,self.subnet.__input_var__,self.subnet.subnet.__input_var__], layers.get_output(self.net,deterministic=True))
        else:
            self.__train_fn__ = theano.function([self.__input_var__,self.__target_var__], loss, updates=update_rule)
            self.__predict_fn__ = theano.function([self.__input_var__], layers.get_output(self.net,deterministic=True))
    
    def __init__(self,nn_name,batch_size=256,freeze=1,l_rates = sp.float32(0.05)*sp.ones(120,dtype=sp.float32),verbose = 1,subnet= None):
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
            
    def fit(self,X,y,X12 = None,X24 = None):
        X = X.astype(sp.float32)
        y = y.astype(sp.int32)
        if X12 != None:
            X12 = X12.astype(sp.float32)
        if X24 != None:
            X24 = X24.astype(sp.float32)
        if self.nn_name == '24-net' :
            X = [(u,v) for u,v  in zip(X,X12)]
        elif self.nn_name =='48-net':
            X = [(u,v,q) for u,v,q  in zip(X,X24,X12)]
        for epoch in range(self.max_epochs):
            self.eta.set_value(self.l_rates[epoch])
            loss = 0
            start = time()
            for input_batch, target in self.iterate_minibatches(X,y,self.batch_size):
                if self.nn_name == '24-net':
                    x = []
                    sx = []
                    for u,v in input_batch:
                        x.append(u)
                        sx.append(v)
                    x = sp.array(x)
                    sx = sp.array(sx)
                    loss += self.__train_fn__(x,sx,target)
                elif self.nn_name == '48-net':
                    x = []
                    sx = []
                    ssx = []
                    for u,v,q in input_batch:
                        x.append(u)
                        sx.append(v)
                        ssx.append(q)
                    x = sp.array(x)
                    sx = sp.array(sx)
                    ssx = sp.array(ssx)
                    loss += self.__train_fn__ (x,sx,ssx ,target)
                else:
                    loss += self.__train_fn__ (input_batch, target)
            if self.verbose:
                dur = time() - start
                print("epoch %d  out of %d \t loss %g \t time %d s" % (epoch + 1,self.max_epochs, loss / len(X),dur))
        
    def predict(self,X,X12 = None,X24 = None):
        proba = self.predict_proba(X=X,X24=X24,X12=X12)
        y_pred = sp.argmax(proba,axis=1)
        return sp.array(y_pred)
        
    def predict_proba(self,X,X12 = None,X24 = None):
        proba = []
        N = max(1,self.batch_size)
        if X12 != None:
            X12 = X12.astype(sp.float32)
        if X24 != None:
            X24 = X24.astype(sp.float32)
        if self.nn_name == '24-net' :
            X = [(u,v) for u,v  in zip(X,X12)]
        elif self.nn_name =='48-net':
            X = [(u,v,q) for u,v,q  in zip(X,X24,X12)]
        for x_chunk in [X[i:i + N] for i in range(0, len(X), N)]:
            if self.nn_name == '24-net':
                x = []
                sx = []
                for u,v in x_chunk:
                    x.append(u)
                    sx.append(v)
                x = sp.array(x)
                sx = sp.array(sx)
                chunk_proba = self.__predict_fn__(x,sx)
            elif self.nn_name == '48-net':
                x = []
                sx = []
                ssx = []
                for u,v,q in x_chunk:
                    x.append(u)
                    sx.append(v)
                    ssx.append(q)
                x = sp.array(x)
                sx = sp.array(sx)
                ssx = sp.array(ssx)
                chunk_proba = self.__predict_fn__(x,sx,ssx)
            else:
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