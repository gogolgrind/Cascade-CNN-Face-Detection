# -*- coding: utf-8 -*-
"""
Created on Mon Feb 01 16:00:25 2016

@author: Kostya S.
This module  contains functions for processing on Datasets
"""
from __future__ import division
import scipy as sp
from scipy.io import loadmat
from frames import Frames as fr
from fnmatch import fnmatch
import os
import sqlite3
from os.path import join
from sklearn.utils import shuffle
from util import Util as util

class Datasets():
    
    @staticmethod
    def get_aflw_face_data(k = 12):
        dbpath = 'F:\\datasets\\image_data_sets\\faces\\AFLW'
        dbpath = join(dbpath,'aflw.sqlite')
        rfpath = 'F:\\datasets\\image_data_sets\\faces\\AFLW\\img'
        conn = sqlite3.connect(dbpath)
        X = []
        c = 0
        for file_id,x,y,h,w in conn.execute('SELECT file_id,x,y,h,w FROM Faces NATURAL JOIN FaceRect'):
            fpath = join(rfpath,file_id)
            frame = fr.get_frame(fpath)
            no_neg = sp.all(sp.array([x,y,h,w]) > 0) ## ignore a bad data in sql table
            if frame != None and no_neg:
                
                face = fr.get_patch(frame,y,x,(h,w))
                face_r,good_example = Datasets.sample_resize(face,k,k)
                if good_example:
                    print('face:',fpath)
                    vec = fr.frame_to_vect(face_r)
                    X.append(vec)
                    #face_flip = fr.flip_frame(face)
                    face_flip_r = fr.flip_frame(face_r)
                    vec = fr.frame_to_vect(face_flip_r)
                    X.append(vec)
                    #if c % 1000 == 0:
                    #fr.write_frame('F:\\1\\'+'flip'+str(file_id),face_flip)
                    #fr.write_frame('F:\\1\\'+str(file_id),face_r)
                    #fr.write_frame('F:\\1\\'+'flip'+str(file_id)+'_r',face_flip_r)
                    c += 1
        X = sp.array(X)
        y = sp.ones(len(X))
        return X,y
                
    @staticmethod
    def load_wider_face(path_to_mat):
        ## load dataset event labels, bboxes
        ## works with mat files < v7.3, other case it should to be converted
        r_dict = {}
        mat = loadmat(path_to_mat)
        files =  mat['file_list']
        bboxs = mat['face_bbx_list']
        for cell1,cell2 in zip(files,bboxs):
            for img,bx in zip(cell1[0],cell2[0]):
                fname =  img[0][0]
                bbox_r = []
                for b in bx:
                    b = sp.vectorize(lambda x: int(round(x)))(b)
                    bbox_r.append(b)
                    bbox_r = sp.array(bbox_r[0])
                r_dict[(fname)] = bbox_r
        return r_dict
    
    @staticmethod
    def sample_resize(subframe,k_check = 12, k_res = 12):
        '''        
        return resized frame or reject it if any of side less than k
        '''
        H,W,dim = subframe.shape
        # there are negative coord's in datasets :(
        if  H <= 0 or W <= 0:
            return (subframe,False)

        #print(H,W,dim)
        if H >= k_check or W >= k_check:
            return (fr.resize_frame(subframe,(k_res,k_res)),True)
        else:
            return (subframe,False)
            
    @staticmethod
    def get_train_face_wider_data(k = 12,write_to_disk = False):
        '''
        cut faces (positive examples) by bboxes from all images in  dataset
        return X - features
               y - labels
               cnt - count of examples
        '''
        X,y = [],[]
        root = 'F:\\Datasets\\image_data_sets\\faces\\WIDERFace\\'
        pattern = "*.jpg"
        
        bboxs = Datasets.load_wider_face(os.path.join(root,'wider_face_split','wider_face_train_v7.mat'))
        for path, subdirs, files in os.walk(root,'WIDER_train'):
            for indx,iname in enumerate(files):
                if fnmatch(iname, pattern):
                    ipath = os.path.join(path, iname)
                    print('face:',ipath)
                    img = fr.get_frame(ipath)
                    H,W,dim =  img.shape
                    bbox_list = bboxs[iname[:-4]]
                    for bbox in bbox_list:
                        face = fr.get_patch(img,bbox[1],bbox[0],(bbox[2],bbox[3]))
                        #fr.write_frame('F:\\1\\' + str(c),face)
                        
                        if indx % 20 == 0:
                            face_pathes = shuffle(fr.split_frame(face,wshape=(k,k)),random_state = 42)[:20]
                            for e in face_pathes:
                                print('in_face:',ipath)
                                X.append(fr.frame_to_vect(e))
                                y.append(0)
                        
                        face_r,good_example = Datasets.sample_resize(face,k,k)
                        
                        if good_example:
                            
                            vec = fr.frame_to_vect(face_r)
                            X.append(vec)
                            y.append(1)
                        
                            face_r_flip = fr.flip_frame(face_r)
                            vec = fr.frame_to_vect(face_r_flip)                            
                            X.append(vec)
                            y.append(1)
                            
        X = sp.array(X)
        y = sp.array(y)
        #y = sp.ones(len(X))                    
        return X,y
    
    @staticmethod
    def get_train_non_face_data(k = 12,write_to_disk = False):
        '''
        cut non-faces (negative examples) by pick random patch (if in not overlaped with any 
        face bbox  from all images  in dataset
        return X - features
               y - labels
               cnt - count of examples
        '''
        X = []
        def yield_gen(data):
            data = shuffle(data)[:100]
            for e in data:
                yield e
                
        root = 'F:\\Datasets\\image_data_sets\\non-faces'
        pattern = "*.jpg"
        c = 0
        for path, subdirs, files in os.walk(root):
            for iname in files:
                if fnmatch(iname, pattern):
                    ipath = os.path.join(path, iname)
                    img = fr.get_frame(ipath)
                    print('non_face:',ipath)
                    if img == None:
                        continue
                    H,W =  img.shape[:2]
                    for e  in yield_gen(fr.split_frame(img,wshape=(k,k))):
                        X.append(fr.frame_to_vect(e))
                        if c % 10000 == 0:
                            fr.write_frame('F:\\1\\non_face_'+str(c),e)
                        c += 1                        
        X = sp.array(X)
        y = sp.zeros(len(X))
        return X,y
        
    @staticmethod
    def get_train_wider_calib_data(n = None,k = 12):
        '''
        for calibration net
        return X - features
               y - labels
               cnt - count of examples
        '''
        sp.random.seed(42)
        X_data,y_data = [],[]
        
        suff = str(k)
        c = 0
        X_name = 'train_data_icalib_'+ suff +  '.npz'
        y_name = 'labels_icalib_'+ suff + '.npz'
        label = -1
        root = 'F:\\Datasets\\image_data_sets\\faces\\WIDERFace\\'
        pattern = "*.jpg"
        bboxs = Datasets.load_wider_face(os.path.join(root,'wider_face_split','wider_face_train_v7.mat'))
        for path, subdirs, files in os.walk(root,'WIDER_train'):
            for iname in files:
                if fnmatch(iname, pattern):
                    ipath = os.path.join(path, iname)
                    img = fr.get_frame(ipath)
                    H,W =  img.shape[:2]
                    bbox_list = bboxs[iname[:-4]]
                    for bbox in bbox_list:
                        label = sp.random.randint(0,45)                            
                        i,j,h,w = [int(e) for e in util.inv_calib(bbox[1],bbox[0],bbox[2],bbox[3],label)]
                        face = fr.get_patch(img,i,j,(h,w))
                        face_r,good_example = Datasets.sample_resize(face,k,k)
                        if good_example:
                            vec_icalib = fr.frame_to_vect(face_r)                            
                            X_data.append(vec_icalib)
                            y_data.append(label)
                            c += 1
                            print('face calib:',label,ipath) 

        
        dbpath = 'F:\\datasets\\image_data_sets\\faces\\AFLW'
        dbpath = join(dbpath,'aflw.sqlite')
        rfpath = 'F:\\datasets\\image_data_sets\\faces\\AFLW\\img'
        conn = sqlite3.connect(dbpath)
        
        for file_id,x,y,h,w in conn.execute('SELECT file_id,x,y,h,w FROM Faces NATURAL JOIN FaceRect'):
            fpath = join(rfpath,file_id)
            frame = fr.get_frame(fpath)
            no_neg = sp.all(sp.array([x,y,h,w]) > 0) ## ignore a bad data in sql table
            
            if frame != None and no_neg:
                #face_o = fr.get_patch(frame,y,x,(h,w))
                x,y,h,w = [int(e) for e in util.inv_calib(x,y,h,w,label)]
                face = fr.get_patch(frame,x,y,(h,w))
                face_r,good_example = Datasets.sample_resize(face,k,k)
                face_flip_r,_ = Datasets.sample_resize(face,k,k)
                if good_example:
                    label = sp.random.randint(0,45)
             
                    vec_icalib = fr.frame_to_vect(face_r)                            
                    X_data.append(vec_icalib)
                    y_data.append(label)
                    print('face calib:',label,file_id) 
                    label = sp.random.randint(0,45)
                    vec_icalib = fr.frame_to_vect(face_flip_r)                            
                    X_data.append(vec_icalib)
                    y_data.append(label)
                    print('face calib:',label,file_id) 
        
                    #fr.write_frame('F:\\1\\' + str(c) + '_calib',face)
                    #fr.write_frame('F:\\1\\' + str(c),face_o)
                                        
                    c += 1
                
        y_data = sp.array(y_data)
        sp.savez(y_name,y_data)
        X_data = sp.array(X_data)
        sp.savez(X_name,X_data)
        return X_data,y_data
    
    @staticmethod
    def get_train_data(n_pos = 46443, n_neg = 206940,k=12):        
        '''
        megre positive and negative examples
        '''
        suff = str(k)        
        X_name = 'train_data_'+ suff + '.npz'
        y_name = 'labels_'+ suff + '.npz' 
        if not(os.path.exists(X_name) and os.path.exists(y_name)):
            X_pos = []            
#            X_train_face,y_train_face  = Datasets.get_train_face_wider_data(k = k)
#            X_pos = X_train_face[y_train_face==1]
#            X_pos = X_train_face
            X_aflw,y_train_face_aflw  = Datasets.get_aflw_face_data(k = k)
#            if len(X_pos) > 0:
#                X_pos = sp.vstack( [X_pos,X_aflw] )
#            else:
#                X_pos = X_aflw
            X_pos = X_aflw
            X_train_non_face,y_train_non_face =  Datasets.get_train_non_face_data(k = k)
            print('c1_pos:',len(X_pos))
            #print((X_train_face[y_train_face==0].shape,X_train_non_face.shape))
#            if len(X_train_face[y_train_face==0]) > 0:
#                X_neg = sp.vstack( (X_train_face[y_train_face==0],X_train_non_face) )
#            else:
#                X_neg = X_train_non_face
            X_neg = X_train_non_face
            X_pos = shuffle(X_pos,random_state=42)
            X_neg = shuffle(X_neg,random_state=42)
            X_pos = X_pos[:n_pos]
            X_neg = X_neg[:n_neg]
            
            n_neg = len(X_neg)
            n_pos = len(X_pos)
            y_pos = sp.ones(n_pos,int)
            y_neg = sp.zeros(n_neg,int)
            X = sp.vstack((X_pos,X_neg))
            y = sp.hstack( (y_pos,y_neg) )
            X,y = shuffle(X,y,random_state=42)
            sp.savez(X_name,X)
            sp.savez(y_name,y)