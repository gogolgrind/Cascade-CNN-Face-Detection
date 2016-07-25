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
    def frames2batch(k = 12,batch_size = 1024, is_calib = False):
        pos = util.get_files(rootdir = 'F:\\train_data\\pos\\')
        neg = util.get_files(rootdir = 'F:\\train_data\\neg\\')
        pos = shuffle(pos)
        neg = shuffle(neg)
        total = pos + neg
        total  = shuffle(total)
        batch = []
        c = 0
        bpath = 'F:\\train_data\\batch\\'
        for item_path in total:
            
            frame = fr.get_frame(item_path)
            frame_r = fr.resize_frame(frame,(k,k))
            if frame_r == None:
                continue
            vec = fr.frame_to_vect(frame_r)
            label = 1 if item_path.split('\\')[-1].find('pos') > 0 else 0
            print(item_path,label)
            batch.append((vec,label))
            if len(batch) > 0 and len(batch) % batch_size == 0:
                batch = sp.array(batch)
                sp.savez(bpath + str(c) + '_' + str(k) + ('_' if not is_calib else '_calib-')  + 'net',batch)
                batch = []
                
                c += 1
        if len(batch) > 0 and len(batch) % batch_size == 0:
            batch = sp.array(batch)
            sp.savez(bpath + str(c) + '_' + str(k) + ('_' if not is_calib else '_calib')  + '-net',batch)
            batch = []
            c += 1
    
    @staticmethod 
    def data_augmentation(frame,y,x,w,h):
        face = fr.get_patch(frame,y,x,(w,h))
        face_flip = fr.flip_frame(face)
        t1 = sp.random.randint(0,3)
        t2 = sp.random.randint(0,3)
        face_narrow = fr.get_patch(frame,y,x,(w-t2,h-t2))
        face_wide = fr.get_patch(frame,y,x,(w+t2,h+t2))
        t1 = sp.random.randint(0,3)
        t2 = sp.random.randint(0,3)
        
        face_shift1 = fr.get_patch(frame,y+t1,x+t1,(w+t2,h+t2))
        face_shift2 = fr.get_patch(frame,y-t1,x-t1,(w-t2,h-t2))
        th = float((1 if sp.random.randint(0,2) % 2 == 0 else -1) * sp.random.randint(45,90))
        
        face_rot = fr.frame_rotate(face,theta = th)
        
        faces_list = filter(lambda x: x != None,[face,face_flip,face_narrow,face_wide,face_shift1,face_shift2,face_rot])
        return faces_list
    
    @staticmethod
    def get_fddb_face_data(k = 12, on_drive = False):
        root = 'F:\\datasets\\image_data_sets\\faces\\FDDB\\'
        iroot = os.path.join(root,'originalPics')
        eroot = os.path.join(root,'FDDB-folds')
        pattern = '-ellipseList.txt'
        c = 0
        X,y = [],[]
        for path, subdirs, files in os.walk(eroot):
            for fname in files:
                if fname.find(pattern) > 0:
                    fpath = os.path.join(path,fname)
                    print(fpath)
                    with open(fpath) as f:
                        lines = sp.array(f.readlines())
                        paths_indx = sp.where([line.find('/') > 0 for line in lines])[0]
                        counts_indx = paths_indx + 1
                        
                        paths = sp.array([e.strip() for e in lines[paths_indx]])
                        ellipces = []
                        for i in counts_indx:
                            cnt = int(lines[i])
                            ellipces.append(lines[i+1:i+cnt+1])
                        ellipces = [ [ [float(num) for num in line.split()[:-1]] for line in e] for e in ellipces]
                        ellipces = sp.array(ellipces)
                        for iname,ells in zip(paths[:],ellipces[:]):
                            ppath = os.path.join(iroot,iname.replace('/','\\')) + '.jpg'
                            file_id = iname.split('/')[-1]
                            
                            frame = fr.get_frame(ppath)
                            for item in ells:
                                ra,rb,theta,x,y = item
                                x1,y1,x2,y2 = util.ellipse2bbox(a = ra, b = rb, angle = theta, cx = x, cy = y)
                                x = x1
                                y = y1
                                h = abs(y2-y1)
                                w = abs(x2-x1)
                                print(file_id,(y,x,h,w))
                                
                                non_neg = x > 0 and y > 0
                                if not non_neg:
                                    continue
                                if on_drive:   
                                    for item in Datasets.data_augmentation(frame,y,x,w,h):
                                        fr.write_frame('F:\\train_data\\pos\\' + str(c) + '_' + str(file_id) + '_pos',item)
                                        c +=1
                                else:
                                    pass
        X = sp.array(X)
        y = sp.ones(len(X))
        return X,y                    
                        
    @staticmethod
    def get_aflw_face_data(k = 12, on_drive = False):
        dbpath = 'F:\\datasets\\image_data_sets\\faces\\AFLW'
        dbpath = join(dbpath,'aflw.sqlite')
        rfpath = 'F:\\datasets\\image_data_sets\\faces\\AFLW\\img'
        conn = sqlite3.connect(dbpath)
        X = []
        c = 0
        for file_id,x,y,ra,rb,theta in conn.execute('SELECT file_id,x,y,ra,rb,theta FROM Faces NATURAL JOIN FaceEllipse'):
            fpath = join(rfpath,file_id)
            frame = fr.get_frame(fpath)
            x1,y1,x2,y2 = util.ellipse2bbox(a = ra, b = rb, angle = theta, cx = x, cy = y)
            x = x1
            y = y1
            h = abs(y2-y1)
            w = abs(x2-x1)
            no_neg = sp.all(sp.array([x,y,h,w]) > 0) ## ignore a bad data in sql table
            if frame != None and no_neg:
                y,x,w,h = [int(e) for e in (y,x,w,h)]
                face = fr.get_patch(frame,y,x,(w,h))
                face_r,good_example = Datasets.sample_resize(face,k,k)
                if good_example:
                    print('face:',fpath)
                    vec = fr.frame_to_vect(face_r)
                    if not on_drive:
                        X.append(vec)
                        face_flip_r = fr.flip_frame(face_r)
                        vec = fr.frame_to_vect(face_flip_r)
                        X.append(vec)
                    else:
                        for item in Datasets.data_augmentation(frame,y,x,w,h):
                            fr.write_frame('F:\\train_data\\pos\\' + str(c) + '_' + str(file_id)[:-4] + '_' + 'pos',item)
                            c +=1
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
    def get_train_non_face_data(k = 12,patch_per_img = 25,on_drive = False):
        '''
        cut non-faces (negative examples) by pick random patch (if in not overlaped with any 
        face bbox  from all images  in dataset
        return X - features
               y - labels
               cnt - count of examples
        '''
        X = []
        def yield_gen(data):
            data = shuffle(data)[:patch_per_img]
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
                    if not on_drive:
                        for e  in yield_gen(fr.split_frame(img,wshape=(k,k))):
                            X.append(fr.frame_to_vect(e))
                    else:
                        for e  in yield_gen(fr.split_frame(img,wshape=(util.k_max,util.k_max))):
                            fr.write_frame('F:\\train_data\\neg\\' + str(c) + '_' 'nonface'+'_neg',e)
                            c += 1
        X = sp.array(X)
        y = sp.zeros(len(X))
        return X,y
        
    @staticmethod
    def get_train_calib_data(k = 12):
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
        dbpath = 'F:\\datasets\\image_data_sets\\faces\\AFLW'
        dbpath = join(dbpath,'aflw.sqlite')
        rfpath = 'F:\\datasets\\image_data_sets\\faces\\AFLW\\img'
        conn = sqlite3.connect(dbpath)
        c = 0
        for file_id,x,y,ra,rb,theta in conn.execute('SELECT file_id,x,y,ra,rb,theta FROM Faces NATURAL JOIN FaceEllipse'):
            fpath = join(rfpath,file_id)
            frame = fr.get_frame(fpath)
            x1,y1,x2,y2 = util.ellipse2bbox(a = ra, b = rb, angle = theta, cx = x, cy = y)
            x = x1
            y = y1
            h = abs(y2-y1)
            w = abs(x2-x1)
            no_neg = sp.all(sp.array([x,y,h,w]) > 0) ## ignore a bad data in sql table
            if frame != None and no_neg:
                y,x,w,h = [int(e) for e in (y,x,w,h)]
                face = fr.get_patch(frame,y,x,(w,h))
                #fr.write_frame('F:\\1\\' + str(c) + 'orig',face)
                c += 1
                for ((new_y,new_x,new_w,new_h),label) in [(util.calib(y,x,w,h,k),k) for k in sp.random.randint(0,45,5)]:
                    face = fr.get_patch(frame,new_y,new_x,(new_w,new_h))
                    no_neg_calib = sp.all(sp.array([new_x,new_y,new_h,new_w]) > 0)
                    face_r,good_example = Datasets.sample_resize(face,k,k)
                    
                    if good_example and no_neg_calib:
                        #fr.write_frame('F:\\1\\' + str(c) + 'calib_'+str(label) ,face)
                        print('face:',fpath,label)
                        vec = fr.frame_to_vect(face_r)    
                        X_data.append(vec)
                        y_data.append(label)
                        
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