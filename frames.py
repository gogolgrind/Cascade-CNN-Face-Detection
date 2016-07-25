# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 13:37:40 2016
@author: Kostya S.
This module contains functions for processing on images  
"""
import scipy as sp
import cv2
from skimage.transform import pyramid_gaussian,rotate

class Frames():
    ## use open cv or other image processing backed
    with_open_cv = 1
    
    @staticmethod
    def get_random_frame(shape = (12,12,3)):
        return sp.random.random(shape)
        
    @staticmethod
    def resize_frame(frame,shape = (12,12)):
        if not Frames.with_open_cv:
            return sp.misc.imresize(frame,shape)
        try:
            return cv2.resize(frame, shape)
        except:
            print('Bad Data')
            return None
    
    
    @staticmethod       
    def get_frame(path_to_frame = None):
        # load image to numpy array
        if not Frames.with_open_cv:
            frame = sp.misc.imread(path_to_frame)
            # rbg -> bgr for ccompatibility with cv2 
            frame = frame[...,::-1]
            return frame
        return cv2.imread(path_to_frame)
        
    @staticmethod
    def frame_rotate(frame,theta = 45):
        (h, w) = frame.shape[:2]
        center = (w / 2, h / 2)
         
        # rotate the image by 180 degrees
        M = cv2.getRotationMatrix2D(center, theta, 1.0)
        rotated = cv2.warpAffine(frame, M, (w, h))
        return rotated

    @staticmethod
    def flip_frame(frame):    
        return sp.fliplr(frame)
    
    def shift_frame(frame):
        return sp.ndimage.interpolation.shift(frame, 2.0, mode='reflect')
    
    @staticmethod
    def write_frame(name,frame):
        if not Frames.with_open_cv:
            sp.misc.imsave(name+'.jpg',frame)
        else:
            cv2.imwrite(name+'.jpg',frame)
            
    @staticmethod  
    def __pyramid_ski__(frame,steps = 7,scale = 2):
        # create image pyramid from img, skimage implemetation
        py = pyramid_gaussian(frame, downscale = scale)
        return [sp.round_(255.0*e).astype(sp.uint8) for e in py][:steps]
        
    @staticmethod  
    def __pyramid_my__(frame_,steps = 7,scale = 2):
        # create image pyramid from img, skimage implemetation
        frame = frame_.copy()        
        py = []
        s = 1
        H,W,dim = frame.shape
        for e in range(steps):
            h,w = round(H/s),round(W/s) 
            r_frame  = Frames.resize_frame(frame,(w,h))
            py.append(r_frame)
            s *= scale
            
        py = sp.array(py)
        return py
    
        
    @staticmethod        
    def get_frame_pyramid(frame, steps = 7, scale = 2):
        # create image pyramid from img
        return Frames.__pyramid_my__(frame,steps,scale)
        #return Frames.__pyramid_ski__(frame,steps,scale)
            
    @staticmethod            
    def get_patch(frame,i,j,wshape = (12,12)):
        if sp.any([i,j,wshape[0],wshape[1]]) < 0:
            return None
        # get fix size part(patch)  of image
        H,W = frame.shape[:2]
        h,w = wshape    
        #subframe = frame[i:i+w,j:j+h]
        return frame[i:i+w,j:j+h]
        
    @staticmethod                
    def split_frame(frame,wshape = (12,12)):
        # split image to fix size patchs
        H,W = frame.shape[:2]
        h,w = wshape
        r = []
        for i in range(0,H - h + 1,4):
            for j in range(0,W - w + 1,4):
                subframe = Frames.get_patch(frame,i,j,wshape)
                r.append(subframe)  
        return sp.array(r)
    
    @staticmethod        
    def frame_to_vect(frame):
        # tranform rgb image for CNN input layer 
        H,W = frame.shape[:2]
        frame = sp.asarray(frame, dtype = sp.float16) / 255.0
        features = frame.transpose(2,0,1).reshape(3, H, W)
        return features        