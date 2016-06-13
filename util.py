# -*- coding: utf-8 -*-
"""
Created on Wed May 25 14:54:47 2016
@author: Kostya S, nms code from: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/ and
    https://github.com/layumi/2015_Face_Detection/blob/master/nms.m
"""

import scipy as sp

class Util():
    @staticmethod
    def nms(boxes, T = 0.5):
        if len(boxes) == 0:
            return []
        boxes = boxes.astype("float")
        pick = []
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]    
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = sp.argsort(y2)    
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            xx1 = sp.maximum(x1[i], x1[idxs[:last]])
            yy1 = sp.maximum(y1[i], y1[idxs[:last]])
            xx2 = sp.minimum(x2[i], x2[idxs[:last]])
            yy2 = sp.minimum(y2[i], y2[idxs[:last]])
            w = sp.maximum(0, xx2 - xx1 + 1)
            h = sp.maximum(0, yy2 - yy1 + 1)
            I = w * h
            #overlap_ratio = I / area[idxs[:last]]
            overlap_ratio = I /(area[i] +  area[idxs[:last]] - I)
            idxs = sp.delete(idxs, sp.concatenate(([last], sp.where(overlap_ratio > T)[0])))
        return boxes[pick].astype("int")
    
    @staticmethod
    def kfold(X,y, k_fold = 2):
        for k in range(k_fold):
            t_indx = [i for i, e in enumerate(X) if i % k_fold != k]
            v_indx = [i for i, e in enumerate(X) if i % k_fold == k]
            yield t_indx,v_indx