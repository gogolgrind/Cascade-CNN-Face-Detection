# -*- coding: utf-8 -*-
"""
Created on Wed May 25 14:54:47 2016
@author: Kostya S, nms code from: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/ and
    https://github.com/layumi/2015_Face_Detection/blob/master/nms.m
"""

import numpy as np

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
    idxs = np.argsort(y2)    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        I = w * h
        #overlap_ratio = I / area[idxs[:last]]
        overlap_ratio = I /(area[i] +  area[idxs[:last]] - I)
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap_ratio > T)[0])))
    return boxes[pick].astype("int") 