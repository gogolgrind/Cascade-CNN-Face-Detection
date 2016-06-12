# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:50:33 2016

@author: Kostya S.
"""
import scipy as sp
from frames import Frames as fr
from cnn_cascade_lasagne import Cnn as Cnnl
import math
import cv2
from time import time
import seaborn as sns
import itertools
from util import nms

sn = (0.83, 0.91, 1.0, 1.10, 1.21)
xn = (-0.17, 0.0, 0.17)
yn = (-0.17, 0.0, 0.17)
prod = [e for e in itertools.product(sn,xn,yn)]
calib = lambda i,j,h,w,n: [ round(i-(prod[n][1]*w/prod[n][0])),round(j-(prod[n][2]*h/prod[n][0])),round(h/prod[n][0]),round(w/prod[n][0]) ]
calib_apply = lambda i,j,h,w,sn,xn,yn: [ round(i-(xn*w/sn)),round(j-(yn*h/sn)),round(h/sn),round(w/sn) ]
inv_calib = lambda i,j,h,w,n:  [ round(i-(-prod[n][1]*w/(prod[n][0]**-1))),round(j-(-prod[n][2]*h/(prod[n][0]**-1))),round(h/prod[n][0]**-1),round(w/prod[n][0]**-1) ]


## ?????
def mean_pattern(proba,t):
    proba_t = proba[proba > t]
    z = len(proba_t)
    if z == 0:
        return (1,1,1)
    s,x,y = 0,0,0
    for j,ind in enumerate(proba > t):
        if ind:
            s += prod[j][0]
            y += prod[j][2]
            x += prod[j][1]
    s /= z
    x /= z
    y /= z
    return (s,x,y)


def apply_12_net(image,iname,nn,p_level = 16,k = 1.18,debug = 1):
    py = fr.get_frame_pyramid(image,k=k,t=p_level) 
    suff = nn.nn_name[:2]
    h,w = int(suff),int(suff)
    rows,cols,d = image.shape
    s = time()
    rbb = []
    ## check last k levels
    k = 1
    t_level = p_level-k
    for m,frame in enumerate(py):
        if m < t_level:
            continue
        H,W,dim = frame.shape
        X_test = []
        pnt = []
        for y in range(0,H - h + 1,4):
            for x in range(0,W - w + 1,4):
                subframe = frame[y:y+h,x:x+w]
                X_test.append(fr.frame_to_vect(subframe))
                pnt.append((x,y))
        pnt = sp.array(pnt)
        X_test = sp.array(X_test)
        pred = nn.predict(X_test)
        pnt = pnt[pred==1]
        if len(pred[pred==1]) == 0:
            print('no detection on',str(m))
            continue
        else:
            print('detection on',str(m))
        bb = []
        for p in pnt:
            i = p[0]
            j = p[1]
            if debug:
                cv2.rectangle(frame, (i, j), (i+w, j+h), (255,0,255), 1)    
            bb.append([i,j,i+w,j+h])
        for e in bb:
            x1,y1,x2,y2 = e
            ox1 = math.floor(image.shape[1] * (x1/frame.shape[1]))
            oy1 = math.floor(image.shape[0] * (y1/frame.shape[0]))
            ox2 = math.floor(image.shape[1] * (x2/frame.shape[1]))
            oy2 = math.floor(image.shape[0] * (y2/frame.shape[0]))
            if m >= p_level - k:
                rbb.append([ox1,oy1,ox2,oy2])
        if debug:        
            cv2.imwrite(iname[:-4]+'_pipeline_' + 'lev'+str(m) + '_' + nn.nn_name  + '_.jpg',frame)
    
    f = time()
    print('time is:',f-s)
    return sp.array(rbb)
    
def apply_24_net(image,iname,nn,bb):
    X24 = []
    X12 = [] 
    for c,e in enumerate(bb):
        x1,y1,x2,y2 = e
        sub = image[y1:y2,x1:x2]     
        sub12 = fr.resize_frame(sub,(12,12))
        sub24 = fr.resize_frame(sub,(24,24))
        X12.append(fr.frame_to_vect(sub12))
        X24.append(fr.frame_to_vect(sub24))
    X12 = sp.array(X12)
    X24 = sp.array(X24)
    
    pred = nn.predict(X=X24,X12=X12)
    return bb[pred==1]
    
def apply_48_net(image,iname,nn,bb):
    X24 = []
    X12 = []
    X48 = []
    for c,e in enumerate(bb):
        x1,y1,x2,y2 = e
        sub = image[y1:y2,x1:x2]
        sub12 = fr.resize_frame(sub,(12,12))
        sub24 = fr.resize_frame(sub,(24,24))
        sub48 = fr.resize_frame(sub,(48,48))
        X12.append(fr.frame_to_vect(sub12))
        X24.append(fr.frame_to_vect(sub24))
        X48.append(fr.frame_to_vect(sub48))
    X12 = sp.array(X12)
    X24 = sp.array(X24)
    X48 = sp.array(X48)
    pred = nn.predict(X=X48,X24=X24,X12=X12)
    return bb[pred==1]
 
def apply_calib_net(image,iname,nn,bb):
    X_calib = []
    for c,e in enumerate(bb):
        x1,y1,x2,y2 = e
        sub = image[y1:y2,x1:x2]
        v = int(nn.nn_name[:2])
        subr = fr.resize_frame(sub,(v,v))
        X_calib.append(fr.frame_to_vect(subr))
        #fr.write_frame('F:\\1\\'+str(c),sub)
    X_calib = sp.array(X_calib)
    pred_proba = nn.predict_proba(X_calib)
    pred = nn.predict(X_calib)
    nbb = []
    for e,lb in zip(bb,pred_proba):
        t = 0.3
        x1,y1,x2,y2 = e
        sn,xn,yn = mean_pattern(lb,t)
        h = abs(y1-y2)
        w = abs(x1-x2)
        
        #ii,jj,hh,ww = [int(e) for e in calib(x1,y1,h,w,lb)]
        #if sp.any(sp.array([ii,jj,hh,ww]) < 0):
        #    continue
        ii,jj,hh,ww = [max(0,int(e)) for e in calib_apply(x1,y1,h,w,sn,xn,yn)]
        #print(ii,jj,hh,ww)
        nbb.append([ii,jj,ii+ww,jj+hh])
    for c,e in enumerate(nbb):
        x1,y1,x2,y2 = e
        sub = image[y1:y2,x1:x2]
        #fr.write_frame('F:\\1\\'+str(c) + 'calib',sub)
    return sp.array(nbb)
    
def draw_bboxes(iname,bb):
    debug = 1
    image = fr.get_frame(iname)
    marked_image = image.copy()
    for box in bb:
        x1,y1,x2,y2 = box
        cv2.rectangle(marked_image, (x1, y1), (x2, y2), (64,0,192), 1)
    if debug:            
        cv2.imwrite('result'+'_pipeline_rescale_image_' + '.jpg',marked_image)
    else:
        sns.plt.imshow(marked_image)

def main():
    iname = 'img/test/1226.jpg'
    nn12 = Cnnl('12-net').load_model('12-net_lasagne_.pickle')
    nn_calib12 =  Cnnl('12-calib_net').load_model('12-calib_net_lasagne_.pickle')
    nn24 = Cnnl(nn_name = '24-net',subnet=nn12).load_model('24-net_lasagne_.pickle')
    nn_calib24 =  Cnnl('24-calib_net').load_model('24-calib_net_lasagne_.pickle')
    nn48 = Cnnl(nn_name = '48-net',subnet=nn24).load_model('48-net_lasagne_.pickle')
    nn_calib48 =  Cnnl('48-calib_net').load_model('48-calib_net_lasagne_.pickle')
    image = fr.get_frame(iname)
    p = 16
    k = 1.18
    bb = apply_12_net(image = image,iname = iname,nn = nn12,p_level=p,k=k,debug=0)
    bb = apply_calib_net(image = image, iname = iname,bb=bb,nn=nn_calib12)
    bb = nms(bb,T=0.8)
    bb = apply_24_net(image = image, iname = iname, bb = bb,nn = nn24 )
    bb = apply_calib_net(image = image, iname = iname,bb = bb,nn=nn_calib24)
    bb = nms(bb,T=0.8)
    bb = apply_48_net(image = image, iname = iname, bb = bb,nn = nn48 )
    bb = nms(bb,T=0.1)
    bb = apply_calib_net(image = image, iname = iname,bb=bb,nn=nn_calib48)    
    draw_bboxes(iname,bb)
if __name__ == '__main__':
    main()