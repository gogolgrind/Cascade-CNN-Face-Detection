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
from util import Util as util

def mean_pattern(proba,t):
    proba_t = proba[proba > t]
    prods = util.prod[proba > t]
    ds, dx, dy =  ((proba_t * prods.T).T ).sum(axis=0) 
    return (ds, dx, dy)


def apply_12_net(image,frame,iname,m,nn,debug = 1):
     
    suff = nn.nn_name[:2]

    patch_size = int(suff)
    
    s = time()
    rbb = []
    
    H,W,dim = frame.shape
    X_test = []
    pnt = []
    for u in range(0,H - patch_size + 1,4):
        for v in range(0,W - patch_size + 1,4):
            subframe = fr.get_patch(frame,u,v,(patch_size,patch_size))
            X_test.append(fr.frame_to_vect(subframe))
            pnt.append((v,u))
            
    pnt = sp.array(pnt)
    X_test = sp.array(X_test)
    proba = nn.predict_proba(X_test)
    pred = nn.predict(X_test)
    
    pnt = pnt[pred==1]
    if len(pnt) == 0:
        print('no detection on',str(m))
    else:
        print('detection on',str(m),frame.shape)
    bb = []
    
    for p in pnt:
        i = p[0]
        j = p[1]
        bb.append([i,j,i+patch_size,j+patch_size])
    bb = sp.array(bb)
    
    for e in bb:
        x1,y1,x2,y2 = e
        ratio_x = image.shape[1]/frame.shape[1]
        ratio_y = image.shape[0]/frame.shape[0]
        ox1 = int(round(x1 * ratio_x))
        oy1 = int(round(y1 * ratio_y))
        ox2 = int(round(x2 * ratio_x))
        oy2 = int(round(y2 * ratio_y))
        
        
        rbb.append([ox1,oy1,ox2,oy2])
        
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
    
def apply_48_net(image,iname,nn,bb,debug = 1):
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
    proba = nn.predict_proba(X=X48,X24=X24,X12=X12)
    if debug:
        j = 0
        for e,lb,p in zip(bb,pred,proba):
            x1,y1,x2,y2 = e
            sub = image[y1:y2,x1:x2]
            fr.write_frame('F:\\1\\'+str(j)+ '-'+str(lb) + '-' +str( p[0]) + '-' + str(p[1]),sub)
            j += 1
    return bb[pred==1]
 
def apply_calib_net(image,iname,nn,bb,T=0.3):
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
    nbb = []
    for e,lb in zip(bb,pred_proba):
        t = 1/45
        x1,y1,x2,y2 = e
        ds, dx, dy = mean_pattern(lb,t)
        h = abs(y1-y2)
        w = abs(x1-x2)
        
        
        ii,jj,hh,ww = [int(max(0,e)) for e in util.calib_apply(x1,y1,h,w,ds,dx,dy)]
        
        nbb.append([ii,jj,ii+ww,jj+hh])
 
    return sp.array(nbb)
    
def draw_bboxes(iname,bb,c = 42):
    debug = 1
    image = fr.get_frame(iname)
    marked_image = image.copy()
    for box in bb:
        x1,y1,x2,y2 = box
        cv2.rectangle(marked_image, (x1, y1), (x2, y2), (0,0,255), 1)
    if debug:            
        cv2.imwrite('result'+'_pipeline_rescale_image_' + str(c) +  '.jpg',marked_image)
    else:
        sns.plt.imshow(marked_image)

def main():
    iname = 'img/test/win1.jpg'
    nn12 = Cnnl('12-net').load_model('12-net_lasagne_.npz')
    nn24 = Cnnl(nn_name = '24-net',subnet=nn12).load_model('24-net_lasagne_.npz')
    nn48 = Cnnl(nn_name = '48-net',subnet=nn24).load_model('48-net_lasagne_.npz')
    nn_calib12 =  Cnnl('12-calib_net').load_model('12-calib_net_lasagne_.npz')
    nn_calib24 =  Cnnl('24-calib_net').load_model('24-calib_net_lasagne_.npz')
    nn_calib48 =  Cnnl('48-calib_net').load_model('48-calib_net_lasagne_.npz')
    image = fr.get_frame(iname)
    p = 18
    k = 1.18
    t_level = 0
    gbb = []
    s = time()
    py = fr.get_frame_pyramid(image,scale=k,steps=p)
    for m,frame in enumerate(py):
        if m < t_level:
            continue
        
        bb = apply_12_net(image = image,frame=frame,iname = iname,nn = nn12,m=m,debug=0)
        #bb = util.nms(bb,T=0.8)
        cnt_detect = len(bb)
        if cnt_detect  == 0:
            continue
        for b in bb:
            gbb.append(b)

    gbb = sp.array(gbb)
    draw_bboxes(iname,gbb,c='0-r_12')
    
    bb = gbb   
    bb = apply_calib_net(image = image, iname = iname,bb=bb,nn=nn_calib12)
    cnt_detect = len(bb)
    draw_bboxes(iname,bb,c='1-r_calib_12')    
    bb = util.nms(bb,T=0.75)
    draw_bboxes(iname,bb,c='2-r_nms_12')
    
    bb = apply_24_net(image = image, iname = iname, bb = bb,nn = nn24 )
    cnt_detect = len(bb)
    draw_bboxes(iname,bb,c='2-r_24')
    
    bb = apply_calib_net(image = image, iname = iname,bb = bb,nn=nn_calib24)
    cnt_detect = len(bb)
    draw_bboxes(iname,bb,c='3-r_calib_24')
    bb = util.nms(bb,T=0.65)
    draw_bboxes(iname,bb,c='4-r_nms_24')
    cnt_detect = len(bb)
    
    bb = apply_48_net(image = image, iname = iname, bb = bb,nn = nn48 )        
    cnt_detect = len(bb)
    draw_bboxes(iname,bb,c='5-r_48')
    
    
    bb = apply_calib_net(image = image,iname = iname,bb=bb,nn=nn_calib48)
    draw_bboxes(iname,bb,c='6-r_calib_48')
    
    bb = util.nms(bb,T=0.5)
    draw_bboxes(iname,bb,c='7-r_nms_48_final')
    print(time()-s)
if __name__ == '__main__':
    main()
    