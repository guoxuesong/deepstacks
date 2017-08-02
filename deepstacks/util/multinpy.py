#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import os
import sys
import numpy as np
from format import open_memmap as open_memmap_partial #https://github.com/jonovik/numpy/raw/offset_memmap/numpy/lib/format.py

file2x={}

def readnpy(paths,shape,a,fast=False,dtype='uint8',maxshape=None):
    dt=np.dtype(dtype)
    if type(paths)==str:
        paths=[paths]
    shapelist=(len(a),)+shape
    shape=(1,)+shape
    shapesize=np.prod(shape)
    #maxsize=sys.maxint//dt.itemsize//shapesize
    maxsize=((1<<31)-1)//dt.itemsize//shapesize
    if maxshape is None:
        maxshape=(maxsize,)+shape[1:]
    res=np.zeros(shapelist)
    if not fast:
        i=0
        for j in a:
            k=j%len(paths)
            j=j//len(paths)
            X=open_memmap_partial(filename=paths[k],dtype=dtype,offset=j*shapesize,shape=shape,mode='r+')
            res[i]=X[0:1]
            i=i+1
    else:
        for k0 in range(len(paths)):
            X=None
            if paths[k0] in file2x:
                X=file2x[paths[k0]]
            else:
                X=open_memmap_partial(filename=paths[k0],dtype=dtype,shape=maxshape,mode='r+')
                file2x[paths[k0]]=X
            i=0
            for j in a:
                k=j%len(paths)
                j=j//len(paths)
                if k0==k:
                    res[i]=X[j:j+1]
                i=i+1
    return res

def writenpy(paths,shape,a,data,fast=False,dtype='uint8',maxshape=None):
    #global last_file
    #global last_X
    dt=np.dtype(dtype)
    #print data.shape
    if type(paths)==str:
        paths=[paths]
    shapelist=(len(a),)+shape
    shape=(1,)+shape
    shapesize=np.prod(shape)
    #maxsize=sys.maxint//dt.itemsize//shapesize
    maxsize=((1<<31)-1)//dt.itemsize//shapesize
    if maxshape is None:
        maxshape=(maxsize,)+shape[1:]
    maxshape0=(maxsize,)+shape[1:]
    #res=np.zeros(shapelist)
    if not fast:
        i=0
        for j in a:
            k=j%len(paths)
            j=j//len(paths)
            if not os.path.exists(paths[k]):
                open_memmap_partial(filename=paths[k],dtype=dtype,shape=maxshape0,mode='w+')
            X=open_memmap_partial(filename=paths[k],dtype=dtype,offset=j*shapesize,shape=shape,mode='r+')
            X[0]=data[i]
            i=i+1
    else:
        for k0 in range(len(paths)):
            if not os.path.exists(paths[k0]):
                #print paths[k0],dtype,maxshape
                open_memmap_partial(filename=paths[k0],dtype=dtype,shape=maxshape,mode='w+')
            X=None
            if paths[k0] in file2x:
                X=file2x[paths[k0]]
            else:
                X=open_memmap_partial(filename=paths[k0],dtype=dtype,shape=maxshape,mode='r+')
                file2x[paths[k0]]=X
#            if paths[k0]==last_file:
#                X=last_X
#            else:
#                last_X=None
#                X=open_memmap_partial(filename=paths[k0],dtype=dtype,shape=maxshape,mode='r+')
#                last_file=paths[k0]
#                last_X=X
            i=0
            for j in a:
                k=j%len(paths)
                j=j//len(paths)
                if k0==k:
                    X[j]=data[i]
                i=i+1

#np.lib.format.open_memmap("test.npy",dtype='uint8',shape=(200*4*270, 3, 64, 64) ,mode='w+')

#writenpy(('test1.npy','test2.npy'),(3,64,64),(200*4*270-1,),np.zeros((3,64,64),dtype='uint8'))
#writenpy(('test1.npy','test2.npy'),(3,64,64),(200*4*270,),np.zeros((3,64,64),dtype='uint8'))
