#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import numpy as np
import random
import theano

class LocalShuffler(object):
    def __init__(self,size,shape):
        self.xpool=np.zeros((size,)+shape,dtype=theano.config.floatX)
        self.ypool=[0]*size
        self.size=size
        self.fill=0
    def feed(self,data,y):
        if data is None:
            if self.fill>0:
                n = self.fill-1
                self.fill-=1
                return self.xpool[n],self.ypool[n]
            else:
                return None
        if self.fill==self.size:
            n = int(random.random()*self.size)
            res = (self.xpool[n],self.ypool[n])
            self.xpool[n]=data
            self.ypool[n]=y
            return res
        else:
            n = self.fill
            self.xpool[n]=data
            self.ypool[n]=y
            self.fill+=1
            return None
    def is_empty(self):
        return self.fill==0
    def is_full(self):
        assert self.fill<=self.size
        return self.fill==self.size
