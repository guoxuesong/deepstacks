#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import theano
import numpy as np

def floatX(val):
    return vars(np)[theano.config.floatX](val)

def create_binary(a,n):
    res = np.zeros((len(a),n),dtype=theano.config.floatX)
    for i in range(n):
        res[:,i]=a%2
        a=a>>1
    return res

class Keeper(object):
    def __init__(self,func):
        self.func=func
        self.args=None
    def __call__(self,*args):
        if self.args is None:
            self.args=args
        print 'Keeper:',self.args
        return self.func(*self.args)
