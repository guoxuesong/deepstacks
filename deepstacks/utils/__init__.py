#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import theano
import numpy as np

def floatX(val):
    return vars(np)[theano.config.floatX](val)

class Keeper(object):
    def __init__(self,func):
        self.func=func
        self.args=None
    def __call__(self,*args):
        if self.args is None:
            self.args=args
        print 'Keeper:',self.args
        return self.func(*self.args)
