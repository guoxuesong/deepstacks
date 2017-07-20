#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import neon
from neon.layers.layer import *
from neon.layers.container import *

import random

def ordered_errors(errors,m=None,prefix=''):
    res=[]
    for t in errors:
        if m is None:
            res+=[[prefix+t,[]]]
            for cost,delta in errors[t]:
                res[-1][-1]+=[(cost,delta)]
        else:
            raise NotImplementedError
    return sorted(res,key=lambda x:x[0])

class MulticostZeros(neon.layers.Multicost):
    def __init__(self,costs,layers,target_layers):
        super(MulticostZeros,self).__init__(costs)
        self.layers=layers
        self.target_layers=target_layers
    def get_cost(self, inputs, targets):
        if type(targets)!=list:
            targets=[targets]
        shapes = map(lambda x:x.out_shape,self.layers)
        res=[targets[0]]
        p=0
        for layer,shape in zip(self.target_layers,shapes):
            if layer==0:
                res+=[np.zeros(shape)]
            else:
                if len(targets)>1:
                    p+=1
                res+=[targets[p]]
        targets=res
        #print inputs,targets
        #targets=targets+map(np.zeros,shapes)
        return super(MulticostZeros,self).get_cost(inputs,targets)
    def get_errors(self, inputs, targets):
        if type(targets)!=list:
            targets=[targets]
        shapes = map(lambda x:x.out_shape,self.layers)
        res=[targets[0]]
        p=0
        for layer,shape in zip(self.target_layers,shapes):
            if layer==0:
                res+=[np.zeros(shape)]
            else:
                if len(targets)>1:
                    p+=1
                res+=[targets[p]]
        targets=res
        #targets=targets+map(np.zeros,shapes)
        return super(MulticostZeros,self).get_errors(inputs,targets)

def get_loss(errors,watchpoints,cost0=None):
    errors = ordered_errors(errors)
    #print errors

    layers=[]
    targets=[]
    costs=[cost0] if cost0 is not None else []
    tagslice=[]
    for a in errors:
        beginpos=len(layers)
        for cost,delta in a[1]:
            if type(delta)==list:
                layers+=[delta[0]]
                targets+=[delta[1]]
            else:
                layers+=[delta]
                targets+=[None]
            costs+=[cost]
        endpos=len(layers)
        tagslice+=[[a[0],slice(beginpos,endpos)]]

    cost = MulticostZeros(costs,layers,targets)
    #if len(layers)>1:
    #    cost = MulticostZeros(costs,layers)
    #else:
    #    cost=costs[0]

    return cost,layers,tagslice

def get_watchslice(watchpoints):
    trainwatch = {}
    valwatch = {}
    for tag,errs in watchpoints:
        if tag.startswith('train:'):
            trainwatch[tag]=errs
        else:
            valwatch[tag]=errs
    ig,train_layers,train_tagslice=get_loss(trainwatch,[])
    ig,val_layers,val_tagslice=get_loss(valwatch,[])
    return train_layers,train_tagslice,val_layers,val_tagslice

def InputLayer(name):
    return neon.layers.layer.DataTransform(neon.transforms.activation.Identity(),name=name)

def get_inputs(network):
    res=[]
    for layer in network.layers_fprop():
        print type(layer)
        if isinstance(layer,neon.layers.layer.DataTransform):
            res+=[layer.name]
    return res
def get_targets(cost):
    return map(lambda x:x.name,filter(lambda x:x is not None,cost.target_layers))

