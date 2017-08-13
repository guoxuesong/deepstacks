#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

#from pympler import tracker
#tr = tracker.SummaryTracker()

from deepstacks.macros import *
#from memory_profiler import memory_usage

#using_nolearn=False

from ..utils.floatXconst import * 
from ..lasagne.utils import ordered_errors as get_ordered_errors

import sys
import os
import time
import numpy as np
import math
import random
import gc
import fcntl
import copy
from collections import OrderedDict
print 'start theano ...'
import theano
import theano.tensor as T

from ..utils import easyshared
from ..utils import lr_policy 

import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import thread
import cv2
#import matplotlib.pyplot as plt
import argparse

sys.setrecursionlimit(50000)
#import nolearn
#import nolearn.lasagne
#import nolearn.lasagne.visualize

floatX=theano.config.floatX

from ..utils.momentum import adamax
from ..utils.curry import *

from ..utils.multinpy import readnpy,writenpy

def sorted_values(m):#{{{
    if isinstance(m,OrderedDict):
        return m.values()
    else:
        a=sorted(m.keys())
        return [m[k] for k in a]#}}}

#
#def enlarge(a,n):#{{{
#    if n<0:
#        return a[::-n,::-n,:]
#    elif n>0:
#        return np.repeat(np.repeat(a,n,0),n,1)#}}}
#def tile(network,width,height):#{{{
#    network = lasagne.layers.ConcatLayer((network,)*width,axis=2)
#    network = lasagne.layers.ConcatLayer((network,)*height,axis=3)
#    return network#}}}
#
#def imnorm(x):#{{{
#    M=np.max(x)
#    m=np.min(x)
#    l=M-m
#    if l==0:
#        l=1.0
#    res=((x-m)*1.0/l*255.0).astype('uint8')
#    return res#}}}
#def im256(x):#{{{
#    M=1.0
#    m=0.0
#    l=M-m
#    if l==0:
#        l=1.0
#    res=((x-m)*1.0/l*255.0).astype('uint8')
#    return res#}}}
#
#def smooth_abs(x):#{{{
#        return (x*x+floatXconst(1e-8))**floatXconst(0.5);#}}}
#


def mylossfunc(a,b):
    return (a-b)**2.0

class ZeroLayer(lasagne.layers.InputLayer):
    pass

def inputlayer_zeroslike(layer):#{{{
    shape=lasagne.layers.get_output_shape(layer)
    res=ZeroLayer(shape,input_var=T.zeros(shape,dtype=floatX))
    return res#}}}
def inputlayer_zeros(shape):#{{{
    return ZeroLayer(shape,input_var=T.zeros(shape,dtype=floatX))#}}}
def inputlayer_oneslike(layer,scale=1.0):#{{{
    shape=lasagne.layers.get_output_shape(layer)
    res=ZeroLayer(shape,input_var=T.ones(shape,dtype=floatX)*floatXconst(scale))
    return res#}}}
def inputlayer_ones(shape,scale=1.0):#{{{
    return ZeroLayer(shape,input_var=T.ones(shape,dtype=floatX)*floatXconst(scale))#}}}

def touch(fname, times=None):#{{{
    with open(fname, 'a'):
        os.utime(fname, times)#}}}

def set_param_value(params,values,ignore_mismatch=False):#{{{
    res=[]
    if len(params) != len(values):
        raise ValueError("mismatch: got %d values to set %d parameters" %
                         (len(values), len(params)))

    for p, v in zip(params, values):
        pshape=p.get_value().shape
        vshape=v.shape
        if len(pshape) != len(vshape):
            if ignore_mismatch:
                print ("WARNING: mismatch: parameter has shape %r but value to "
                        "set has shape %r") % (pshape, vshape)
                res+=[p]
            else:
                raise ValueError("mismatch: parameter has shape %r but value to "
                                 "set has shape %r" %
                                 (pshape, vshape))
        else:
            needpad=False
            setvalue=True
            if ignore_mismatch:
                padflag=True
                padlist=()
                for i in range(len(pshape)):
                    if pshape[i]<vshape[i]:
                        padflag=False

                if padflag:
                    for i in range(len(pshape)):
                        if pshape[i]>vshape[i]:
                            padlist+=((0,pshape[i]-vshape[i]),)
                            needpad=True
                        else:
                            padlist+=((0,0),)
                else:
                    for i in range(len(pshape)):
                        if pshape[i]<vshape[i]:
                            print "mismatch: parameter has shape %r but value to set has shape %r" % (pshape, vshape)
                            res+=[p]
                            setvalue=False
                            break
            else:
                for i in range(len(pshape)):
                    if pshape[i]!=vshape[i]:
                        raise ValueError("mismatch: parameter has shape %r but value to "
                                         "set has shape %r" %
                                         (pshape, vshape))

            if needpad:
                print 'WARNING: pad parameter value from %r to %r' % (vshape,pshape)
                #print pshape,v.shape,padlist
                v=np.pad(v,padlist,'constant')
                res+=[p]
            if setvalue:
                p.set_value(v)
    return res#}}}
def save_params(epoch,layers,global_params,prefix='',deletelayers=[]):#{{{
    for layers,name in zip((layers,),("layers",)):
        for i in range(len(layers)):
            params = lasagne.layers.get_all_params(layers[i])
            for layer in deletelayers:
                newparams = []
                for t in params:
                    #if t != layer.W and t!=layer.b:
                    #    newparams+=[t]
                    if not t in layer.get_params():
                        newparams+=[t]
                params=newparams
            #params=[]
            #for j in range(len(layers[i])):
            #    params=params+layers[i][j].get_params() 
            values=[p.get_value() for p in params]
            np.savez(prefix+'model-'+name+'-'+str(i)+'.npz', *values)
            if len(values)==0:
                touch(prefix+'model-'+name+'-'+str(i)+'.skip')
    params=global_params
    values=[epoch]+[p.get_value() for p in params]
    np.savez(prefix+'model-global-'+str(0)+'.npz', *values)
    if len(values)==0:
        touch(prefix+'model-global-'+str(0)+'.skip')#}}}
def load_params(layers,global_params,prefix='',partial=False,ignore_mismatch=False,newlayers=[]):#{{{
    epoch = 0
    mismatch = []
    for layers,name in zip((layers,),("layers",)):
        for i in range(len(layers)):
            if not os.path.exists(prefix+'model-'+name+'-'+str(i)+'.npz'):
                break;
            if not os.path.exists(prefix+'model-'+name+'-'+str(i)+'.skip'):
                with np.load(prefix+'model-'+name+'-'+str(i)+'.npz') as f:
                    values = [f['arr_%d' % n] for n in range(len(f.files))]
                params = lasagne.layers.get_all_params(layers[i])
                for layer in newlayers:
                    newparams = []
                    for t in params:
                        #if t != layer.W and t!=layer.b:
                        #    newparams+=[t]
                        if not t in layer.get_params():
                            newparams+=[t]
                    params=newparams
                #params=[]
                #for j in range(len(layers[i])):
                    #a=[x for x in layers[i][j].get_params() and x not in params]
                    #params+=a
                if partial:
                    values=values[:len(params)]
                mismatch+=set_param_value(params,values,ignore_mismatch=ignore_mismatch)
    if os.path.exists(prefix+'model-global-'+str(0)+'.npz'):
        if not os.path.exists(prefix+'model-global-'+str(0)+'.skip'):
            with np.load(prefix+'model-global-'+str(0)+'.npz') as f:
                values = [f['arr_%d' % n] for n in range(len(f.files))]
            epoch = values[0]
            values = values[1:]
            params=global_params
            if partial:
                values=values[:len(params)]
            mismatch+=set_param_value(params,values,ignore_mismatch=ignore_mismatch)
    return epoch,mismatch#}}}


#from join import join_layer as JoinLayer
#from join import copy_batch_norm as CopyBatchMorm 

#def get_errors(groups,m=None,prefix=''):#
#    res=[]
#    #print 'DEBUG'
#    for t in groups['errors']:
#        if m is None:
#            res+=[[prefix+t,map(lasagne.layers.get_output,groups['errors'][t])]]
#        else:
#            tmp=map(lambda x:JoinLayer(x,m),groups['errors'][t])
#            res+=[[prefix+t,map(lasagne.layers.get_output,tmp)]]
#            #print [[t,map(lasagne.layers.get_output_shape,groups['errors'][t])]]
#    return sorted(res,key=lambda x:x[0])#
#def get_watchpoints(groups,m=None,prefix=''):#
#    res=[]
#    #print 'DEBUG'
#    for t in groups['watchpoints']:
#        if m is None:
#            res+=[[prefix+t,map(lasagne.layers.get_output,groups['watchpoints'][t])]]
#        else:
#            tmp=map(lambda x:JoinLayer(x,m),groups['watchpoints'][t])
#            res+=[[prefix+t,map(lasagne.layers.get_output,tmp)]]
#            #print [[t,map(lasagne.layers.get_output_shape,groups['watchpoints'][t])]]
#    return sorted(res,key=lambda x:x[0])#

#class Seq:#
#    def __init__(self,key,start=0):
#        self.key=key
#        self.p=start
#    def next(self):
#        p=self.p
#        self.p+=1
#        return self.key+str(p)#
#def sharegroup_replace(m,l):#
#    res=()
#    for a in l:
#        if a[-2]==0:
#            res+=(a,)
#        else:
#            res+=(a[:-2]+(m[a[-2]].next(),a[-1],),)
#    return res#

def create_layers_dict(conv_layers):#{{{
    res=OrderedDict()
    for i,t in enumerate(conv_layers):
        res['layer_'+str(i)]=t
    return res#}}}

#def handle_finish(conv_groups,m):
#    conv_groups['predict']=[JoinLayer(conv_groups['output'][0],{ 
#        conv_groups['freedim'][0]:conv_groups['best_freedim'][0],
#        })]
#
#def build_network(inputs):
#
#    source_image_network=inputs['source_image']
#    action_network=inputs['action']
#    target_image_network=inputs['target_image']
#
#    F=64
#
#    sq1=Seq('conv')
#    sq2=Seq('conv')
#    sq3=Seq('deconv')
#    sq4=Seq('deconv')
#    sq5=Seq('deconv')
#    
#    network,conv_groups,conv_layers = build_convdeconv_network(source_image_network,sharegroup_replace({1:sq1,2:sq2,3:sq3,4:sq4,5:sq5},(
#        ## source
#        ('source_image',0   ,0, 0,0,1,{'noise':1.0/256}),
#        (0,8   ,5, 1,0,1,{}),#{{{
#        (0,8   ,3, 1,0,1,{}),
#        (0,F   ,4, 4,0,1,{}),
#        (0,F   ,3, 1,0,1,{}),
#        (0,F   ,3, 1,0,1,{}),
#        (0,F   ,4, 4,0,1,{}),
#        (0,F   ,3, 1,0,1,{}),
#        (0,F   ,3, 1,0,1,{}),
#        (0,F *9,4, 4,0,1,{}),#}}}
#        (0,(16,6,6),0, 0,0,1,{}),#{{{
#        (0,16   ,1, 1,0,1,{}),
#        (0,16   ,1, 1,0,1,{}),
#        (0,F *9,6, 6,0,1,{}),#}}}
#        (0,(16,6,6),0, 0,0,1,{}),#{{{
#        (0,16   ,1, 1,0,1,{}),
#        (0,16   ,1, 1,0,1,{}),
#        (0,F *9,6, 6,0,1,{}),#}}}
#        (0,(16,6,6),0, 0,0,1,{}),#{{{
#        (0,16   ,1, 1,0,1,{}),
#        (0,16   ,1, 1,0,1,{}),
#        (0,F *9,6, 6,0,1,{}),#}}}
#        (0,0    ,0, 0,'source',1,{'noise':1.0/256}),
#        (0,(16,6,6),0, 0,0,3,{}),#{{{
#        (0,16   ,1, 1,0,3,{}),
#        (0,16   ,1, 1,0,3,{}),
#        (0,F *9,6, 6,0,3,{}),#}}}
#        (0,(16,6,6),0, 0,0,3,{}),#{{{
#        (0,16   ,1, 1,0,3,{}),
#        (0,16   ,1, 1,0,3,{}),
#        (0,F *9,6, 6,0,3,{}),#}}}
#        (0,(16,6,6),0, 0,0,3,{}),#{{{
#        (0,16   ,1, 1,0,3,{}),
#        (0,16   ,1, 1,0,3,{}),
#        (0,F *9,6, 6,0,3,{}),#}}}
#        (0,(F,3,3),0,0,0,3,{}),
#        (0,F   ,3,-4,0,3,{'nopad'}), #{{{
#        (0,F   ,3, 1,0,3,{}),
#        (0,F   ,3, 1,0,3,{}),
#        (0,F   ,3,-4,0,3,{}),
#        (0,F   ,3, 1,0,3,{}),
#        (0,F   ,3, 1,0,3,{}),
#        (0,8   ,3,-4,0,3,{}),
#        (0,8   ,3, 1,0,3,{}),
#        (0,3   ,5, 1,'source_image_recon',3,{'equal':['source_image','source_image_recon',mylossfunc]}), #}}}
#
#        ## target
#        ('target_image',0   ,0, 0,0,2,{'noise':1.0/256}),#{{{
#        (0,8   ,5, 1,0,2,{}),
#        (0,8   ,3, 1,0,2,{}),
#        (0,F   ,4, 4,0,2,{}),
#        (0,F   ,3, 1,0,2,{}),
#        (0,F   ,3, 1,0,2,{}),
#        (0,F   ,4, 4,0,2,{}),
#        (0,F   ,3, 1,0,2,{}),
#        (0,F   ,3, 1,0,2,{}),
#        (0,F *9,4, 4,0,2,{}),#}}}
#        (0,(16,6,6),0, 0,0,2,{}),#{{{
#        (0,16   ,1, 1,0,2,{}),
#        (0,16   ,1, 1,0,2,{}),
#        (0,F *9,6, 6,0,2,{}),#}}}
#        (0,(16,6,6),0, 0,0,2,{}),#{{{
#        (0,16   ,1, 1,0,2,{}),
#        (0,16   ,1, 1,0,2,{}),
#        (0,F *9,6, 6,0,2,{}),#}}}
#        (0,(16,6,6),0, 0,0,2,{}),#{{{
#        (0,16   ,1, 1,0,2,{}),
#        (0,16   ,1, 1,0,2,{}),
#        (0,F *9,6, 6,0,2,{}),#}}}
#        (0,0    ,0, 0,'target',2,{'noise':1.0/256}),
#        (0,(16,6,6),0, 0,0,4,{}),#{{{
#        (0,16   ,1, 1,0,4,{}),
#        (0,16   ,1, 1,0,4,{}),
#        (0,F *9,6, 6,0,4,{}),#}}}
#        (0,(16,6,6),0, 0,0,4,{}),#{{{
#        (0,16   ,1, 1,0,4,{}),
#        (0,16   ,1, 1,0,4,{}),
#        (0,F *9,6, 6,0,4,{}),#}}}
#        (0,(16,6,6),0, 0,0,4,{}),#{{{
#        (0,16   ,1, 1,0,4,{}),
#        (0,16   ,1, 1,0,4,{}),
#        (0,F *9,6, 6,0,4,{}),#}}}
#        (0,(F,3,3),0,0,0,4,{}),
#        (0,F   ,3,-4,0,4,{'nopad'}),#{{{
#        (0,F   ,3, 1,0,4,{}),
#        (0,F   ,3, 1,0,4,{}),
#        (0,F   ,3,-4,0,4,{}),
#        (0,F   ,3, 1,0,4,{}),
#        (0,F   ,3, 1,0,4,{}),
#        (0,8   ,3,-4,0,4,{}),
#        (0,8   ,3, 1,0,4,{}),
#        (0,3   ,5, 1,'target_image_recon',4,{'equal':['target_image','target_image_recon',mylossfunc]} if one_direction else {}), #}}}
#
#        ## freedim
#        (('target','source'),0,0,0,0,0,{'sub'}),
#        (0,0,   0, 0,'freedim',0,{'relu':lambda x:T.eq(abs(x),T.max(abs(x),axis=(1,),keepdims=True))}),
#        ((0,'source'),F *9,1, 1,0,0,{}),
#        (0,(16,6,6),0, 0,0,0,{}),#{{{
#        (0,16   ,1, 1,0,0,{}),
#        (0,16   ,1, 1,0,0,{}),
#        (0,F *9,6, 6,0,0,{}),#}}}
#        (0,(16,6,6),0, 0,0,0,{}),#{{{
#        (0,16   ,1, 1,0,0,{}),
#        (0,16   ,1, 1,0,0,{}),
#        (0,F *9,6, 6,0,0,{}),#}}}
#        (0,(16,6,6),0, 0,0,0,{}), #'linear'#{{{
#        (0,16   ,1, 1,0,0,{}),
#        (0,16   ,1, 1,0,0,{}),
#        (0,F *9,6, 6,0,0,{'linear'}),#}}}
#        (('source',0),
#                0, 0,0,0,{'add':True,'equal':['target','freedim_enumerate',mylossfunc]}),
#
#        (('source','action'),F*9,1, 1,0,0,{}),
#        (0,(16,6,6),0, 0,0,0,{}),#{{{
#        (0,16   ,1, 1,0,0,{}),
#        (0,16   ,1, 1,0,0,{}),
#        (0,F *9,6, 6,0,0,{}),#}}}
#        (0,(16,6,6),0, 0,0,0,{}),#{{{
#        (0,16   ,1, 1,0,0,{}),
#        (0,16   ,1, 1,0,0,{}),
#        (0,F *9,6, 6,0,0,{}),#}}}
#        (0,(16,6,6),0, 0,0,0,{}), #'freedim_predict'#{{{
#        (0,16   ,1, 1,0,0,{}),
#        (0,16   ,1, 1,0,0,{}),
#        (0,F *9,6, 6,'freedim_predict',0,{'equal':['freedim','freedim_predict',mylossfunc]}),#}}}
#        ('freedim_predict',0,0,0,'best_freedim',0,{'relu':lambda x:T.eq(abs(x),T.max(abs(x),axis=(1,),keepdims=True))}),
#
#        (('source','action','freedim'),F*9,1, 1,0,0,{}), # 预测的时候把 freedim 换成 best_freedim
#        (0,(16,6,6),0, 0,0,0,{}),#{{{
#        (0,16   ,1, 1,0,0,{}),
#        (0,16   ,1, 1,0,0,{}),
#        (0,F *9,6, 6,0,0,{}),#}}}
#        (0,(16,6,6),0, 0,0,0,{}),#{{{
#        (0,16   ,1, 1,0,0,{}),
#        (0,16   ,1, 1,0,0,{}),
#        (0,F *9,6, 6,0,0,{}),#}}}
#        (0,(16,6,6),0, 0,0,0,{}), #'target_predict'#{{{
#        (0,16   ,1, 1,0,0,{}),
#        (0,16   ,1, 1,0,0,{}),
#        (0,F *9,6, 6,'target_predict',0,{'equal':['target','target_predict',mylossfunc]}),#}}}
#        (0,(16,6,6),0, 0,0,5,{}),#{{{
#        (0,16   ,1, 1,0,5,{}),
#        (0,16   ,1, 1,0,5,{}),
#        (0,F *9,6, 6,0,5,{}),#}}}
#        (0,(16,6,6),0, 0,0,5,{}),#{{{
#        (0,16   ,1, 1,0,5,{}),
#        (0,16   ,1, 1,0,5,{}),
#        (0,F *9,6, 6,0,5,{}),#}}}
#        (0,(16,6,6),0, 0,0,5,{}),#{{{
#        (0,16   ,1, 1,0,5,{}),
#        (0,16   ,1, 1,0,5,{}),
#        (0,F *9,6, 6,0,5,{}),#}}}
#        (0,(F,3,3),0,0,0,5,{}),
#        (0,F   ,3,-4,0,5,{'nopad'}), #{{{
#        (0,F   ,3, 1,0,5,{}),
#        (0,F   ,3, 1,0,5,{}),
#        (0,F   ,3,-4,0,5,{}),
#        (0,F   ,3, 1,0,5,{}),
#        (0,F   ,3, 1,0,5,{}),
#        (0,8   ,3,-4,0,5,{}),
#        (0,8   ,3, 1,0,5,{}),
#        (0,3   ,5, 1,0,5,{'watch':['target_image','train:predict_recon',mylossfunc]}), #}}}
#        )),{
#                'action':action_network,
#                'source_image':source_image_network,
#                'target_image':target_image_network,
#                },relu=lasagne.nonlinearities.leaky_rectify,init=lasagne.init.HeUniform,autoscale=False,finish=handle_finish)
#
#    assert sq1.next()==sq2.next()
#    assert len(set([sq3.next(),sq4.next(),sq5.next()]))==1
#
#    res=create_layers_dict(conv_layers)
#    res['action']=action_network
#    res['source_image']=source_image_network
#    res['target_image']=target_image_network
#    errors = get_errors(conv_groups)+[
#            ['example_errors',[]],
#            ]
#    val_watch_errors = get_watchpoints(conv_groups)+[ 
#            ]
#    return [res],errors,val_watch_errors,conv_groups

network_builder=None

def register_network_builder(build_network):
    global network_builder
    network_builder=build_network

inference_handler=None

def register_inference_handler(h):
    global inference_handler 
    inference_handler=h

model_handlers=[]

def register_model_handler(h):
    global model_handlers
    model_handlers+=[h]


#quit_flag=False

#frames=270
#rl_dummy=16
#def load_200():
#    if frames==90:
#        npyfile=('200x64x'+str(frames)+'.npy',)
#    else:
#        npyfile=('200x64x'+str(frames)+'-0.npy',
#                '200x64x'+str(frames)+'-1.npy')
#    if not os.path.exists(npyfile[0]):
#        #aa=[] #(frames, 3, 256, 256)*16
#        base=0
#        for dir in ['turnleft','turnright','up','down']:
#            for i in range(200):
#                a=np.load('200x64x'+str(frames)+'/'+dir+'/'+dir+'-'+str(i)+'.npz')['arr_0']
#                writenpy(npyfile,(3,64,64),np.arange(len(a))+base,a,fast=True)
#                base+=len(a)
#    return curry(readnpy,npyfile,(3,64,64))

#minibatch_handlers=[]
#def register_minibatch_handler(h):
#    global minibatch_handlers
#    minibatch_handlers+=[h]

#centralize=False
#one_direction=False
#
#def iterate_minibatches_200(aa,stage,batchsize,iteratesize=400, shuffle=False, idx=False):
#    rangeframes=frames/90*10
#    unitframes=frames/90*10
#    i=0
#    last=None
#    batchsize0=batchsize
#    while i<iteratesize:
#        if stage=='train':
#            if last is not None:
#                batchsize=batchsize0#/2
#            else:
#                batchsize=batchsize0
#        else:
#            batchsize=batchsize0
#        if stage=='train':
#            actions1=np.zeros((batchsize,handledata.num_curr_actions,1,1),dtype=floatX)
#            #actions2=np.zeros((batchsize,handledata.num_curr_actions,1,1),dtype=floatX)
#            k=(np.random.rand(batchsize)*800).astype('int')
#            beginpos=rangeframes+(np.random.rand(batchsize)*(frames-rangeframes*2)).astype('int')
#            if centralize:
#                k[::2]=k[0]
#                beginpos[::2]=beginpos[0]
#            action1=(np.random.rand(batchsize)*rangeframes).astype('int')
#            #action2=(np.random.rand(batchsize)*rangeframes).astype('int')
#            #endpos=beginpos+action2
#            beforpos=beginpos-action1
#            faction1=action1.astype(floatX)/unitframes
#            #faction2=action2.astype(floatX)/unitframes
#            isleft=(k<200)
#            isright=(k>=200)*(k<400)
#            isup=(k>=400)*(k<600)
#            isdown=(k>=600)
#            actions1[:,0,0,0]=isleft*faction1
#            #actions2[:,0,0,0]=isleft*faction2
#            actions1[:,1,0,0]=isright*faction1
#            #actions2[:,1,0,0]=isright*faction2
#            actions1[:,2,0,0]=isup*faction1
#            #actions2[:,2,0,0]=isup*faction2
#            actions1[:,3,0,0]=isdown*faction1
#            #actions2[:,3,0,0]=isdown*faction2
#            assert idx==False
#            actions1=np.concatenate((actions1[:,0:4],
#                np.zeros((batchsize,6,1,1),dtype=floatX)
#                ),axis=1)
#            #actions2=np.concatenate((actions2[:,0:4],
#            #    np.zeros((batchsize,6,1,1),dtype=floatX)
#            #    ),axis=1)
#        else:
#            #actions1=np.zeros((batchsize,handledata.num_curr_actions,1,1),dtype=floatX)
#            actions2=np.zeros((batchsize,handledata.num_curr_actions,1,1),dtype=floatX)
#            k=(np.random.rand(batchsize)*800).astype('int')
#            #action1=(np.random.rand(batchsize)*rangeframes).astype('int')
#            action2=(np.random.rand(batchsize)*rangeframes).astype('int')
#            #beginpos=rangeframes+(np.random.rand(batchsize)*(frames-rangeframes*2)).astype('int')
#            endpos=frames-1-(np.random.rand(batchsize)*rangeframes).astype('int')
#            beginpos=endpos-action2
#            #endpos=beginpos+action2
#            #beforpos=beginpos-action1
#            #faction1=action1.astype(floatX)/unitframes
#            faction2=action2.astype(floatX)/unitframes
#            isleft=(k<200)
#            isright=(k>=200)*(k<400)
#            isup=(k>=400)*(k<600)
#            isdown=(k>=600)
#            #actions1[:,0,0,0]=isleft*faction1
#            actions2[:,0,0,0]=isleft*faction2
#            #actions1[:,1,0,0]=isright*faction1
#            actions2[:,1,0,0]=isright*faction2
#            #actions1[:,2,0,0]=isup*faction1
#            actions2[:,2,0,0]=isup*faction2
#            #actions1[:,3,0,0]=isdown*faction1
#            actions2[:,3,0,0]=isdown*faction2
#            assert idx==False
#            #actions1=np.concatenate((actions1[:,0:4],
#            #    np.zeros((batchsize,6,1,1),dtype=floatX)
#            #    ),axis=1)
#            actions2=np.concatenate((actions2[:,0:4],
#                np.zeros((batchsize,6,1,1),dtype=floatX)
#                ),axis=1)
#
#        if stage=='train':
#            if is_autoencoder:
#                batch = ((aa(k*frames+beforpos)/256.0).astype(floatX),None,actions1,"(aa(k*frames+beginpos)/256.0).astype(floatX)",None,"actions2","(aa(k*frames+endpos)/256.0).astype(floatX)",None,None,None,None)
#                images1, ig1, actions1, images2, ig2, actions2, images3, ig3, rewards, targets, flags = batch
#                images2=images1
#                actions1=np.zeros_like(actions1)
#            else:
#                batch = ((aa(k*frames+beforpos)/256.0).astype(floatX),None,actions1,(aa(k*frames+beginpos)/256.0).astype(floatX),None,"actions2","(aa(k*frames+endpos)/256.0).astype(floatX)",None,None,None,None)
#                images1, ig1, actions1, images2, ig2, actions2, images3, ig3, rewards, targets, flags = batch
#            idx1 = create_fingerprint(k*frames+beforpos,32)
#            idx2 = create_fingerprint(k*frames+beginpos,32)
#            if images1.shape[2]==256:
#                images1=images1[:,:,::4,::4]
#                images2=images2[:,:,::4,::4]
#                #images3=images3[:,:,::4,::4]
#
#            action_samples=build_action_sample(rl_dummy,4,zero=True)
#
#            #if sigma_const is not None:
#            #    sigma=np.ones_like(sigma)*sigma_const
#
#            inputs=images1
#            outputs=images2
#            actions=actions1
#
#            #actions[::8]=np.zeros_like(actions[::8])
#
#            drdscxcy=np.zeros((batchsize,4,1,1),dtype=floatX)
#            dxdy=np.zeros((batchsize,2,1,1),dtype=floatX)
#
#            actions=np.concatenate((actions[:,0:4],
#                drdscxcy[:,2:4].reshape(batchsize,2,1,1),
#                dxdy.reshape(batchsize,2,1,1),
#                drdscxcy[:,0:2].reshape(batchsize,2,1,1),
#                ),axis=1)
#            samples=np.concatenate((action_samples,np.zeros((rl_dummy,num_extra_actions,1,1),dtype=floatX)),axis=1)
#
#            if last is None:
#                for j in range(batchsize):
#                    if not one_direction:
#                        if random.random()*2.0<=1.0:
#                            actions[j]=-actions[j]
#                            tmp=inputs[j]
#                            inputs[j]=outputs[j]
#                            outputs[j]=tmp
#
#                X={
#                        'source_image':inputs,
#                        'target_image':outputs,
#                        'action':actions,
#                        }
#                if using_fingerprint:
#                    X['source_fingerprint']=idx1
#                    X['target_fingerprint']=idx2
#                for h in minibatch_handlers:
#                    h(X)
#            else:
#                inputs=np.concatenate((last['source_image'][batchsize:],inputs),axis=0)
#                outputs=np.concatenate((last['target_image'][batchsize:],outputs),axis=0)
#                actions=np.concatenate((last['action'][batchsize:],actions),axis=0)
#
#                for j in range(batchsize):
#                    if not one_direction:
#                        if random.random()*2.0<=1.0:
#                            actions[j]=-actions[j]
#                            tmp=inputs[j]
#                            inputs[j]=outputs[j]
#                            outputs[j]=tmp
#
#                X={
#                        'source_image':inputs,
#                        'target_image':outputs,
#                        'action':actions,
#                        }
#                if using_fingerprint:
#                    X['source_fingerprint']=idx1
#                    X['target_fingerprint']=idx2
#                for h in minibatch_handlers:
#                    h(X)
#            last=X
#
#            yield X,X['target_image']
#        else:
#            X={
#                    'source_image':(aa(k*frames+beginpos)/256.0).astype(floatX),
#                    'target_image':(aa(k*frames+endpos)/256.0).astype(floatX),
#                    'action':actions2,
#                    }
#            idx1 = create_fingerprint(k*frames+beginpos,32)
#            idx2 = create_fingerprint(k*frames+endpos,32)
#            if using_fingerprint:
#                X['source_fingerprint']=idx1
#                X['target_fingerprint']=idx2
#            for h in minibatch_handlers:
#                h(X)
#            yield X,X['target_image']
#        i+=1

def random_shift(n,*all_inputs):
    actions=(np.random.rand(len(all_inputs[0]),2,1,1)).astype(floatX)
    actions2=(actions*n*2).astype('int8')-n
    all_outputs=[]
    for inputs in all_inputs:
        outputs=np.zeros(inputs.shape,dtype=floatX)
        for i in range(len(inputs)):
            tmp=np.pad(inputs[i:i+1],((0,0),(0,0),(n,n),(n,n)),mode='constant',constant_values=0)
            tmp=np.roll(tmp,actions2[i,0,0,0],2)
            tmp=np.roll(tmp,actions2[i,1,0,0],3)
            if n>0:
                outputs[i:i+1]=tmp[:,:,n:-n,n:-n]
            else:
                outputs[i:i+1]=tmp
        all_outputs+=[outputs]
    return all_outputs+[actions2.reshape(len(inputs),2)]

#def random_rotate(w,h,angle,scale,*all_inputs):
#    cx=(np.random.rand(len(all_inputs[0])).astype(floatX))*w
#    cy=(np.random.rand(len(all_inputs[0])).astype(floatX))*h
#    actions=(np.random.rand(len(all_inputs[0]),4,1,1)).astype(floatX)
#    actions2=np.zeros_like(actions)
#    actions2[:,0]=(actions[:,0]*angle*2-angle).astype(floatX)
#    actions2[:,1]=(actions[:,1]*scale*2-scale).astype(floatX)
#    actions2[:,2,0,0]=cx
#    actions2[:,3,0,0]=cy
#    all_outputs=[]
#    for inputs in all_inputs:
#        outputs=np.zeros(inputs.shape,dtype=floatX)
#        for i in range(len(inputs)):
#            mat = cv2.getRotationMatrix2D((cx[i],cy[i]),actions2[i,0,0,0],1.0+actions2[i,1,0,0])
#            tmp = cv2.warpAffine(inputs[i].transpose(1,2,0),mat,inputs[i].shape[1:]).transpose(2,0,1)
#            #tmp=np.pad(inputs[i:i+1],((0,0),(0,0),(n,n),(n,n)),mode='constant',constant_values=0)
#            #tmp=np.roll(tmp,actions2[i,0,0,0],2)
#            #tmp=np.roll(tmp,actions2[i,1,0,0],3)
#            outputs[i]=tmp
#        all_outputs+=[outputs]
#    return all_outputs+[actions2.reshape(len(inputs),4)]

def random_rotate(w,h,angle,scale,*all_inputs):
    if type(angle)==float:
        angle=(-angle,angle)
    if type(scale)==float:
        scale=(1-scale,1+scale)
    cx=(np.random.rand(len(all_inputs[0])).astype(floatX))*w
    cy=(np.random.rand(len(all_inputs[0])).astype(floatX))*h
    actions=(np.random.rand(len(all_inputs[0]),4,1,1)).astype(floatX)
    actions2=np.zeros_like(actions)
    actions2[:,0]=(actions[:,0]*(angle[1]-angle[0])+angle[0]).astype(floatX)
    actions2[:,1]=(actions[:,1]*(scale[1]-scale[0])+scale[0]).astype(floatX)
    actions2[:,2,0,0]=cx
    actions2[:,3,0,0]=cy
    all_outputs=[]
    for inputs in all_inputs:
        outputs=np.zeros(inputs.shape,dtype=floatX)
        for i in range(len(inputs)):
            mat = cv2.getRotationMatrix2D((cx[i],cy[i]),actions2[i,0,0,0],actions2[i,1,0,0])
            tmp = cv2.warpAffine(inputs[i].transpose(1,2,0),mat,inputs[i].shape[1:]).transpose(2,0,1)
            #tmp=np.pad(inputs[i:i+1],((0,0),(0,0),(n,n),(n,n)),mode='constant',constant_values=0)
            #tmp=np.roll(tmp,actions2[i,0,0,0],2)
            #tmp=np.roll(tmp,actions2[i,1,0,0],3)
            outputs[i]=tmp
        all_outputs+=[outputs]
    return all_outputs+[actions2.reshape(len(inputs),4)]

def list_transpose(a):
    aa=[]
    for i in range(len(a[0])):
        t=[]
        for j in range(len(a)):
            t+=[a[j][i]]
        aa+=[t]
    return aa

#num_extra_actions=4
#
#def enumerate_actions(d):
#    for i in range(d):
#        for j in range(0,30):
#            value=np.zeros((handledata.num_curr_actions+num_extra_actions,1,1),dtype=floatX)
#            value[i,0,0]=j*1.0/30
#            yield value

#def build_action_sample(n,d,zero=False):
#    value=(np.random.rand(n,handledata.num_curr_actions,1,1)*1.0).astype(theano.config.floatX)
#    for j in range(n):
#        k=int(random.random()*d)
#        for i in range(0,handledata.num_curr_actions):
#            if i!=k:
#                value[j,i,0,0]=0
#    if zero:
#        value[0,:,:,:]=np.zeros_like(value[0,:,:,:])
#    return value
#
#def build_freedim_sample(shape):
#    value=np.random.rand(*shape)
#    minval=value.min(axis=1,keepdims=True)
#    value=(value==minval).astype(floatX)
#    return value
#
#def build_freedim_sample_tensor(shape):
#    srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
#    value=np.random.rand(*shape)
#    value=srng.uniform(size=shape)
#    minval=value.min(axis=1,keepdims=True)
#    value=T.eq(value,minval)
#    value=T.set_subtensor(value[0],T.zeros_like(value[0]))
#    return value

#def show(src,norms,predictsloop,predictsloop2,predictsloop3,num_batchsize,t,bottom=None,right=None):
#    t=t%len(predictsloop)
#    w=64
#    h=64
#    xscreenbase=0
#    yscreenbase=0
#    for i in range(num_batchsize):
#        for j in range(len(src)):
#            #j=0
#            imshow64x64("sample-"+str(i)+"-"+str(j),
#                norms[j](src[j][i,0:3,:,:].transpose(1,2,0)))
#            cv2.moveWindow("sample-"+str(i)+"-"+str(j),xscreenbase+j*w,yscreenbase+i*h)
#
#        #j=1
#        #cv2.imshow("sample-"+str(i)+"-"+str(j),
#        #    imnorm(srchide0[i,0:3].transpose(1,2,0)))
#        #cv2.moveWindow("sample-"+str(i)+"-"+str(j),xscreenbase+j*w,yscreenbase+i*h)
#        #j=1
#        #cv2.imshow("sample-"+str(i)+"-"+str(j),
#        #    imnorm(recon[i,0:3].transpose(2,1,0)))
#        #cv2.moveWindow("sample-"+str(i)+"-"+str(j),xscreenbase+j*w,yscreenbase+i*h)
#
#        n=j+1
#
#        #for p in range(1):
#        #    j=p+n
#        #    base=srchide1.shape[1]-1
#        #    cv2.imshow("sample-"+str(i)+"-"+str(j),
#        #        imnorm(enlarge(srchide1[i,base+p:base+p+1].transpose(1,2,0),4)))
#        #    cv2.moveWindow("sample-"+str(i)+"-"+str(j),xscreenbase+j*w,yscreenbase+i*h)
#        #n+=1
#
#
#        for p in range(4):
#            j=p+n
#            imshow64x64("sample-"+str(i)+"-"+str(j),
#                imnorm(predictsloop[t][p][i,0:3].transpose(2,1,0)))
#            cv2.moveWindow("sample-"+str(i)+"-"+str(j),xscreenbase+j*w,yscreenbase+i*h)
#        n+=4
#        for p in range(4): #num_actions+4
#            j=p+n
#            imshow64x64("sample-"+str(i)+"-"+str(j),
#                imnorm(predictsloop2[t][p][i,0:3].transpose(2,1,0)))
#            cv2.moveWindow("sample-"+str(i)+"-"+str(j),xscreenbase+j*w,yscreenbase+i*h)
#        n+=4 #num_actions+4
#        if i>=7:
#            break
#    if bottom is not None:
#        cv2.imshow('bottom',bottom)
#        cv2.moveWindow("bottom",0,64*8)
#    if right is not None:
#        cv2.imshow('right',right)
#        cv2.moveWindow("right",64*10,0)

#ignore_output=False
#def set_ignore_output(val):
#    global ignore_output
#    ignore_output=val
#
#objective_loss_function=mylossfunc
#def set_loss_function(f):
#    global objective_loss_function
#    objective_loss_function=f
#is_classify=False
#def set_classify(val):
#    global is_classify
#    is_classify=val

#def objective(layers,myloss=None,deterministic=False,*args,**kwargs):
#    if not is_autoencoder and not ignore_output:
#        loss = nolearn.lasagne.objective(layers,*args,**kwargs)
#        loss += myloss
#    else:
#        loss = myloss
#    return loss

def myscore(key,val,X,y):
    return val

def make_nolearn_scores(losslist,tagslice):
    res=[]
    for tag,sli in tagslice:
        if len(losslist[sli])>0:
            i=0
            for t in losslist[sli]:
                if len(losslist[sli])>1:
                    res+=[[tag+'-'+str(i),curry(myscore,tag,t)]]
                else:
                    res+=[[tag,curry(myscore,tag,t)]]
                i+=1
    return res

#src=None
#norms=None
#predictsloop=[]
#predictsloop2=[]
#predictsloop3=[]
#bottom=None
#right=None

#sn=0
#def mybatchok(num_batchsize,sigma_base,sigma_var,net,history):
#    #global sn
#
#    sigma=floatXconst(random.random()*sigma_base.get_value())
#    sigma_var.set_value(sigma)
#
#    sys.stdout.write(".")
#    #show(src,norms,predictsloop,predictsloop2,predictsloop3,num_batchsize,sn,bottom=bottom,right=right)
#    #cv2.waitKey(100)
#    #sn+=1

#def plot_loss(net,pltskip):
#    skip=int(pltskip.get_value()+0.5)
#    train_loss = [row['train_loss'] for row in net.train_history_]
#    valid_loss = [row['valid_loss'] for row in net.train_history_]
#    for i in range(skip):
#        if i<len(train_loss):
#            train_loss[i]=None
#        if i<len(valid_loss):
#            valid_loss[i]=None
#    plt.plot(train_loss, label='train loss')
#    plt.plot(valid_loss, label='valid loss')
#    plt.xlabel('epoch')
#    plt.ylabel('loss')
#    plt.legend(loc='best')
#    return plt


#def myepochok():
#    #global bottom,right
#
#    #curr_epoch=epoch_begin+len(history)
#
#    save_params(curr_epoch,[
#        sorted_values(networks) for networks in all_networks
#        ],[],'hideconv-',deletelayers=[])
#    #print ''
#
#    #tr.print_diff()
#
##    easyshared.update()
##
##    fig_size = plt.rcParams["figure.figsize"]
##    fig_size[0] = 10
##    fig_size[1] = 4
##    plt.rcParams["figure.figsize"] = fig_size
##    plt.clf()
##    loss_plt=plot_loss(net,pltskip)
##    loss_plt.savefig('loss.png',dpi=64)
##    #print net.layers_['source']
##    #feature_plt=nolearn.lasagne.visualize.plot_conv_weights(net.layers_['source'],figsize=(6, 6))
##    #feature_plt.savefig('feature.png',dpi=64)
##
##    bottom=cv2.imread('loss.png')
##    #right=cv2.imread('feature.png')
##
##    #myupdate(epoch_begin,num_batchsize,all_networks,predict_fns,walker_fn,net,history)
##
##    if len(history) % 5 == 0:
##        while gc.collect() > 0:
##            pass


#visualize_validation_set=False
#
#def myupdate(epoch_begin,num_batchsize,all_networks,predict_fns,walker_fn,net,history):
#    global src,norms,predictsloop,predictsloop2,predictsloop3
#    predict_fn,visual_fn=predict_fns
#    if visualize_validation_set:
#        iterate_fn=net.batch_iterator_test
#    else:
#        iterate_fn=net.batch_iterator_train
#    src=None
#    norms=None
#    predictsloop=[]
#    predictsloop2=[]
#    predictsloop3=[]
#    it=iterate_fn(None,None)
#    for batch in it:
#        #inputs, inputs2, actions, outputs, outputs2, rewards, targets, flags = batch
#
#        inputs = batch[0]['source_image']
#        actions = batch[0]['action']
#        outputs = batch[0]['target_image']
#
#        #if inputs.shape[2]==256:
#        #    inputs=inputs[:,:,::4,::4]
#        #    outputs=outputs[:,:,::4,::4]
#
#        vis_args = [batch[0][key] for key in visual_varnames]
#        vis = visual_fn(inputs,actions,*vis_args)
#        #srchide0=hides[0]
#        #srchide1=hides[1]
#        #srchide2=hides[2]
#        #srchide0=srchide0.transpose(0,1,3,2)
#        #srchide1=srchide1.transpose(0,1,3,2)
#        #srchide2=srchide2.transpose(0,1,3,2)
#
#        #recon=recon_fn(inputs,
#        #        np.concatenate((actions,np.zeros((num_batchsize,2,1,1),dtype=floatX)),axis=1),
#        #        outputs)
#
#        batchsize,_,ig1,ig2=actions.shape
#        num_actions=handledata.num_curr_actions
#
#        p=[inputs]*(num_actions+4)
#        for t in range(5):
#            #sources=[] 
#            predicts=[] 
#            predicts2=[] 
#            for i in range(num_actions+4):
#
#                if t>0:
#                    actions1=np.concatenate((np.eye(num_actions+4,dtype=floatX)[i:i+1],)*batchsize,axis=0).reshape(batchsize,num_actions+4,1,1)*(0.1)
#                else:
#                    actions1=np.concatenate((np.eye(num_actions+4,dtype=floatX)[i:i+1],)*batchsize,axis=0).reshape(batchsize,num_actions+4,1,1)*(0.0)
#                    #print 'predict actions',actions1
#                if not using_fingerprint:
#                    predict=predict_fn(
#                        p[i],
#                        actions1,
#                        )
#                else:
#                    predict=predict_fn(
#                        p[i],
#                        batch[0]['source_fingerprint'], #XXX
#                        outputs,#XXX
#                        batch[0]['target_fingerprint'], #XXX
#                        actions1,
#                        )
#                predicts+=[predict]
#
#                actions2=np.concatenate((np.eye(num_actions+4,dtype=floatX)[i:i+1],)*batchsize,axis=0).reshape(batchsize,num_actions+4,1,1)*(0.1*t)
#                if not using_fingerprint:
#                    predict=predict_fn(
#                        inputs,
#                        actions2,
#                        )
#                else:
#                    predict=predict_fn(
#                        inputs,
#                        batch[0]['source_fingerprint'],
#                        outputs,
#                        batch[0]['target_fingerprint'],
#                        actions2,
#                        )
#                predicts2+=[predict]
#            p=predicts
#            predictsloop+=[predicts]
#            #predictsloop+=[map(deconv3_fn,predicts)]
#            predictsloop2+=[predicts2]
#
#        #tmp=list_transpose(sources) # (batchsize,actions,[features,width,height])
#        #for j in range(8):
#        #    print [(x**2).sum()**0.5 for x in tmp[j][0:4]]
#        #print np. asarray(bn_std)
#        #src=[inputs.transpose(0,1,3,2),np.roll(inputs.transpose(0,1,3,2),1,axis=0)]
#        src=[inputs.transpose(0,1,3,2),outputs.transpose(0,1,3,2)]+[t.transpose(0,1,3,2) for t in vis]
#        norms=[imnorm]*len(src)
#
#        #print action
#        for t in range(len(predictsloop)):
#            show(src,norms,predictsloop,predictsloop2,predictsloop3,num_batchsize,t)
#            if 0xFF & cv2.waitKey(100) == 27:
#                break_flag = True
#                break
#
#        it.close()
#        break

#class PrintLayerInfo:
#    def __init__(self):
#        pass
#
#    def __call__(self, nn, train_history=None):
#        verbose=nn.verbose
#        nn.verbose=2
#        nolearn.lasagne.PrintLayerInfo()(nn,train_history)
#        nn.verbose=verbose
#
#class PrintLog(nolearn.lasagne.PrintLog):
#    def __call__(self, nn, train_history):
#        self.first_iteration=True
#        train_history = [x for x in train_history]
#        train_history[-1]=train_history[-1].copy()
#        info = train_history[-1]
#        for name, func in nn.scores_train:
#            if info[name]<0.001:
#                info[name] = '[{:0.6e}]'.format(info[name])
#        print self.table(nn, train_history)
#        sys.stdout.flush()
#
#class Net(nolearn.lasagne.NeuralNet):
#    def initialize_layers(self, layers=None):
#
#        from nolearn.lasagne.base import Layers
#        from lasagne.layers import get_all_layers
#        from lasagne.layers import get_output
#        from lasagne.layers import InputLayer
#        from lasagne.layers import Layer
#        from lasagne.utils import floatX
#        from lasagne.utils import unique
#
#        if layers is not None:
#            self.layers = layers
#        self.layers_ = Layers()
#
#        assert isinstance(self.layers[0], Layer)
#
#        if isinstance(self.layers[0], Layer):
#            j = 0
#            for out_layer in self.layers:
#                for i, layer in enumerate(get_all_layers(out_layer)):
#                    if layer not in self.layers_.values():
#                        name = layer.name or self._layer_name(layer.__class__, j)
#                        j+=1
#                        if name in self.layers_:
#                            print 'WARNING: ',name,'exists.'
#                        self.layers_[name] = layer
#                        if self._get_params_for(name) != {}:
#                            raise ValueError(
#                                "You can't use keyword params when passing a Lasagne "
#                                "instance object as the 'layers' parameter of "
#                                "'NeuralNet'."
#                                )
#            return self.layers[-1]

#visual_keys=[]
#visual_vars=[]
#visual_varnames=[]
#def register_visual(key):
#    global visual_keys
#    visual_keys+=[key]
#def register_visual_var(name,var):
#    global visual_vars
#    global visual_varnames
#    visual_vars+=[var]
#    visual_varnames+=[name]

batch_iterator_train=None
batch_iterator_test=None
batch_iterator_inference=None
def register_batch_iterator(train,test,inference=None):
    global batch_iterator_train,batch_iterator_test,batch_iterator_inference
    batch_iterator_train,batch_iterator_test,batch_iterator_inference=train,test,inference
#def wrap_batch_iterator_train(num_batchsize,igx,igy):
#    return batch_iterator_train(num_batchsize)
#def wrap_batch_iterator_test(num_batchsize,igx,igy):
#    return batch_iterator_test(num_batchsize)

def layers(l):
    return macros(l)

def newlayers(l):
    res=()
    l=macros(l)
    i=0   
    for a in l:
        assert a[-2]==0
        m=dict(a[-1].copy())
        m['saveparamlayer']='newlayer'
        res+=(a[:-1]+(m,),)
    return res

def deletelayers(l):
    res=()
    l=macros(l)
    i=0   
    for a in l:
        m=dict(a[-1].copy())
        m['saveparamlayer']='deletelayer'
        res+=(a[:-1]+(m,),)
    return res

on_batch_finished=[]
on_epoch_finished=[]
on_training_started=[]
on_training_finished=[]

def register_training_callbacks(bf,ef,ts,tf):
    global on_batch_finished, on_epoch_finished, on_training_started, on_training_finished
    on_batch_finished+=bf
    on_epoch_finished+=ef
    on_training_started+=ts
    on_training_finished+=tf

on_validation_started=[]
on_validation_finished=[]
def register_validation_callbacks(vs,vf):
    global on_validation_started,on_validation_finished
    on_validation_started+=vs
    on_validation_finished+=vf

data_shaker=None

def register_data_shaker(f):
    global data_shaker
    data_shaker=f

train_fn = None
val_fn = None
inference_fn = None

def inference(num_batchsize,inference_db):
    for batch in batch_iterator_inference(num_batchsize,inference_db):
        out = inference_fn(*sorted_values(batch))
        if inference_handler is not None:
            inference_handler(out[0])
    return out[0]

loss_handler=None

def register_loss_handler(h):
    global loss_handler
    loss_handler=h

try:
    import caffe
    from caffe.proto import caffe_pb2
    def load_mean_file(mean_file):
        with open(mean_file, 'rb') as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            #pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            #print pixel.shape
            #t.set_mean('data', pixel)
            mean=np.reshape(blob.data, blob_dims[1:])
            #mean=mean[:,(256-224)//2:(256-224)//2+224,(256-224)//2:(256-224)//2+224]
        return mean
except:
    def load_mean_file(mean_file):
        assert mean_file.endswith('.npy')
        return np.load(mean_file)

lrpolicy = None
def run(args):
    global lrpolicy
    global train_fn,val_fn,inference_fn
    if args.train_db != '':
        mode='training'
    elif args.validation_db != '':
        mode='validation'
    elif args.inference_db != '':
        mode='inference'
    else:
        mode='training'

    print mode

    num_epochs=args.epoch
    num_batchsize=args.batch_size
    learning_rate=args.lr_base_rate
    momentum=args.momentum
    grads_clip=1.0
    accumulation=args.accumulation

#    if batch_iterator_train is None:
#        loader=load_200
#        iterate_minibatches=iterate_minibatches_200
#        aa = loader()
#        register_batch_iterator(AsyncIterate(curry(iterate_minibatches,aa,'train',batchsize=num_batchsize,iteratesize=400,shuffle=True)),AsyncIterate(curry(iterate_minibatches,aa,'val',batchsize=num_batchsize,iteratesize=1,shuffle=True)))

    m={}
    dtypes={}
    #print batch_iterator_train
    print args
    print mode
    if mode == 'training':
        it=batch_iterator_train(num_batchsize,args.train_db)
    elif mode == 'validation':
        it=batch_iterator_test(num_batchsize,args.validation_db)
    elif mode == 'inference':
        it=batch_iterator_inference(num_batchsize,args.inference_db)

    for X in it:
        for t in X:
            m[t]=X[t].shape
            dtypes[t]=X[t].dtype
        it.close()
        break

    lr=easyshared.add('lr.txt',learning_rate)
    sigma_base=easyshared.add('sigma.txt',1.0)
    pltskip=easyshared.add('pltskip.txt',0.0)
    decay=easyshared.add('decay.txt',1e-8)

    subtractMean = args.subtractMean #'image', 'pixel' or 'none'
    
    mean_data = load_mean_file(args.mean) if args.mean!='' else None

    if subtractMean == 'pixel' and mean_data is not None:
        pixel = mean_data.mean(axis=(1,2),keepdims=True,dtype=floatX)
        mean_data = np.tile(pixel,(1,mean_data.shape[1],mean_data.shape[2]))

    easyshared.update()
    
    sigma_var=theano.shared(floatXconst(1.0))

    inputs={}
    if mean_data is not None:
        m['mean']=mean_data
    for k in m:
        print k,m[k],dtypes[k]
        name=k
        input_var_type = T.TensorType(dtypes[k],
                [False,]+[s == 1 for s in m[k][1:]])
        var_name = ("%s.input" % name) if name is not None else "input"
        input_var = input_var_type(var_name)
        inputs[k]=lasagne.layers.InputLayer(name=name,input_var=input_var,shape=m[k])
        #print lasagne.layers.get_output_shape(inputs[k])

#    source_image_network=lasagne.layers.InputLayer(name='source_image',shape=m['source_image'],input_var=source_image_var)
#    target_image_network=lasagne.layers.InputLayer(name='target_image',shape=m['target_image'],input_var=target_image_var)
#    action_network=lasagne.layers.InputLayer(name='action',shape=m['action'],input_var=action_var)

    print("Building model and compiling functions...")
    delta_errors,state_layers,hidestate_layers,delta_layers,delta_predict_networks = [],[],[],[],[]
    zeroarch_networks,zeroarch_bnlayer,watcher_network,updater = None,None,None,None
    network,stacks,layers,raw_errors,raw_watchpoints = network_builder(inputs)
    for h in model_handlers:
        h(inputs,network,stacks,layers,raw_errors,raw_watchpoints)
    
    #all_networks,ordered_errors,ordered_watch_errors,conv_groups = network_builder(inputs)
    all_networks=[create_layers_dict(layers)]
    ordered_errors = get_ordered_errors(raw_errors)
    print ordered_errors
    ordered_val_errors = get_ordered_errors(raw_errors,deterministic=True)
    print ordered_val_errors
    ordered_watch_errors = get_ordered_errors(raw_watchpoints)
    ordered_val_watch_errors = get_ordered_errors(raw_watchpoints,deterministic=True)
    conv_groups = stacks

    errors = []
    val_errors = []
    val_watch_errors = []
    train_watch_errors = []
    tagslice = []
    count = 0
    valtagslice = []
    valcount = 0
    for tag,errs in ordered_errors:
        errors += errs
        tagslice += [[tag,slice(count,count+len(errs))]]
        count += len(errs)
    for tag,errs in ordered_val_errors:
        val_errors += errs
        valtagslice += [[tag,slice(valcount,valcount+len(errs))]]
        valcount += len(errs)
    assert len(val_errors)==len(errors)
    i=0
    for tag,errs in ordered_watch_errors:
        valtag,valerrs=ordered_val_watch_errors[i]
        assert tag==valtag
        assert len(errs)==len(valerrs)
        if tag.startswith('train:'):
            train_watch_errors += errs
            tagslice += [[tag[len('train:'):],slice(count,count+len(errs))]]
            count += len(errs)
        elif tag.startswith('val:'):
            val_watch_errors += valerrs
            valtagslice += [[tag[len('val:'):],slice(valcount,valcount+len(errs))]]
            valcount += len(errs)
        else:
            val_watch_errors += errs
            valtagslice += [[tag,slice(valcount,valcount+len(errs))]]
            valcount += len(errs)
        i+=1
    errors = [errors]
    val_errors = [val_errors]
    val_watch_errors = [val_watch_errors]
    train_watch_errors = [train_watch_errors]

    has_loading_networks=False
    loading_networks_list=[]
    for networks in all_networks:
        if 'loading_networks' in networks:
            if networks['loading_networks'] is not None:
                has_loading_networks=True
            loading_networks=networks['loading_networks']
            networks.pop('loading_networks')
        else:
            loading_networks=networks
        if loading_networks is not None:
            loading_networks_list+=[loading_networks]

    newlayers = conv_groups['newlayer'] if 'newlayer' in conv_groups else []
    epoch_begin,mismatch=load_params([
        sorted_values(loading_networks) for loading_networks in loading_networks_list
        ],[],os.path.join(args.save,args.snapshotPrefix),ignore_mismatch=True,newlayers=newlayers)
    print 'epoch_begin=',epoch_begin
    if 'deletelayer' in conv_groups:
        deletelayers = conv_groups['deletelayer']
        save_params(epoch_begin,[
            sorted_values(networks) for networks in all_networks
            ],[],os.path.join(args.save,args.snapshotPrefix),deletelayers=deletelayers)
        print 'layer(s) deleted.'
        exit(0)
    if has_loading_networks:
        save_params(epoch_begin,[
            sorted_values(networks) for networks in all_networks
            ],[],os.path.join(args.save,args.snapshotPrefix))
        print 'save.'
        exit(0)

    if updater is not None:
        updater()
    params = lasagne.layers.get_all_params(sum([networks.values() for networks in all_networks],[]), trainable=True)

    loss = 0.0
    valloss = 0.0
    losslist = []
    vallosslist = []
    tmp = 0.0
    for ee in errors:
        for err in ee:
            if err!=None:
                tmp = err.mean(dtype=floatX)
                losslist = losslist+[tmp]
                loss = loss+tmp
    for ee in val_errors:
        for err in ee:
            if err!=None:
                tmp = err.mean(dtype=floatX)
                vallosslist = vallosslist+[tmp]
                valloss = valloss+tmp
    for ee in val_watch_errors:
        for err in ee:
            if err!=None:
                tmp = err.mean(dtype=floatX)
                vallosslist = vallosslist+[tmp]
    for ee in train_watch_errors:
        for err in ee:
            if err!=None:
                tmp = err.mean(dtype=floatX)
                losslist = losslist+[tmp]

    print 'count_params:',sum([lasagne.layers.count_params(networks.values(),trainable=True) for networks in all_networks],0)

    loss0 = loss
    if loss_handler is not None:
        loss = loss_handler(loss,sum([networks.values() for networks in all_networks],[]))

    extra_loss = loss - loss0

    # l2_penalty 的权重选择：先计算平均，然后在理想的现象数上平摊，因为每个
    # 现象提供 64*64*3 个方程，所以我们只需要 700 多个“理想的”现象，就可以
    # 确定所有的参数，就在这 700 多个现象上平摊

    #l2_penalty = lasagne.regularization.regularize_network_params(networks.values(),lasagne.regularization.l1)/floatXconst(2.0)
    #loss = loss+l2_penalty/(lasagne.layers.count_params(networks.values(),regularizable=True))/(lasagne.layers.count_params(networks.values(),trainable=True)/(64*64*3))

    

#    walker_fn = None
#
#    if 'predict' in conv_groups:
#        key='predict'
#    else:
#        key='output'
#    if not using_fingerprint:
#        predict_fn = theano.function([inputs['source_image'].input_var,inputs['action'].input_var], 
#            lasagne.layers.get_output(conv_groups[key][0],deterministic=True),
#            on_unused_input='warn', allow_input_downcast=True)
#    else:
#        predict_fn = theano.function([
#            inputs['source_image'].input_var,
#            inputs['source_fingerprint'].input_var,
#            inputs['target_image'].input_var, #XXX
#            inputs['target_fingerprint'].input_var, #XXX
#            inputs['action'].input_var], 
#            lasagne.layers.get_output(conv_groups[key][0],deterministic=True),
#            on_unused_input='warn', allow_input_downcast=True)
#    visual_fn = theano.function([inputs['source_image'].input_var,inputs['action'].input_var]+visual_vars, 
#        [lasagne.layers.get_output(conv_groups[key][0],deterministic=True) for key in visual_keys],
#        on_unused_input='warn', allow_input_downcast=True)

    #predict2_fn = theano.function([source_image_var,target_image_var,action_var], 
    #    lasagne.layers.get_output(conv_groups['output'][0],deterministic=True),
    #    on_unused_input='warn', allow_input_downcast=True)

    gc.collect()

    if updater is not None:
        updater()
    layers=list(set(sum([networks.values() for networks in all_networks],[]))-{conv_groups['output'][0]})+[conv_groups['output'][0]]

    conv_groups=None

#    if using_nolearn:
#        net = Net(
#                layers=layers,
#                update=crossbatch_momentum.adamax,
#                update_learning_rate=learning_rate,
#                update_average=accumulation,
#                update_grads_clip=grads_clip,
#                update_noise=False,
#                objective=objective,
#                objective_loss_function=objective_loss_function,
#                objective_myloss=loss,
#                scores_train=make_nolearn_scores(losslist,tagslice),
#                scores_valid=make_nolearn_scores(vallosslist,valtagslice),
#                y_tensor_type=y_tensor_type,
#                train_split=lambda X,y,net:(X,X,y,y),
#                verbose=0,
#                regression=not is_classify,
#                batch_iterator_train=curry(wrap_batch_iterator_train,num_batchsize),
#                batch_iterator_test=curry(wrap_batch_iterator_test,num_batchsize),
#                check_input=False,
#                on_batch_finished=[
#                    #curry(mybatchok,num_batchsize,sigma_base,sigma_var)
#                    ]+on_batch_finished,
#                on_epoch_finished=[
#                    curry(myepochok,epoch_begin,num_batchsize,all_networks,easyshared,pltskip),
#                    #curry(myupdate,epoch_begin,num_batchsize,all_networks,[predict_fn,visual_fn],walker_fn)
#                    PrintLog(),
#                    ]+on_epoch_finished,
#                on_training_started=[
#                    #PrintLayerInfo(), #bugy
#                    #curry(myupdate,epoch_begin,num_batchsize,all_networks,[predict_fn,visual_fn],walker_fn)
#                    ]+on_training_started,
#                on_training_finished=[
#                    ]+on_training_finished,
#                max_epochs=num_epochs,
#
#                )
#
#        net.initialize_layers()
#
#        assert net.layers_[-1]==net.layers[-1] # nolearn bug, two layer with same name would cause this
#
#        X0={}
#        for X,y in batch_iterator_train(num_batchsize):
#            for t in X:
#                X0[t]=None
#            break
#
#        net.fit(X0,np.zeros((num_batchsize,),dtype=floatX))
#    else:

    if mode=='inference':
        if 'predict' in stacks:
            key='predict'
        else:
            key='output'
        if inference_fn is None:
            inference_fn = theano.function(
                    map(lambda x:x.input_var,sorted_values(inputs)), 
                    map(lambda x:lasagne.layers.get_output(x,deterministic=True),stacks[key]),
                    on_unused_input='warn', 
                    allow_input_downcast=True,
                    )
        print 'num_batchsize',num_batchsize
        inference(num_batchsize,args.inference_db)
#        for batch in batch_iterator_inference(num_batchsize,args.inference_db):
#            out = inference_fn(*sorted_values(batch))
#            if inference_handler is not None:
#                inference_handler(out[0])
#        return out[0]
    elif mode=='training' or mode=='validation':
        #updates = lasagne.updates.adamax(loss, params, learning_rate=lr)
        updates = adamax(loss, params, learning_rate=lr, grads_clip=grads_clip, average=accumulation)
        if mode=='training':
            if train_fn is None:
                train_fn = theano.function(
                        map(lambda x:x.input_var,sorted_values(inputs)), 
                        [loss,extra_loss]+losslist, 
                        updates=updates, 
                        on_unused_input='warn', 
                        allow_input_downcast=True,
                        )

        if val_fn is None:
            val_fn = theano.function(
                    map(lambda x:x.input_var,sorted_values(inputs)), 
                    [valloss]+vallosslist, 
                    on_unused_input='warn', 
                    allow_input_downcast=True,
                    )

        if mode=='training':
            for h in on_training_started:
                h(locals())

        min_loss=float('inf')
        min_valloss=float('inf')
        # We iterate over epochs:
        for epoch in range(epoch_begin,epoch_begin+num_epochs):

            if lrpolicy is not None:
                lr.set_value(lrpolicy.get_learning_rate(epoch-epoch_begin))

            easyshared.update()

            break_flag = False

            if mode=='training':

                # In each epoch, we do a full pass over the training data:
                train_err = 0
                train_penalty = 0
                train_errlist = None
                train_batches = 0
                start_time = time.time()
                count = 0
                loopcount = 0
                train_it=batch_iterator_train(num_batchsize,args.train_db)

                train_len = 80
                err=None
                while True:
                    stop=False
                    for i in range(train_len):
                        try:
                            batch = next(train_it)
                        except StopIteration:
                            stop=True
                            break

                        if data_shaker is not None:
                            batch = data_shaker(batch)

                        res=train_fn(*sorted_values(batch))
                        err=res[0]
                        penalty=res[1]
                        errlist=res[2:]
                        train_err += err
                        train_penalty += penalty
                        if train_errlist is None:
                            train_errlist=errlist
                        else:
                            for j in range(len(errlist)):
                                train_errlist[j]=train_errlist[j]+errlist[j]

                        train_batches += 1
                        count = count+1

                        for h in on_batch_finished:
                            h(locals())

                        sys.stdout.write(".")

                        #show(src,norms,predictsloop,predictsloop2,predictsloop3,num_batchsize,num_actions,i)
                        if 0xFF & cv2.waitKey(100) == 27:
                            break_flag = True
                            break

                    if lrpolicy is None:
                        total_training_steps = num_epochs*train_batches
                        lrpolicy = lr_policy.LRPolicy(args.lr_policy,
                                                      args.lr_base_rate,
                                                      args.lr_gamma,
                                                      args.lr_power,
                                                      total_training_steps,
                                                      args.lr_stepvalues)
                    print '' 
                    # Then we print the results for this epoch:
                    if not stop:
                        print "Epoch {}:{} of {} took {:.3f}s".format(
                            epoch + 1, loopcount+1, epoch_begin+num_epochs, time.time() - start_time) 
                    else:
                        print "Training {} of {} took {:.3f}s".format(
                            epoch + 1, epoch_begin+num_epochs, time.time() - start_time) 
                    avg_err = train_err / train_batches
                    avg_penalty = train_penalty / train_batches
                    #print "  training loss:\t\t{:.6f}".format(avg_err)
                    print ' ','training loss',':',avg_err
                    print ' ','training penalty',':',avg_penalty
                    tmp = map(lambda x:x/train_batches,train_errlist)
                    for tag,sli in tagslice:
                        if len(tmp[sli])>0:
                            print ' ',tag,':',tmp[sli]

                    #vals = []
                    #for t in lasagne.layers.get_all_params(networks1.values(),regularizable=True):
                    #    val = abs(t.get_value()).max()
                    #    vals += [val]
                    #print 'max |w|:',max(vals)
                    if stop:
                        break
                    loopcount+=1

                if (epoch+1)%args.validation_interval!=0:
                    continue

            # And a full pass over the validation data:
            val_err = 0
            val_errlist = None
            val_batches = 0
            start_time = time.time()
            count = 0
            loopcount = 0
            val_it=batch_iterator_test(num_batchsize,args.validation_db)

            for h in on_validation_started:
                h(locals())
            val_len = 80
            err=None
            while True:
                stop=False
                for i in range(val_len):
                    try:
                        batch = next(val_it)
                    except StopIteration:
                        stop=True
                        break


                    res=val_fn(*sorted_values(batch))
                    err=res[0]
                    errlist=res[1:]
                    val_err += err
                    if val_errlist is None:
                        val_errlist=errlist
                    else:
                        for j in range(len(errlist)):
                            val_errlist[j]=val_errlist[j]+errlist[j]

                    val_batches += 1
                    count = count+1

                    sys.stdout.write("o")

                    #show(src,norms,predictsloop,predictsloop2,predictsloop3,num_batchsize,num_actions,i)
                    if 0xFF & cv2.waitKey(100) == 27:
                        break_flag = True
                        break

                print '' 
                # Then we print the results for this epoch:
                if not stop:
                    print "Validation {}:{} of {} took {:.3f}s".format(
                        epoch + 1, loopcount+1, epoch_begin+num_epochs, time.time() - start_time) 
                else:
                    print "Validation {} of {} took {:.3f}s".format(
                        epoch + 1, epoch_begin+num_epochs, time.time() - start_time) 
                avg_err = val_err / val_batches
                #print "  validation loss:\t\t{:.6f}".format(avg_err)
                print ' ','validation loss',':',avg_err
                tmp = map(lambda x:x/val_batches,val_errlist)
                for tag,sli in valtagslice:
                    if len(tmp[sli])>0:
                        print ' ',tag,':',tmp[sli]

                #vals = []
                #for t in lasagne.layers.get_all_params(networks1.values(),regularizable=True):
                #    val = abs(t.get_value()).max()
                #    vals += [val]
                #print 'max |w|:',max(vals)
                if stop:
                    break
                loopcount+=1

            for h in on_validation_finished:
                h(locals())

            save_params(epoch+1,[
                sorted_values(networks) for networks in all_networks
                ],[],os.path.join(args.save,args.snapshotPrefix),deletelayers=[])
            if mode=='training':
                if train_err / train_batches < min_loss:
                    min_loss = train_err / train_batches
                    print 'New low training loss',':',min_loss
            if val_err / val_batches < min_valloss:
                min_valloss = val_err / val_batches
                print 'New low validation loss',':',min_valloss
            if mode=='training':
                if (epoch+1)%max(1,int(args.snapshotInterval))==0:
                    save_params(epoch+1,[
                        sorted_values(networks) for networks in all_networks
                        ],[],os.path.join(args.save,args.snapshotPrefix+'epoch'+str(epoch+1)+'-'),deletelayers=[])
            if mode=='training':
                for h in on_epoch_finished:
                    h(locals())
            while gc.collect() > 0:
                pass
            #print 'gc'
            #tmp=memory_usage((train_fn, (batch[0]['source_image'],batch[0]['action'],batch[0]['target_image'],)))
            #print np.mean(tmp)
            if break_flag or 0xFF & cv2.waitKey(100) == 27:
                break
        if mode=='training':
            for h in on_training_finished:
                h(locals())

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgumentParser,self).__init__(description='Deepstacks.')
        parser=self

        def define_integer(key,default,desc):
            parser.add_argument('--'+key,type=int,default=default,help=desc)

        def define_string(key,default,desc):
            parser.add_argument('--'+key,type=str,default=default,help=desc)

        def define_float(key,default,desc):
            parser.add_argument('--'+key,type=float,default=float(default),help=desc)

        def define_boolean(key,default,desc):
            parser.add_argument('--'+key,type=bool,default=default,help=desc)

        define_integer('accumulation', 1, """Accumulate gradients over multiple batches.""")

        # Basic model parameters. #float, integer, boolean, string
        define_integer('batch_size', 16, """Number of images to process in a batch""")
        #define_integer(
        #    'croplen', 0, """Crop (x and y). A zero value means no cropping will be applied""")
        define_integer('epoch', 1, """Number of epochs to train, -1 for unbounded""")
        define_string('inference_db', '', """Directory with inference file source""")
        define_integer(
            'validation_interval', 1, """Number of train epochs to complete, to perform one validation""")
        #define_string('labels_list', '', """Text file listing label definitions""")
        define_string('mean', '', """Mean image file""")
        define_float('momentum', '0.9', """Momentum""")  # Not used by DIGITS front-end
        #define_string('network', '', """File containing network (model)""")
        #define_string('networkDirectory', '', """Directory in which network exists""")
        #define_string('optimization', 'sgd', """Optimization method""")
        define_string('save', 'results', """Save directory""")
        #define_integer('seed', 0, """Fixed input seed for repeatable experiments""")
        #define_boolean('shuffle', False, """Shuffle records before training""")
        define_float(
            'snapshotInterval', 1.0,
            """Specifies the training epochs to be completed before taking a snapshot""")
        define_string('snapshotPrefix', '', """Prefix of the weights/snapshots""")
        define_string(
            'subtractMean', 'none',
            """Select mean subtraction method. Possible values are 'image', 'pixel' or 'none'""")
        define_string('train_db', '', """Directory with training file source""")
        #define_string(
        #    'train_labels', '',
        #    """Directory with an optional and seperate labels file source for training""")
        define_string('validation_db', '', """Directory with validation file source""")
        #define_string(
        #    'validation_labels', '',
        #    """Directory with an optional and seperate labels file source for validation""")
        #define_string(
        #    'visualizeModelPath', '', """Constructs the current model for visualization""")
        #define_boolean(
        #    'visualize_inf', False, """Will output weights and activations for an inference job.""")
        #define_string(
        #    'weights', '', """Filename for weights of a model to use for fine-tuning""")

        # @TODO(tzaman): is the bitdepth in line with the DIGITS team?
        #define_integer('bitdepth', 8, """Specifies an image's bitdepth""")

        # @TODO(tzaman); remove torch mentions below
        define_float('lr_base_rate', '0.01', """Learning rate""")
        define_string(
            'lr_policy', 'fixed',
            """Learning rate policy. (fixed, step, exp, inv, multistep, poly, sigmoid)""")
        define_float(
            'lr_gamma', -1,
            """Required to calculate learning rate. Applies to: (step, exp, inv, multistep, sigmoid)""")
        define_float(
            'lr_power', float('Inf'),
            """Required to calculate learning rate. Applies to: (inv, poly)""")
        define_string(
            'lr_stepvalues', '',
            """Required to calculate stepsize of the learning rate. Applies to: (step, multistep, sigmoid).
            For the 'multistep' lr_policy you can input multiple values seperated by commas""")

        ## Tensorflow-unique arguments for DIGITS
        #define_string(
        #    'save_vars', 'all',
        #    """Sets the collection of variables to be saved: 'all' or only 'trainable'.""")
        #define_string('summaries_dir', '', """Directory of Tensorboard Summaries (logdir)""")
        #define_boolean(
        #    'serving_export', False, """Flag for exporting an Tensorflow Serving model""")
        #define_boolean('log_device_placement', False, """Whether to log device placement.""")
        #define_integer(
        #    'log_runtime_stats_per_step', 0,
        #    """Logs runtime statistics for Tensorboard every x steps, defaults to 0 (off).""")
         
        ## Augmentation
        #define_string(
        #    'augFlip', 'none',
        #    """The flip options {none, fliplr, flipud, fliplrud} as randompre-processing augmentation""")
        #define_float(
        #    'augNoise', 0., """The stddev of Noise in AWGN as pre-processing augmentation""")
        #define_float(
        #    'augContrast', 0., """The contrast factor's bounds as sampled from a random-uniform distribution
        #     as pre-processing  augmentation""")
        #define_boolean(
        #    'augWhitening', False, """Performs per-image whitening by subtracting off its own mean and
        #    dividing by its own standard deviation.""")
        #define_float(
        #    'augHSVh', 0., """The stddev of HSV's Hue shift as pre-processing  augmentation""")
        #define_float(
        #    'augHSVs', 0., """The stddev of HSV's Saturation shift as pre-processing  augmentation""")
        #define_float(
        #    'augHSVv', 0., """The stddev of HSV's Value shift as pre-processing augmentation""")


args = None
def main():
    global args
#    parser = argparse.ArgumentParser(description='Trains a neural network to handle 3d scene transforming on actions.')
#    parser.add_argument('mode',metavar='MODE',help="'training/inference'")
#    parser.add_argument('num_epochs',metavar='EPOCHS',type=int,help="number of training epochs to perform")
#    parser.add_argument('num_batchsize',metavar='BATCHSIZE',type=int,help="batchsize")
#    parser.add_argument('learning_rate',metavar='LEARNING_RATE',type=float,help="learning rate")
#    parser.add_argument('accumulation',metavar='ACCUMULATION',type=int,help="batch accumulation")

    parser = ArgumentParser()

    args = parser.parse_args()

    run(args)
    #quit_flag=True
    #time.sleep(5)

