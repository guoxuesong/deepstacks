#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import deepstacks
from deepstacks.framework.main import *
import deepstacks.framework.using_mnist
from deepstacks.macros import *
from deepstacks.framework.macros import *
from deepstacks.lasagne import curr_layer,curr_stacks,curr_flags,curr_model,curr_batchsize

from deepstacks.util.curry import curry


def build_cnn(inputs):
    network=inputs['image']
    if 'mean' in inputs:
        network=lasagne.layers.ElemwiseMergeLayer((network,inputs['mean']),T.sub)
    y=inputs['target']
    return deepstacks.lasagne.build_network(network, prettylayers
        (0,32,5,1,0,0)
        (0,0,2,2,0,0,maxpool=True)
        (0,32,5,1,0,0)
        (0,0,2,2,0,0,maxpool=True)
        (0,0,0,0,0,0,layer=(lasagne.layers.DropoutLayer,curr_layer,{'p':0.5}))
        (0,256,0,0,0,0,dense=True)
        (0,0,0,0,0,0,layer=(lasagne.layers.DropoutLayer,curr_layer,{'p':0.5}))
        (0,10,0,0,0,0,
            dense=True,
            nonlinearity=lasagne.nonlinearities.softmax,
            )
        (classify,'y')
        ,{
            'y':y
            })

register_network_builder(build_cnn)


if __name__ == '__main__':
    main()
