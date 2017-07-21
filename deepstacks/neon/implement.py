#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import neon
import random

from ..stacked import Layers, register_layers_class
from ..stacked import register_concat_handler, register_inputs_handler
from ..stacked import register_flag_handler, register_network_wrapper
from ..stacked import register_macro_handler
from ..stacked import *
from neon.layers.layer import Affine, Activation, Linear
from neon.layers.layer import Bias, BranchNode, GeneralizedCost
from neon.layers.layer import Reshape, Pooling
from neon.layers.container import Sequential, MergeSum, MergeMultistream

network_branch = {}
branch_notfirst = set()


class NeonLayers(Layers):
    def get_layer(self, k):
        layer = super(NeonLayers, self).get_layer(k)

        if layer in network_branch:
            print 'Found branch', layer
            b = network_branch[layer]
            if b in branch_notfirst:
                layer = b
            else:
                branch_notfirst.add(b)
        return layer
register_layers_class(NeonLayers)


def sequential(layers):
    a = ()
    for t in layers:
        if type(t) == Sequential:
            a += tuple(t.layers)
        else:
            a += (t, )
    return Sequential(layers=a)


def concat_handler(layers, flags, stacks, this_model):
    return MergeMultistream(layers=layers, merge="depth")


def merge_handler(layers, flags, stacks, this_model):
    raise NotImplementedError


def add_handler(layers, flags, stacks, this_model):
    return MergeSum(layers)


def sub_handler(layers, flags, stacks, this_model):
    if len(layers) > 2:
        left = layers[0]
        right = sequential(
                layers=(
                    MergeSum(layers[1:]),
                    Activation(neon.transforms.Normalizer(divisor=-1))))
        network = MergeSum(layers=(left, right))
    elif len(layers) == 2:
        left = layers[0]
        right = sequential(
                layers=(
                    layers[1],
                    Activation(neon.transforms.Normalizer(divisor=-1))))
        network = MergeSum(layers=(left, right))
    else:
        network = layers[0]
    return network

register_concat_handler(concat_handler)
register_inputs_handler('op', merge_handler)
register_inputs_handler('add', add_handler)
register_inputs_handler('sub', sub_handler)


def reshape_handler(network, flags, stacks, this_model):
    network = sequential(layers=(network, Reshape(reshape=flags['reshape'])))
    return network, ()


def slice_handler(network, flags, stacks, this_model):
    raise NotImplementedError


def maxpool_handler(network, flags, stacks, this_model):
    # num_filters=flags['num_filters']
    conv_stride = flags['stride'] if 'stride' in flags else 0

    layername = flags['layername'] if 'layername' in flags else None
    filter_size = flags['filter_size'] if 'filter_size' in flags else 0

    # dim=len(lasagne.layers.get_output_shape(network))-2 #XXX
    # dim=2
    assert filter_size > 0
    network = sequential(layers=(network, Pooling(
        fshape=filter_size,
        stride=max(1, conv_stride),
        pad=0,
        op='max',
        name=layername,
        )))
    return network, ()


def meanpool_handler(network, flags, stacks, this_model):
    # num_filters=flags['num_filters']
    conv_stride = flags['stride'] if 'stride' in flags else 0

    layername = flags['layername'] if 'layername' in flags else None
    filter_size = flags['filter_size'] if 'filter_size' in flags else 0

    # dim=len(lasagne.layers.get_output_shape(network))-2 #XXX
    # dim=2
    assert filter_size > 0
    network = sequential(layers=(network, Pooling(
        fshape=filter_size,
        stride=max(1, conv_stride),
        pad=0,
        op='avg',
        name=layername,
        )))
    return network, ()


def upscale_handler(network, flags, stacks, this_model):
    raise NotImplementedError


def num_filters_handler(network, flags, stacks, this_model):
    paramlayers = []
    if 'sharegroup2params' not in this_model:
        this_model['sharegroup2params'] = {}
    sharegroup2params = this_model['sharegroup2params']

    if 'layer2sharegroup' not in this_model:
        this_model['layer2sharegroup'] = {}
    layer2sharegroup = this_model['layer2sharegroup']
    if 'constlayer2sharegroup' not in this_model:
        this_model['constlayer2sharegroup'] = {}
    constlayer2sharegroup = this_model['constlayer2sharegroup']

    num_filters = flags['num_filters']
    conv_stride = flags['stride'] if 'stride' in flags else 0
    layername = flags['layername'] if 'layername' in flags else None
    filter_size = flags['filter_size'] if 'filter_size' in flags else 0

    if conv_stride == 0 or conv_stride == 1:
        pad = 'same'
    elif conv_stride > 0:
        if filter_size == conv_stride:
            pad = 0
        else:
            pad = num_filters//2
    else:  # conv_stride<0
        num_filters = num_filters*(-conv_stride)*(-conv_stride)
        if 'nopad' not in flags:
            pad = num_filters//2
        else:
            pad = 0
    nonlinearity = None
    if 'linear' in flags:
        pass
    elif 'nonlinearity' in flags:
        nonlinearity = flags['nonlinearity']
    else:
        nonlinearity = this_model.get('relu', neon.transforms.Rectlin())

    sharegroup = flags['sharegroup'] if 'sharegroup' in flags else 0

    if sharegroup and sharegroup in sharegroup2params:
        paramlayer = None  # sharegroup2params[sharegroup]
    else:
        paramlayer = None
        init = this_model.get('init', neon.initializers.GlorotUniform())
        if 'init' in flags:
            init = flags['init']

    # dim=len(lasagne.layers.get_output_shape(network))-2 #XXX
    # dim=2
#    if 'maxpool' in flags:
#        assert filter_size>0
#        network=sequential(layers=(network, Pooling(
#            fshape=filter_size,
#            stride=max(1, conv_stride),
#            pad=0,
#            op='max',
#            name=layername,
#            )))
#    elif 'meanpool' in flags:
#        assert filter_size>0
#        network=sequential(layers=(network, Pooling(
#            fshape=filter_size,
#            stride=max(1, conv_stride),
#            pad=0,
#            op='avg',
#            name=layername,
#            )))
#    elif 'upscale' in flags:
#        raise NotImplementedError
#    else:

    if 'dense' in flags:  # or dim==0
        paramlayer = sequential(layers=Affine(
                nout=num_filters,
                init=init,
                bias=neon.initializers.Constant(0.0),
                activation=nonlinearity))
        if sharegroup:
            if 'const' in flags:
                constlayer2sharegroup[paramlayer] = sharegroup
            else:
                layer2sharegroup[paramlayer] = sharegroup
        network = sequential(layers=(
            network,
            paramlayer,
            ))
    else:
        # input_shape = lasagne.layers.get_output_shape(network)
        if 'local' not in flags:
            assert filter_size > 0
            paramlayer = sequential(layers=Conv(
                    fshape=(filter_size, filter_size, num_filters),
                    init=init,
                    bias=neon.initializers.Constant(0.0),
                    strides=max(1, conv_stride),
                    pading=pad,
                    activation=nonlinearity,
                    name=layername,
                    dilation=-conv_stride if conv_stride < 0 else {}
                    ))
            if sharegroup:
                if 'const' in flags:
                    constlayer2sharegroup[paramlayer] = sharegroup
                else:
                    layer2sharegroup[paramlayer] = sharegroup
            network = sequential(layers=(
                network,
                paramlayer,
                ))
        else:  # local
            raise NotImplementedError
    paramlayers += [paramlayer]
    if sharegroup and sharegroup not in sharegroup2params:
        sharegroup2params[sharegroup] = ['W', 'b']
    if 'saveparamlayer' in flags and flags['saveparamlayer'] is not None:
        g = flags['saveparamlayer']
        if g not in stacks:
            stacks[g] = []
        stacks[g] += [paramlayer]
    return network, paramlayers


def dimshuffle_handler(network, flags, stacks, this_model):
    raise NotImplementedError


def noise_handler(network, flags, stacks, this_model):
    raise NotImplementedError


def watch_handler(network, flags, stacks, this_model):
    raise NotImplementedError
#    get_layer=this_model['get_layer']
#
#    tmp=None
#    g=None
#    if type(flags['watch'])==str:
#        g = flags['watch']
#        tmp=network
#    else:
#        if len(flags['watch'])==2:
#            to, g=flags['watch']
#            eq=lasagne.objectives.squared_error
#        else:
#            to, g, eq=flags['watch']
#        if type(to)==type(lambda x:x):
#            tmp=lasagne.layers.ExpressionLayer(network, to, output_shape=(batchsize, ))
#        elif to=='zeros':
#            s0=lasagne.layers.get_output_shape(network)
#            target=ZeroLayer(shape=s0, input_var=T.zeros(s0, dtype=theano.config.floatX))
#            #tmp=lasagne.layers.NonlinearityLayer(network,
#            #        nonlinearity=lambda x:x**2.0
#            #        )
#            tmp=lasagne.layers.ElemwiseMergeLayer((network, target), eq)
#        else:
#            target=get_layer(to)
#            tmp=lasagne.layers.ElemwiseMergeLayer((network, target), eq)
#    if 'sum' in flags:
#        if type(flags['sum'])==int:
#            n=flags['sum']
#        else:
#            n=1
#        shape=lasagne.layers.get_output_shape(tmp)[:n]
#        tmp=lasagne.layers.ExpressionLayer(tmp, curry(lambda n, shape, x:x.flatten(ndim=n+1).sum(axis=n), n, shape), output_shape=shape)
#    if g not in watchpoints:
#        watchpoints[g]=[]
#    watchpoints[g]+=[tmp]
#    return network, ()


def equal_handler(network, flags, stacks, this_model):
    get_layer = this_model['get_layer']

    if 'errors' not in this_model:
        this_model['errors'] = {}
    errors = this_model['errors']
    if len(flags['equal']) == 2:
        to, g = flags['equal']
        eq = neon.transforms.cost.MeanSquared()
    else:
        to, g, eq = flags['equal']
    if g not in errors:
        errors[g] = []
    if to == 'zeros':
        delta = network
    else:
        target = get_layer(to)
        if isinstance(target, neon.layers.layer.DataTransform):
            delta = [network, target]
        else:
            assert isinstance(eq, neon.transforms.cost.MeanSquared)
            tmp = sequential(
                    layers=(
                        target,
                        Activation(neon.transforms.Normalizer(divisor=-1))))
            delta = MergeSum(layers=(network, tmp))
    cost = GeneralizedCost(eq, name=g)
    # tmp=lasagne.layers.ElemwiseMergeLayer((network, target), eq)
    if 'sum' in flags:
        raise NotImplementedError
#        if type(flags['sum'])==int:
#            n=flags['sum']
#        else:
#            n=1
#        shape=lasagne.layers.get_output_shape(tmp)[:n]
#        tmp=lasagne.layers.ExpressionLayer(tmp, curry(lambda n, shape, x:x.flatten(ndim=n+1).sum(axis=n), n, shape), output_shape=shape)
    errors[g] += [(cost, delta)]
    return network, ()


def relu_handler(network, flags, stacks, this_model):
    assert flags['relu'] is True
    nonlinearity = this_model.get('relu', neon.transforms.Rectlin())
    if 'shape' in flags:
        raise NotImplementedError
    else:
        network = sequential(
                layers=(network, Activation(transform=nonlinearity)))
    return network, ()


def nonlinearity_handler(network, flags, stacks, this_model):
    relu = this_model.get('relu', neon.transforms.Rectlin())
    if type(flags) == dict:
        if 'nonlinearity' in flags:
            nonlinearity = flags['nonlinearity']
        if not callable(nonlinearity):
            nonlinearity = relu
    else:
        nonlinearity = relu
    if 'shape' in flags:
        raise NotImplementedError
    else:
        network = sequential(
                layers=(network, Activation(transform=nonlinearity)))
    return network, ()


def argmax_handler(network, flags, stacks, this_model):
    raise NotImplementedError
#    if type(flags['argmax'])==tuple:
#        axis=flags['argmax']
#    else:
#        axis=(1, )
#    shape=lasagne.layers.get_output_shape(network)
#    output_shape=()
#    for idx, w in enumerate(shape):
#        if idx not in axis:
#            output_shape+=(w, )
#    network = lasagne.layers.ExpressionLayer(network, curry(lambda shape, axis, beta, x: goroshin_argmax(x, shape, axis=axis, beta=beta).astype(theano.config.floatX), shape, axis, flags['beta']),  output_shape=output_shape[0:1]+(len(axis), )+output_shape[1:])
#    return network, ()


def unargmax_handler(network, flags, stacks, this_model):
    raise NotImplementedError
#    if type(flags['unargmax'])==tuple:
#        axis=flags['unargmax']
#    else:
#        axis=(1, )
#    shape=flags['shape']
#    if type(shape)==tuple:
#        shape=list(shape)
#    if type(shape)==list and shape[0]==None:
#        shape[0]=lasagne.layers.get_output_shape(network)[0]
#    network = lasagne.layers.ExpressionLayer(network, curry(lambda shape, axis, x: goroshin_unargmax(x, shape, axis=axis).astype(theano.config.floatX), shape, axis),  output_shape=shape)
#    return network, ()


def max_handler(network, flags, stacks, this_model):
    raise NotImplementedError
#    if type(flags['max'])==tuple:
#        axis=flags['max']
#    else:
#        axis=(1, )
#    shape=list(lasagne.layers.get_output_shape(network))
#    for i in axis:
#        shape[i]=1
#    network = lasagne.layers.ExpressionLayer(network, curry(lambda axis, beta, x: goroshin_max(x, axis=axis, beta=beta, keepdims=True).astype(theano.config.floatX), axis, flags['beta']),  output_shape=shape)
#    return network, ()


def branch_handler(network, flags, stacks, this_model):
    global network_branch
    b = BranchNode(name='branch_'+str(flags['branch']))
    network = sequential(layers=(network, b))
    network_branch[network] = b
    return network, ()

register_flag_handler('equal', equal_handler)
register_flag_handler('watch', watch_handler)

register_flag_handler('branch', branch_handler)
register_flag_handler('relu', relu_handler)
register_flag_handler('nonlinearity', nonlinearity_handler, ('num_filters', ))
register_flag_handler('noise', noise_handler)
register_flag_handler('unargmax', unargmax_handler)
register_flag_handler('argmax', argmax_handler)
register_flag_handler('max', max_handler)
register_flag_handler('dimshuffle', dimshuffle_handler)
register_flag_handler('num_filters', num_filters_handler, (
    'maxpool', 'meanpool', 'upscale'))
register_flag_handler('upscale', upscale_handler)
register_flag_handler('meanpool', meanpool_handler)
register_flag_handler('maxpool', maxpool_handler)
register_flag_handler('slice', slice_handler)
register_flag_handler('reshape', reshape_handler)


class LayerSelector(neon.layers.Sequential):
    def __init__(self, *args, **kwargs):
        assert 'layer2sharegroup' in kwargs
        assert 'constlayer2sharegroup' in kwargs
        self.layer2sharegroup = kwargs['layer2sharegroup']
        self.constlayer2sharegroup = kwargs['constlayer2sharegroup']
        self.last_selected_layers = {}
        kwargs.pop('layer2sharegroup')
        kwargs.pop('constlayer2sharegroup')
        # print args, kwargs
        super(LayerSelector, self).__init__(*args, **kwargs)

        self.sharegroup2layers = {}
        self.sharegroup2constlayers = {}
        for k in self.layer2sharegroup:
            v = self.layer2sharegroup[k]
            if v not in self.sharegroup2layers:
                self.sharegroup2layers[v] = []
            if v not in self.sharegroup2constlayers:
                self.sharegroup2constlayers[v] = []
            self.sharegroup2layers[v] += [k]
        for k in self.constlayer2sharegroup:
            v = self.constlayer2sharegroup[k]
            if v not in self.sharegroup2layers:
                self.sharegroup2layers[v] = []
            if v not in self.sharegroup2constlayers:
                self.sharegroup2constlayers[v] = []
            self.sharegroup2constlayers[v] += [k]

    @property
    def layers_to_optimize(self):
        for g in self.last_selected_layers:
            l = self.last_selected_layers[g]
            for layer in l.layers_fprop():
                if isinstance(layer, Bias):
                    b = layer.W
                if isinstance(layer, Linear):
                    W = layer.W
            for ll in self.sharegroup2layers[g]+self.sharegroup2constlayers[g]:
                if ll != l:
                    b_done = False
                    W_done = False
                    for layer in ll.layers_fprop():
                        if isinstance(layer, Bias):
                            assert not b_done
                            if b:
                                assert layer.W.shape == b.shape
                                layer.W = b
                                layer.dW = layer.be.empty_like(layer.W)
                            b_done = True
                        if isinstance(layer, Linear):
                            assert not W_done
                            if W:
                                assert layer.W.shape == W.shape
                                layer.W = W
                                layer.dW = layer.be.empty_like(layer.W)
                            W_done = True

        self.last_selected_layers = {}
        select = self.last_selected_layers
        for g in self.sharegroup2layers:
            k = int(random.random()*len(self.sharegroup2layers[g]))
            select[g] = self.sharegroup2layers[g][k]
        a = super(LayerSelector, self).layers_to_optimize
        lto = []
        done = {}
        for l in a:
            if l in self.layer2sharegroup:
                sharegroup = self.layer2sharegroup[l]
                if not done[sharegroup] and l == select[sharegroup]:
                    lto += [l]
                    done[sharegroup] = True
            else:
                lto += [l]
        return lto


def network_wrapper(network, stacks, this_model):
    return LayerSelector(
            network,
            layer2sharegroup=this_model['layer2sharegroup'],
            constlayer2sharegroup=this_model['constlayer2sharegroup'])

register_network_wrapper(network_wrapper)


def branchs(a):
    refs = {}
    network = (-1, )
    refs[network] = []
    stacks = {}
    all_layers = Layers(network, stacks)

    get_layer = all_layers.get_layer

    count = 0
    for info in a:
        inputs = info[0]
        flags = info[-1]
        if type(inputs) == int or type(inputs) == str or type(inputs) == list:
            layers = [get_layer(inputs)]
        elif type(inputs) == tuple:
            layers = map(get_layer, inputs)
        else:
            print type(inputs)
            raise Exception
        network = count
        refs[network] = []
        if 'push' in flags:
            push = flags['push']
            if type(push) == str:
                push = [push]
            for t in push:
                if t not in stacks:
                    stacks[t] = []
                stacks[t] += [network]
        if 'pop' in flags:
            stacks[flags['pop']] = stacks[flags['pop']][:-1]
        for t in layers:
            refs[t] += [network]
        count += 1
        all_layers.add(network)

    for t in refs:
        if len(refs[t]) > 1:
            a[t][-1]['branch'] = t
    return a


register_macro_handler(branchs)
