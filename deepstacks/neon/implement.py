#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import neon
import random
import numpy as np
import math

from ..stacked import Layers, register_layers_class
from ..stacked import register_concat_handler, register_inputs_handler
from ..stacked import register_flag_handler, register_network_wrapper
from ..stacked import register_macro_handler, register_nonlinearities
from ..stacked import *
from neon.layers.layer import Affine, Activation, Linear, SkipNode
from neon.layers.layer import Bias, BranchNode, GeneralizedCost
from neon.layers.layer import Reshape, Pooling, Conv, Layer, LRN
from neon.layers.container import Sequential, MergeSum, MergeBroadcast
from neon.initializers import GlorotUniform as NeonGlorotUniform
import utils


def get_output_shape(network):
    if not isinstance(network, neon.layers.layer.DataTransform):
        network.configure(None)
    return (network.be.bsz,)+network.out_shape


class GaussianNoiseLayer(Layer):
    def __init__(self, sigma=0.1, name=None):
        super(GaussianNoiseLayer, self).__init__(name)
        self.sigma = sigma
        self.owns_delta = True
        self.is_mklop = True

    def fprop(self, inputs=None, inference=False, beta=0):
        self.be.fill_normal(self.noisebuf, stdv=self.sigma)
        self.be.fprop_skipnode(inputs, self.outputs, beta)
        self.outputs[:] = self.outputs + self.noisebuf
        return self.outputs

    def configure(self, in_obj):
        super(GaussianNoiseLayer, self).configure(in_obj)
        self.out_shape = self.in_shape

        self.noisebuf = self.be.iobuf(self.in_shape, dtype=np.float32)
        # self.noisebuf = self.be.iobuf(self.in_shape)
        return self

    def bprop(self, error, alpha=1.0, beta=0.0):
        # for better performance, mkl do nothing
        # otherwise, convert back and deal with beta and alpha.
        self.be.bprop_skipnode(error, self.deltas, alpha, beta)
        return self.deltas


class DimshuffleLayer(Layer):
    def __init__(self, pattern, name=None):
        super(GaussianNoiseLayer, self).__init__(name)
        self.pattern = pattern
        self.owns_delta = True
        self.is_mklop = True

    def fprop(self, inputs=None, inference=False, beta=0):
        self.be.copy_transpose(inputs, self.outputs, axis=self.pattern)
        return self.outputs

    def configure(self, in_obj):
        super(GaussianNoiseLayer, self).configure(in_obj)

        input_shape = (self.be.bsz,)+self.in_shape

        # Copy from lasagne/layers/shape.py
        #
        # Build output shape while keeping track of the dimensions that we are
        # attempting to collapse, so we can ensure that they are broadcastable
        output_shape = []
        dims_used = [False] * len(input_shape)
        for p in self.pattern:
            if isinstance(p, int):
                if p < 0 or p >= len(input_shape):
                    raise ValueError("pattern contains {0}, but input shape "
                                     "has {1} dimensions "
                                     "only".format(p, len(input_shape)))
                # Dimension p
                o = input_shape[p]
                dims_used[p] = True
            elif p == 'x':
                # Broadcast; will be of size 1
                o = 1
            output_shape.append(o)

        for i, (dim_size, used) in enumerate(zip(input_shape, dims_used)):
            if not used and dim_size != 1 and dim_size is not None:
                raise ValueError(
                    "pattern attempted to collapse dimension "
                    "{0} of size {1}; dimensions with size != 1/None are not"
                    "broadcastable and cannot be "
                    "collapsed".format(i, dim_size))

        ###

        self.out_shape = tuple(output_shape)

        return self

    def bprop(self, error, alpha=1.0, beta=0.0):
        self.be.copy_transpose(
                error, self.deltas, axis=np.argsort(self.pattern))
        return self.deltas


class ConstantLayer(Layer):
    def __init__(self, name=None):
        super(ConstantLayer, self).__init__(name)
        self.owns_delta = True
        self.is_mklop = True

    def fprop(self, inputs=None, inference=False, beta=0):
        self.be.fprop_skipnode(inputs, self.outputs, beta)
        return self.outputs

    def configure(self, in_obj):
        super(ConstantLayer, self).configure(in_obj)
        self.out_shape = self.in_shape
        return self

    def bprop(self, error, alpha=1.0, beta=0.0):
        self.deltas[:] = 0
        return self.deltas

network_branch = {}
branch_notfirst = set()


class NeonLayers(Layers):
    def get_layer(self, k):
        constant_flag = False
        if type(k) == list and len(k) == 1:
            constant_flag = True
        layer = super(NeonLayers, self).get_layer(k)

        if layer in network_branch:
            print 'Found branch', layer
            b = network_branch[layer]
            if b in branch_notfirst:
                layer = b
            else:
                branch_notfirst.add(b)
        if constant_flag:
            layer = sequential(layers=(layer, ConstantLayer()))
        return layer
register_layers_class(NeonLayers)


def sequential(layers):
    a = ()
    for t in layers:
        if type(t) == Sequential:
            a += tuple(t.layers)
        else:
            a += (t, )
    res=Sequential(layers=a)
    #print 'in_shape:',a[0].in_shape
    #res.configure(a[0].in_shape)
    return res


def concat_handler(layers, flags, stacks, this_model):
    head, ls = split_merge_layers(layers)
    return Sequential(layers=head+(MergeBroadcast(ls, merge="depth"),))
    # return MergeBroadcast(layers=layers, merge="depth")
    # network = Tree(layers=layers)
    # return MergeMultistream(layers=network, merge="depth")


def merge_handler(layers, flags, stacks, this_model):
    raise NotImplementedError


def split_list(a, d):
    res = [[], ]
    for t in a:
        if t != d:
            res[-1] += [t]
        else:
            res += [[], ]
    print a
    print [d]
    print res
    return res


def split_merge_layers(layers):
    bs = []
    ls = []
    for layer in layers:
        if type(layer) == BranchNode:
            b = layer
            l = SkipNode()
        elif type(layer.layers[0]) == BranchNode:
            b = layer.layers[0]
            l = Sequential(tuple(layer.layers[1:]))
        else:
            b = None
            l = None
        bs += [b]
        ls += [l]
    bset = set(bs)-set({None})
    if len(bset) > 1:
        print bset
    assert len(bset) <= 1
    if len(bset) == 1:
        for b in bset:
            pass
    print 'bs:', bs
    print 'ls:', ls

    head = ()
    for i, layer in enumerate(layers):
        if ls[i] is None:
            ll = split_list(layers[i].layers, b)
            assert len(ll) <= 2
            if len(ll) == 2:
                assert head == ()
                head, l = ll
                head = tuple(head)
                l = tuple(l)
                ls[i] = l
            else:
                ls[i] = layers[i].layers

    print 'bs:', bs
    print 'head:', head
    print 'ls:', ls

    return head, tuple(ls)


def add_handler(layers, flags, stacks, this_model):
    head, ls = split_merge_layers(layers)
    return Sequential(layers=head+(MergeSum(ls),))

#    if type(layers[1])==BranchNode:
#        b=layers[1]
#        l3=SkipNode()
#    elif type(layers[1].layers[0]==BranchNode):
#        b=layers[1].layers[0]==BranchNode
#        l3=Sequential(tuple(layers[1].layers[1:]))
#    else:
#        b=None
#
#    if b is not None:
#        l1,l2=split_list(layers[0].layers,b)
#        l1=tuple(l1)
#        l2=tuple(l2)
#        return Sequential(layers=l1+(MergeSum((l2,l3)),))
#    else:
#        return MergeSum(layers)


def sub_handler(layers, flags, stacks, this_model):
    head, layers = split_merge_layers(layers)
    if len(layers) > 2:
        left = layers[0]
        right = sequential(
                layers=(
                    MergeSum(layers[1:]),
                    Activation(neon.transforms.Normalizer(divisor=-1))))
        network = Sequential(layers=head+(MergeSum(layers=(left, right)),))
    elif len(layers) == 2:
        left = layers[0]
        right = sequential(
                layers=(
                    layers[1],
                    Activation(neon.transforms.Normalizer(divisor=-1))))
        network = Sequential(layers=head+(MergeSum(layers=(left, right)),))
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
    layername = flags.get('layername', None)
    filter_size = flags.get('filter_size', 0)
    conv_stride = flags.get('stride', 0)
    if conv_stride == 0 or conv_stride == 1:
        pad = filter_size//2
    elif conv_stride > 0:
        if filter_size == conv_stride:
            pad = 0
        else:
            pad = filter_size//2
    if 'pad' in flags:
        pad = flags['pad']

    dim = len(get_output_shape(network))-2
    print 'pooling debug:',filter_size,max(1, conv_stride),pad
    assert filter_size > 0
    network = sequential(layers=(network, Pooling(
        fshape=(filter_size,)*dim if dim>=2 else filter_size,
        strides=max(1, conv_stride),
        padding=pad,
        op='max',
        name=layername,
        )))
    return network, ()


def meanpool_handler(network, flags, stacks, this_model):
    # num_filters=flags['num_filters']
    layername = flags.get('layername', None)
    filter_size = flags.get('filter_size', 0)
    conv_stride = flags.get('stride', 0)
    if conv_stride == 0 or conv_stride == 1:
        pad = filter_size//2
    elif conv_stride > 0:
        if filter_size == conv_stride:
            pad = 0
        else:
            pad = filter_size//2
    if 'pad' in flags:
        pad = flags['pad']

    dim = len(get_output_shape(network))-2
    assert filter_size > 0
    network = sequential(layers=(network, Pooling(
        fshape=(filter_size,)*dim if dim>=2 else filter_size,
        strides=max(1, conv_stride),
        padding=pad,
        op='avg',
        name=layername,
        )))
    return network, ()


def upscale_handler(network, flags, stacks, this_model):
    raise NotImplementedError


class GlorotUniform(NeonGlorotUniform):
    def __init__(self, name="autouniformInit", gain=1.0):
        if gain == 'relu':
            gain = np.sqrt(2)
        super(GlorotUniform, self).__init__(name)
        self.gain = gain

    def fill(self, param):
        k = self.gain * np.sqrt(6.0 / (param.shape[0] + param.shape[1]))
        param[:] = self.be.rng.uniform(-k, k, param.shape)


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
    conv_stride = flags.get('stride', 0)
    layername = flags.get('layername', None)
    filter_size = flags.get('filter_size', 0)
    bn = flags.get('bn', False)

    if conv_stride == 0 or conv_stride == 1:
        pad = filter_size//2
    elif conv_stride > 0:
        if filter_size == conv_stride:
            pad = 0
        else:
            pad = filter_size//2
    else:  # conv_stride<0
        num_filters = num_filters*(-conv_stride)*(-conv_stride)
        if 'nopad' not in flags:
            pad = filter_size//2
        else:
            pad = 0
    if 'pad' in flags:
        pad = flags['pad']
    nonlinearity = None
    if 'linear' in flags:
        pass
    elif 'nonlinearity' in flags:
        nonlinearity = flags['nonlinearity']
    else:
        nonlinearity = this_model.get('relu', neon.transforms.Rectlin())

    sharegroup = flags.get('sharegroup', 0)

    # if sharegroup and sharegroup in sharegroup2params:
    #    paramlayer = None  # sharegroup2params[sharegroup]
    # else:
    #    paramlayer = None
    init = this_model.get('init', GlorotUniform())
    if 'init' in flags:
        init = flags['init']
    if 'init_gain' in flags:
        init = GlorotUniform(gain=flags['init_gain'])
    else:
        if nonlinearity == neon.transforms.Rectlin and nonlinearity.slope > 0:
            alpha = nonlinearity.slope
            init = GlorotUniform(gain=math.sqrt(2/(1+alpha**2)))
        elif nonlinearity == neon.transforms.Rectlin:
            init = GlorotUniform(gain='relu')
        else:
            pass
    if 'nobias' in flags:
        bias = None
    else:
        bias = neon.initializers.Constant(0.0)

    # utils.walk(network)
    dim = len(get_output_shape(network))-2

    if 'dense' in flags or dim <= 1:
        paramlayer = sequential(layers=Affine(
                nout=num_filters,
                init=init,
                bias=bias,
                batch_norm=bn,
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
                    fshape=(filter_size,)*dim+(num_filters,),
                    init=init,
                    bias=bias,
                    strides=max(1, conv_stride),
                    padding=pad,
                    activation=nonlinearity,
                    name=layername,
                    batch_norm=bn,
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
    pattern = flags['dimshuffle']
    return sequential(layers=(network, DimshuffleLayer(pattern))), ()


def noise_handler(network, flags, stacks, this_model):
    sigma = flags['noise']
    if sigma is True:
        sigma = 0.1
    return sequential(layers=(network, GaussianNoiseLayer(sigma))), ()


def lrn_handler(network, flags, stacks, this_model):
    if type(flags['lrn']) == dict:
        lasagne_lru = flags['lrn']
    else:
        lasagne_lru = {}
    N = 1e4  # XXX
    kwargs = {}
    k = lasagne_lru.get('k', 2)
    assert k == 1
    kwargs['ascale'] = lasagne_lru.get('alpha', 1e-4)*N  # XXX
    kwargs['bpower'] = lasagne_lru.get('beta', 0.75)
    kwargs['depth'] = lasagne_lru.get('n', 5)
    return sequential(network, LRN(**kwargs)), ()

def dropout_handler(network, flags, stacks, this_model):
    p = flags['dropout']
    if p is True:
        p = 0.5
    if p>1.0:
        p=1.0
    return sequential(layers=(network, Dropout(1.0-p))), ()

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
#            tmp=lasagne.layers.ExpressionLayer(
#                network, to, output_shape=(batchsize, ))
#        elif to=='zeros':
#            s0=lasagne.layers.get_output_shape(network)
#            target=ZeroLayer(shape=s0, input_var=T.zeros(
#                s0, dtype=theano.config.floatX))
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
#        tmp=lasagne.layers.ExpressionLayer(
#            tmp,
#            curry(
#                lambda n, shape, x:x.flatten(ndim=n+1).sum(axis=n), n, shape),
#            output_shape=shape)
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
#        tmp=lasagne.layers.ExpressionLayer(
#            tmp,
#            curry(
#                lambda n, shape, x:x.flatten(ndim=n+1).sum(axis=n), n, shape),
#            output_shape=shape)
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


def unargmax_handler(network, flags, stacks, this_model):
    raise NotImplementedError


def max_handler(network, flags, stacks, this_model):
    raise NotImplementedError


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
register_flag_handler('lrn', lrn_handler)
register_flag_handler('dropout', dropout_handler)
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

def layer_handler(network):
    print 'output_shape:', get_output_shape(network)

register_layer_handler(layer_handler)

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

register_nonlinearities({
            'softmax': neon.transforms.activation.Softmax(),
            'rectify': neon.transforms.activation.Rectlin(),
            'sigmoid': neon.transforms.activation.Logistic(),
            'tanh': neon.transforms.activation.Tanh(),
            'linear': neon.transforms.activation.Identity(),
            })
