#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import theano
import theano.tensor as T
import lasagne

import math
from .. import stacked
from ..stacked import Layers, register_layers_class
from ..stacked import register_concat_handler, register_inputs_handler
from ..stacked import register_flag_handler, register_flag_handler_closer
from ..stacked import register_layer_handler, register_nonlinearities
from ..stacked import *
from ..utils.curry import curry
from .. import utils
from .argmax import goroshin_max, goroshin_argmax, goroshin_unargmax


def replace_input(layer, m, done=set({})):
    if layer in m:
        return m[layer]
    if layer in done:
        return layer
    done.add(layer)
    if hasattr(layer,  'input_layer'):
        if layer.input_layer in m:
            layer.input_layer = m[layer.input_layer]
        else:
            replace_input(layer.input_layer, m, done)
    if hasattr(layer,  'input_layers'):
        for i, t in enumerate(layer.input_layers):
            if t in m:
                layer.input_layers[i] = m[t]
            else:
                replace_input(t, m, done)
    return layer


def stacks_replace_input(stacks, m):
    for k in stacks:
        a = stacks[k]
        if type(a) == list:
            for i, t in enumerate(a):
                a[i] = replace_input(t, m)
        else:
            for k in a:
                aa = a[k]
                for i, t in enumerate(aa):
                    aa[i] = replace_input(t, m)


class PlaceHolderLayer(lasagne.layers.InputLayer):
    pass


class ZeroLayer(lasagne.layers.InputLayer):
    pass


class LasagneLayers(Layers):
    def get_layer(self, k):
        if type(k) == list and len(k) == 1:
            res = lasagne.layers.NonlinearityLayer(
                    self.get_layer(k[0]), theano.gradient.disconnected_grad)
            return res

        if type(k) == int:
            return self.layers[::-1][k]
        if type(k) == list and len(k) > 1:
            assert len(k) == 3
            if not k[0] in self.future:
                batchsize = lasagne.layers.get_output_shape(self.layers[0])[0]
                self.future[k[0]] = PlaceHolderLayer(shape=(batchsize, )+k[1])
            return self.future[k[0]]
        return self.stacks[k][-1]

    def finish(self):
        m = {}
        for k in self.future:
            m[self.future[k]] = self.stacks[k][0]
        if stacked.verbose:
            print m
        stacks_replace_input(self.stacks, m)

register_layers_class(LasagneLayers)


def concat_handler(layers, flags, stacks, this_model):
    if 'axis' in flags:
        axis = flags['axis']
    else:
        axis = 1
    return lasagne.layers.ConcatLayer(layers, axis=axis)


def merge_handler(layers, flags, stacks, this_model):
    return lasagne.layers.ElemwiseMergeLayer(layers, flags['op'])


def add_handler(layers, flags, stacks, this_model):
    return lasagne.layers.ElemwiseMergeLayer(layers, T.add)


def sub_handler(layers, flags, stacks, this_model):
    return lasagne.layers.ElemwiseMergeLayer(layers, T.sub)

register_concat_handler(concat_handler)
register_inputs_handler('op', merge_handler)
register_inputs_handler('add', add_handler)
register_inputs_handler('sub', sub_handler)


def reshape_handler(network, flags, stacks, this_model):
    if 'raw' in flags:
        network = lasagne.layers.ReshapeLayer(network, flags['reshape'])
    else:
        network = lasagne.layers.ReshapeLayer(network, (-1, )+flags['reshape'])
    return network, ()


def slice_handler(network, flags, stacks, this_model):
    if 'axis' in flags:
        axis = flags['axis']
    else:
        axis = 1
    network = lasagne.layers.SliceLayer(network, flags['slice'], axis=axis)
    return network, ()


def maxpool_handler(network, flags, stacks, this_model):
    layername = flags['layername'] if 'layername' in flags else None
    filter_size = flags['filter_size'] if 'filter_size' in flags else 0
    conv_stride = flags['stride'] if 'stride' in flags else 0
    if conv_stride == 0 or conv_stride == 1:
        pad = filter_size//2
    elif conv_stride > 0:
        if filter_size == conv_stride:
            pad = 0
        else:
            pad = filter_size//2
    if 'pad' in flags:
        pad = flags['pad']
#    else: #conv_stride<0
#        num_filters=num_filters*(-conv_stride)*(-conv_stride)
#        if not 'nopad' in flags:
#            pad='same'
#        else:
#            pad=0

    dim = len(lasagne.layers.get_output_shape(network))-2
    convs = {
            1: lasagne.layers.Pool1DLayer,
            2: lasagne.layers.Pool2DLayer,
            3: lasagne.layers.Pool3DLayer,
            }
    assert dim in convs
    conv = convs[dim]
    assert filter_size > 0
    network = conv(
        network,
        pool_size=filter_size,
        stride=max(1, conv_stride),
        pad=pad,
        mode='max',
        name=layername,
        )
    return network, ()


def meanpool_handler(network, flags, stacks, this_model):
    layername = flags['layername'] if 'layername' in flags else None
    filter_size = flags['filter_size'] if 'filter_size' in flags else 0
    conv_stride = flags['stride'] if 'stride' in flags else 0
    if conv_stride == 0 or conv_stride == 1:
        pad = filter_size//2
    elif conv_stride > 0:
        if filter_size == conv_stride:
            pad = 0
        else:
            pad = filter_size//2
    if 'pad' in flags:
        pad = flags['pad']
#    else: #conv_stride<0
#        num_filters=num_filters*(-conv_stride)*(-conv_stride)
#        if not 'nopad' in flags:
#            pad='same'
#        else:
#            pad=0

    dim = len(lasagne.layers.get_output_shape(network))-2
    convs = {
            1: lasagne.layers.Pool1DLayer,
            2: lasagne.layers.Pool2DLayer,
            3: lasagne.layers.Pool3DLayer,
            }
    assert dim in convs
    conv = convs[dim]
    assert filter_size > 0
    network = conv(
        network,
        pool_size=filter_size,
        stride=max(1, conv_stride),
        pad=pad,
        mode='average_inc_pad',
        name=layername,
        )
    return network, ()


def upscale_handler(network, flags, stacks, this_model):
    layername = flags['layername'] if 'layername' in flags else None
    filter_size = flags['filter_size'] if 'filter_size' in flags else 0

    dim = len(lasagne.layers.get_output_shape(network))-2
    assert filter_size > 0
    convs = {
            1: lasagne.layers.Upscale1DLayer,
            2: lasagne.layers.Upscale2DLayer,
            3: lasagne.layers.Upscale3DLayer,
            }
    assert dim in convs
    conv = convs[dim]
    network = conv(
        network,
        scale_factor=filter_size,
        name=layername,
        mode='repeat',
        )
    return network, ()


def num_filters_handler(network, flags, stacks, this_model):
    paramlayers = []
    if 'sharegroup2params' not in this_model:
        this_model['sharegroup2params'] = {}
    sharegroup2params = this_model['sharegroup2params']

    num_filters0 = flags['num_filters']
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
            pad = 'same'
    else:  # conv_stride<0
        num_filters = num_filters*(-conv_stride)*(-conv_stride)
        if 'nopad' not in flags:
            pad = 'same'
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
        nonlinearity = this_model.get('relu', lasagne.nonlinearities.rectify)

    sharegroup = flags['sharegroup'] if 'sharegroup' in flags else 0

    if sharegroup and sharegroup in sharegroup2params:
        ww = sharegroup2params[sharegroup][0]
        bb = sharegroup2params[sharegroup][1]
        if 'const' in flags:
            ww = theano.gradient.disconnected_grad(ww)
            if bb is not None:
                bb = theano.gradient.disconnected_grad(bb)
    else:
        init = this_model.get('init', lasagne.init.GlorotUniform)
        if 'init' in flags:
            init = flags['init']
        if 'init_gain' in flags:
            ww = init(gain=flags['init_gain'])
        else:
            if nonlinearity == lasagne.nonlinearities.leaky_rectify:
                alpha = 0.01
                ww = init(gain=math.sqrt(2/(1+alpha**2)))
            elif nonlinearity == lasagne.nonlinearities.rectify:
                ww = init(gain='relu')
            else:
                ww = init()
        if 'nobias' in flags:
            bb = None
        else:
            bb = lasagne.init.Constant(0.0)

    dim = len(lasagne.layers.get_output_shape(network))-2

    if 'dense' in flags or dim == 0:
        if 'bn' in flags:
            network = lasagne.layers.DenseLayer(
                    network,
                    num_units=num_filters,
                    W=ww,
                    b=None,
                    nonlinearity=None,
                    name=layername,
                    )
            savew = network.W
            paramlayers += [network]
            network = lasagne.layers.BatchNormLayer(network, beta=bb)
            saveb = network.beta
            paramlayers += [network]
            network = lasagne.layers.NonlinearityLayer(
                    network, nonlinearity=nonlinearity)
        else:
            network = lasagne.layers.DenseLayer(
                    network,
                    num_units=num_filters,
                    W=ww,
                    b=bb,
                    nonlinearity=nonlinearity,
                    name=layername,
                    )
            savew = network.W
            saveb = network.b
            paramlayers += [network]
    else:
        # input_shape = lasagne.layers.get_output_shape(network)
        if 'local' not in flags:
            convs = {
                    1: lasagne.layers.Conv1DLayer,
                    2: lasagne.layers.Conv2DLayer,
                    3: lasagne.layers.Conv3DLayer,
                    }
            assert dim in convs
            conv = convs[dim]

            assert filter_size > 0
            if 'bn' in flags:
                network = conv(
                    network,  num_filters=num_filters,
                    filter_size=filter_size,
                    stride=max(1, conv_stride),
                    pad=pad,
                    W=ww,
                    b=None,
                    nonlinearity=None,
                    name=layername,
                    )
                savew = network.W
                paramlayers += [network]
                network = lasagne.layers.BatchNormLayer(network, beta=bb)
                saveb = network.beta
                paramlayers += [network]
                network = lasagne.layers.NonlinearityLayer(
                        network, nonlinearity=nonlinearity)
            else:
                network = conv(
                    network,  num_filters=num_filters,
                    filter_size=filter_size,
                    stride=max(1, conv_stride),
                    pad=pad,
                    W=ww,
                    b=bb,
                    nonlinearity=nonlinearity,
                    name=layername,
                    )
                savew = network.W
                saveb = network.b
                paramlayers += [network]
        else:
            convs = {
                    1: lasagne.layers.LocallyConnected1DLayer,
                    2: lasagne.layers.LocallyConnected2DLayer,
                    3: lasagne.layers.LocallyConnected3DLayer,
                    }
            assert dim in convs
            conv = convs[dim]
            assert conv_stride == 1
            assert filter_size > 0
            if 'bn':
                network = conv(
                    network, num_filters=num_filters,
                    filter_size=filter_size,
                    stride=max(1, conv_stride),
                    pad=pad,
                    W=ww,
                    b=None,
                    nonlinearity=None,
                    name=layername,
                    untie_biases=True,
                    )
                savew = network.W
                paramlayers += [network]
                network = lasagne.layers.BatchNormLayer(network, beta=bb)
                saveb = network.beta
                paramlayers += [network]
                network = lasagne.layers.NonlinearityLayer(
                        network, nonlinearity=nonlinearity)
            else:
                network = conv(
                    network, num_filters=num_filters,
                    filter_size=filter_size,
                    stride=max(1, conv_stride),
                    pad=pad,
                    W=ww,
                    b=bb,
                    nonlinearity=nonlinearity,
                    name=layername,
                    untie_biases=True,
                    )
                paramlayers += [network]
                savew = network.W
                saveb = network.b
    # paramlayers += [network]
    if sharegroup and sharegroup not in sharegroup2params:
        sharegroup2params[sharegroup] = [savew, saveb]
    if 'saveparamlayer' in flags and flags['saveparamlayer'] is not None:
        g = flags['saveparamlayer']
        if g not in stacks:
            stacks[g] = []
        stacks[g] += [network]
    if conv_stride < 0:
        b, c, width, height = lasagne.layers.get_output_shape(network)
        network = lasagne.layers.ReshapeLayer(
                network,
                (b, num_filters0, -conv_stride, -conv_stride, width, height))
        network = lasagne.layers.DimshuffleLayer(network, (0, 1, 4, 2, 5, 3))
        network = lasagne.layers.ReshapeLayer(
                network,
                (b, num_filters0, width*(-conv_stride), height*(-conv_stride)))
    return network, paramlayers


def dimshuffle_handler(network, flags, stacks, this_model):
    return lasagne.layers.DimshuffleLayer(network, flags['dimshuffle']), ()


def noise_handler(network, flags, stacks, this_model):
    sigma = flags['noise']
    if sigma is True:
        sigma = 0.1
    return lasagne.layers.GaussianNoiseLayer(network, sigma), ()


def lrn_handler(network, flags, stacks, this_model):
    if type(flags['lrn']) == dict:
        return lasagne.layers.LocalResponseNormalization2DLayer(
            network, **flags['lrn']), ()
    else:
        return lasagne.layers.LocalResponseNormalization2DLayer(network), ()

def dropout_handler(network, flags, stacks, this_model):
    p = flags['dropout']
    if p is True:
        p = 0.5
    return lasagne.layers.DropoutLayer(network,p=p), ()

def watch_handler(network, flags, stacks, this_model):
    get_layer = this_model['get_layer']

    if 'watchpoints' not in this_model:
        this_model['watchpoints'] = {}
    watchpoints = this_model['watchpoints']
    tmp = None
    g = None
    if type(flags['watch']) == str:
        g = flags['watch']
        tmp = network
    else:
        if len(flags['watch']) == 2:
            to, g = flags['watch']
            eq = lasagne.objectives.squared_error
        else:
            to, g, eq = flags['watch']
        if callable(to):  # type(to)==type(lambda x:x):
            #batchsize = lasagne.layers.get_output_shape(network)[0]
            tmp = lasagne.layers.NonlinearityLayer(
                    network, to)
        elif to == 'zeros':
            s0 = lasagne.layers.get_output_shape(network)
            target = ZeroLayer(
                    shape=s0,
                    input_var=T.zeros(s0, dtype=theano.config.floatX))

            # tmp=lasagne.layers.NonlinearityLayer(network,
            #        nonlinearity=lambda x:x**2.0
            #        )
            tmp = lasagne.layers.ElemwiseMergeLayer((network, target), eq)
        else:
            target = get_layer(to)
            tmp = lasagne.layers.ElemwiseMergeLayer((network, target), eq)
    if 'sum' in flags:
        if type(flags['sum']) == int:
            n = flags['sum']
        else:
            n = 1
        shape = lasagne.layers.get_output_shape(tmp)[:n]
        tmp = lasagne.layers.ExpressionLayer(
                tmp,
                curry(
                    lambda n, shape, x: x.flatten(ndim=n+1).sum(axis=n),
                    n, shape),
                output_shape=shape)
    if g not in watchpoints:
        watchpoints[g] = []
    watchpoints[g] += [tmp]
    return network, ()


def equal_handler(network, flags, stacks, this_model):
    get_layer = this_model['get_layer']

    if 'errors' not in this_model:
        this_model['errors'] = {}
    errors = this_model['errors']
    if len(flags['equal']) == 2:
        to, g = flags['equal']
        eq = lasagne.objectives.squared_error
        w = None
    elif len(flags['equal']) == 3:
        to, g, eq = flags['equal']
        w = None
    else:
        to, g, eq, w = flags['equal']
    if g not in errors:
        errors[g] = []
    if callable(to):  # type(to)==type(lambda x:x):
        #batchsize = lasagne.layers.get_output_shape(network)[0]
        tmp = lasagne.layers.NonlinearityLayer(
                network, to)
    elif to == 'zeros':
        s0 = lasagne.layers.get_output_shape(network)
        target = ZeroLayer(
                shape=s0,
                input_var=T.zeros(s0, dtype=theano.config.floatX))
        tmp = lasagne.layers.ElemwiseMergeLayer((network, target), eq)
    else:
        target = get_layer(to)
        tmp = lasagne.layers.ElemwiseMergeLayer((network, target), eq)
    if w is not None:
        w = get_layer(w)
        tmp = lasagne.layers.ElemwiseMergeLayer((tmp,w),lambda x,y:x*y/(y.sum(dtype=theano.config.floatX)+utils.floatX(1e-4))*T.prod(y.shape,dtype=theano.config.floatX))
    if 'sum' in flags:
        if type(flags['sum']) == int:
            n = flags['sum']
        else:
            n = 1
        shape = lasagne.layers.get_output_shape(tmp)[:n]
        tmp = lasagne.layers.ExpressionLayer(
                tmp,
                curry(
                    lambda n, shape, x: x.flatten(ndim=n+1).sum(axis=n),
                    n, shape),
                output_shape=shape)
    errors[g] += [tmp]
    return network, ()


def relu_handler(network, flags, stacks, this_model):
    assert flags['relu'] is True
    nonlinearity = this_model.get('relu', lasagne.nonlinearities.rectify)
    if 'shape' in flags:
        shape = flags['shape']
        if type(shape) == tuple:
            shape = list(shape)
        if type(shape) == list and shape[0] is None:
            shape[0] = lasagne.layers.get_output_shape(network)[0]
        network = lasagne.layers.ExpressionLayer(
                network, nonlinearity, output_shape=shape)
    else:
        network = lasagne.layers.NonlinearityLayer(
                network, nonlinearity=nonlinearity)
    return network, ()


def nonlinearity_handler(network, flags, stacks, this_model):
    relu = this_model.get('relu', lasagne.nonlinearities.rectify)
    if type(flags) == dict:
        if 'nonlinearity' in flags:
            nonlinearity = flags['nonlinearity']
        if not callable(nonlinearity):
            nonlinearity = relu
    else:
        nonlinearity = relu
    if 'shape' in flags:
        shape = flags['shape']
        if type(shape) == tuple:
            shape = list(shape)
        if type(shape) == list and shape[0] is None:
            shape[0] = lasagne.layers.get_output_shape(network)[0]
        network = lasagne.layers.ExpressionLayer(
                network, nonlinearity, output_shape=shape)
    else:
        network = lasagne.layers.NonlinearityLayer(
                network, nonlinearity=nonlinearity)
    return network, ()


def argmax_handler(network, flags, stacks, this_model):
    if type(flags['argmax']) == tuple:
        axis = flags['argmax']
    else:
        axis = (1, )
    shape = lasagne.layers.get_output_shape(network)
    output_shape = ()
    for idx, w in enumerate(shape):
        if idx not in axis:
            output_shape += (w, )
    network = lasagne.layers.ExpressionLayer(
            network, curry(
                lambda shape, axis, beta, x: goroshin_argmax(
                    x, shape, axis=axis, beta=beta
                    ).astype(theano.config.floatX),
                shape, axis, flags['beta']),
            output_shape=output_shape[0:1]+(len(axis), )+output_shape[1:])
    return network, ()


def unargmax_handler(network, flags, stacks, this_model):
    if type(flags['unargmax']) == tuple:
        axis = flags['unargmax']
    else:
        axis = (1, )
    shape = flags['shape']
    sigma = flags['sigma'] if 'sigma' in flags else 1.0
    if type(shape) == tuple:
        shape = list(shape)
    if type(shape) == list and shape[0] is None:
        shape[0] = lasagne.layers.get_output_shape(network)[0]
    network = lasagne.layers.ExpressionLayer(
            network,
            curry(
                lambda shape, axis, x: goroshin_unargmax(
                    x, shape, axis=axis, sigma=sigma
                    ).astype(theano.config.floatX),
                shape, axis),
            output_shape=shape)
    return network, ()


def max_handler(network, flags, stacks, this_model):
    if type(flags['max']) == tuple:
        axis = flags['max']
    else:
        axis = (1, )
    shape = list(lasagne.layers.get_output_shape(network))
    for i in axis:
        shape[i] = 1
    network = lasagne.layers.ExpressionLayer(
            network, curry(
                lambda axis, beta, x: goroshin_max(
                    x, axis=axis, beta=beta, keepdims=True
                    ).astype(theano.config.floatX),
                axis, flags['beta']),
            output_shape=shape)
    return network, ()

register_flag_handler('equal', equal_handler)
register_flag_handler('watch', watch_handler)

register_flag_handler('relu', relu_handler)
register_flag_handler('nonlinearity', nonlinearity_handler, ('num_filters', ))
register_flag_handler('noise', noise_handler)
register_flag_handler('lrn', lrn_handler)
register_flag_handler('dropout', dropout_handler)
register_flag_handler('unargmax', unargmax_handler)
register_flag_handler('argmax', argmax_handler)
register_flag_handler('max', max_handler)
register_flag_handler('dimshuffle', dimshuffle_handler)
# register_flag_handler_closer(num_filters_handler, num_filters_handler_closer)
register_flag_handler('num_filters', num_filters_handler, (
    'maxpool', 'meanpool', 'upscale'))
register_flag_handler('upscale', upscale_handler)
register_flag_handler('meanpool', meanpool_handler)
register_flag_handler('maxpool', maxpool_handler)
register_flag_handler('slice', slice_handler)
register_flag_handler('reshape', reshape_handler)


def layer_handler(network):
    if stacked.verbose:
        print 'output_shape:', lasagne.layers.get_output_shape(network)

register_layer_handler(layer_handler)


register_nonlinearities({
            'softmax': lasagne.nonlinearities.softmax,
            'rectify': lasagne.nonlinearities.rectify,
            'sigmoid': lasagne.nonlinearities.sigmoid,
            'tanh': lasagne.nonlinearities.tanh,
            'linear': lasagne.nonlinearities.linear,
            })
