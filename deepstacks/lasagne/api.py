#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import lasagne
from ..stacked import curr_layer


def shape_fn(curr_layer,n):
    return lasagne.layers.get_output_shape(curr_layer)[n]

curr_batchsize = (shape_fn, curr_layer, 0, {})
curr_filters = (shape_fn, curr_layer, 1, {})
curr_width = (shape_fn, curr_layer, 2, {})
curr_height = (shape_fn, curr_layer, 3, {})
curr_depth = (shape_fn, curr_layer, 4, {})
