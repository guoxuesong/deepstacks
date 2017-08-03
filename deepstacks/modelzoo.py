#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

from macros import macros
from macros import inception
from macros import namespace
from stacked import softmax


def googlenet(n):
    return macros((
        (0, 64, 7, 2, 0, 0, {'layername': 'conv1/7x7_s2'}),
        (0, 0, 3, 2, 0, 0, {'maxpool': True, 'layername': 'pool1/3x3_s2'}),
        (0, 0, 0, 0, 0, 0, {
            'lru': {'alpha': 0.00002, 'k': 1},
            'layername': 'pool1/norm1'}),
        (0, 64, 1, 1, 0, 0, {'layername': 'conv2/3x3_reduce'}),
        (0, 192, 3, 1, 0, 0, {'layername': 'conv2/3x3'}),
        (0, 0, 0, 0, 0, 0, {
            'lru': {'alpha': 0.00002, 'k': 1},
            'layername': 'conv2/norm2'}),
        (0, 0, 3, 2, 0, 0, {'maxpool': True, 'layername': 'pool2/3x3_s2'}),
        (namespace, 'inception_3a/', (
            (inception, (32,  64,  96,  128,  16,  32,)),
            )),
        (namespace, 'inception_3b/', (
            (inception, (64,  128,  128,  192,  32,  96, )),
            )),
        (0, 0, 3, 2, 0, 0, {'maxpool': True, 'layername': 'pool3/3x3_s2'}),
        (namespace, 'inception_4a/', (
            (inception, (64,   192,  96,   208,  16,  48,)),
            )),
        (namespace, 'inception_4b/', (
            (inception, (64,   160,  112,  224,  24,  64,)),
            )),
        (namespace, 'inception_4c/', (
            (inception, (64,   128,  128,  256,  24,  64,)),
            )),
        (namespace, 'inception_4d/', (
            (inception, (64,   112,  144,  288,  32,  64,)),
            )),
        (namespace, 'inception_4e/', (
            (inception, (128,  256,  160,  320,  32,  128, )),
            )),
        (0, 0, 3, 2, 0, 0, {'maxpool': True, 'layername': 'pool4/3x3_s2'}),
        (namespace, 'inception_5a/', (
            (inception, (128,  256,  160,  320,  32,  128,)),
            )),
        (namespace, 'inception_5b/', (
            (inception, (128,  384,  192,  384,  48,  128,)),
            )),
        (0, 0, 7, 1, 0, 0, {
            'meanpool': True, 'pad': 0, 'layername': 'pool5/7x7_s1'}),
        (0, n, 0, 0, 0, 0, {
            'dense': True, 'linear': True, 'layername': 'loss3/classifier'}),
        (0, 0, 0, 0, 0, 0, {
            'nonlinearity': (lambda x:x,softmax), 'layername': 'prob'}),
        ))
