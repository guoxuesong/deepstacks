#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

from macros import macros
from macros import inception
from macros import namespace
from macros import ln
from stacked import softmax

from framework.main import newlayers,layers

def googlenet(n,auxiliary_classifier=True):
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
        (ln,0,'inception_3a'),
        (namespace, 'inception_3b/', (
            (inception, (64,  128,  128,  192,  32,  96, )),
            )),
        (ln,0,'inception_3b'),
        (0, 0, 3, 2, 0, 0, {'maxpool': True, 'layername': 'pool3/3x3_s2'}),
        (namespace, 'inception_4a/', (
            (inception, (64,   192,  96,   208,  16,  48,)),
            )),
        (ln,0,'inception_4a'),
        (switch,auxiliary_classifier,(
            (0, 0, 5, 3, 0, 0, {
                'meanpool': True, 'pad': 0, 'layername': 'loss1/ave_pool'}),
            (0, 128, 1, 1, 0, 0, {'layername': 'loss1/conv'}),
            (0, 1024, 0, 0, 'loss1/fc', 0, {
                'dense': True, 'layername': 'loss1/fc'}),
            (0, 0, 0, 0, 0, 0, {
                'dropout': 0.7, 'layername': 'loss1/dropout'}),
            (0, n, 0, 0, 'loss1/classifier', 0, {
                'dense': True, 'linear': True, 'layername': 'loss1/classifier'}),
            (0, 0, 0, 0, 'loss1/prob', 0, {
                'nonlinearity': (lambda x:x,softmax), 'layername': 'loss1/prob'}),
            ),()),
        (ln,'inception_4a'),
        (namespace, 'inception_4b/', (
            (inception, (64,   160,  112,  224,  24,  64,)),
            )),
        (ln,0,'inception_4b'),
        (namespace, 'inception_4c/', (
            (inception, (64,   128,  128,  256,  24,  64,)),
            )),
        (ln,0,'inception_4c'),
        (namespace, 'inception_4d/', (
            (inception, (64,   112,  144,  288,  32,  64,)),
            )),
        (ln,0,'inception_4d'),
        (switch,auxiliary_classifier,(
            (0, 0, 5, 3, 0, 0, {
                'meanpool': True, 'pad': 0, 'layername': 'loss2/ave_pool'}),
            (0, 128, 1, 1, 0, 0, {'layername': 'loss2/conv'}),
            (0, 1024, 0, 0, 'loss2/fc', 0, {
                'dense': True, 'layername': 'loss2/fc'}),
            (0, 0, 0, 0, 0, 0, {
                'dropout': 0.7, 'layername': 'loss2/dropout'}),
            (0, n, 0, 0, 'loss2/classifier', 0, {
                'dense': True, 'linear': True, 'layername': 'loss2/classifier'}),
            (0, 0, 0, 0, 'loss2/prob', 0, {
                'nonlinearity': (lambda x:x,softmax), 'layername': 'loss2/prob'}),
            ),()),
        (ln,'inception_4d'),
        (namespace, 'inception_4e/', (
            (inception, (128,  256,  160,  320,  32,  128, )),
            )),
        (ln,0,'inception_4e'),
        (0, 0, 3, 2, 0, 0, {'maxpool': True, 'layername': 'pool4/3x3_s2'}),
        (namespace, 'inception_5a/', (
            (inception, (128,  256,  160,  320,  32,  128,)),
            )),
        (ln,0,'inception_5a'),
        (namespace, 'inception_5b/', (
            (inception, (128,  384,  192,  384,  48,  128,)),
            )),
        (ln,0,'inception_5b'),
        (0, 0, 7, 1, 0, 0, {
            'meanpool': True, 'pad': 0, 'layername': 'pool5/7x7_s1'}),
        (0, 0, 0, 0, 0, 0, {
            'dropout': 0.7, 'layername': 'pool5/dropout'}),
        (0, n, 0, 0, 'loss3/classifier', 0, {
            'dense': True, 'linear': True, 'layername': 'loss3/classifier'}),
        (0, 0, 0, 0, 'prob', 0, {
            'nonlinearity': (lambda x:x,softmax), 'layername': 'prob'}),
        ))
