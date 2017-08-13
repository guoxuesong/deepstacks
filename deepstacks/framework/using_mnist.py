#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import numpy as np
from .main import register_batch_iterator
from ..utils.curry import curry
from mnist import load_dataset

def iterate_minibatches(inputs, targets, batchsize, database='.', shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        X = {
                'image':inputs[excerpt],
                'target':targets[excerpt].astype('int64'),
                }
        yield X

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
register_batch_iterator(curry(iterate_minibatches,X_train,y_train,shuffle=True),curry(iterate_minibatches,X_val,y_val,shuffle=False))
