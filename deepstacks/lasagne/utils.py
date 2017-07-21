#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import theano
import lasagne
from join import join_layer as JoinLayer

floatX = theano.config.floatX


def ordered_errors(errors, m=None, prefix=''):
    res = []
    for t in errors:
        if m is None:
            res += [[prefix+t, map(lasagne.layers.get_output, errors[t])]]
        else:
            tmp = map(lambda x: JoinLayer(x, m), errors[t])
            res += [[prefix+t, map(lasagne.layers.get_output, tmp)]]
    return sorted(res, key=lambda x: x[0])


def get_loss(errors, watchpoints, loss0=None):
    errors = ordered_errors(errors)
    watch_errors = ordered_errors(watchpoints)

    errors1 = []
    watch_errors1 = []
    train_watch_errors1 = []
    tagslice = []
    count = 0
    valtagslice = []
    valcount = 0
    for tag, errs in errors:
        errors1 += errs
        tagslice += [[tag, slice(count, count+len(errs))]]
        count += len(errs)
    for tag, errs in watch_errors:
        if tag.startswith('train:'):
            train_watch_errors1 += errs
            tagslice += [[tag, slice(count, count+len(errs))]]
            count += len(errs)
        else:
            watch_errors1 += errs
            valtagslice += [[tag, slice(valcount, valcount+len(errs))]]
            valcount += len(errs)
    errors1 = [errors1]
    watch_errors1 = [watch_errors1]
    train_watch_errors1 = [train_watch_errors1]

    loss = loss0 if loss0 is not None else 0.0
    losslist = []
    vallosslist = []
    tmp = 0.0
    for ee in errors1:
        for err in ee:
            if err is not None:
                tmp = err.mean(dtype=floatX)
                losslist = losslist+[tmp]
                loss = loss+tmp
    for ee in watch_errors1:
        for err in ee:
            if err is not None:
                tmp = err.mean(dtype=floatX)
                vallosslist = vallosslist+[tmp]
                # loss = loss+tmp
    for ee in train_watch_errors1:
        for err in ee:
            if err is not None:
                tmp = err.mean(dtype=floatX)
                losslist = losslist+[tmp]
                # loss = loss+tmp
    return loss, losslist, tagslice


def get_watchslice(watchpoints):
    trainwatch = {}
    valwatch = {}
    for tag, errs in watchpoints:
        if tag.startswith('train:'):
            trainwatch[tag] = errs
        else:
            valwatch[tag] = errs
    ig, train_values, train_tagslice = get_loss(trainwatch, [])
    ig, val_values, val_tagslice = get_loss(valwatch, [])
    return train_values, train_tagslice, val_values, val_tagslice
