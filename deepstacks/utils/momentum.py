#!/usr/bin/env python
#coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

# (2)比(1)更好让我非常吃惊，理论上，(3)和(1)是等价的，(2)只用了(1)的1/5的计算能
# 力，(1)使用了“准确的”梯度，(2)使用了5次采样平均的梯度，momentum经过调整使得在
# 时间轴上覆盖相同数量的样本数量。为什么(2)会比(1)好呢？存在比准确的梯度更准确
# 的梯度吗？我们能够分析的是，假设我们有两个维度x,y，z=(a/2)x+(b/2)y，那么准确
# 的梯度是(a/2,b/2)，如果我们做两次采样，样本是稀疏的，我们采样到 z1=(a+0)x1,
# z2=(b+0)y2，梯度平均得到(a/2,b/2)，如果样本不是稀疏的，我们采样到
# z1=(a/2+0)x1+(b/2+0)y1, z2=(a/2+0)x2+(b/2+0)y2，梯度平均得到(a/2,b/2)。
#
# 错了，(5)和(1)是等价的，所以实验符合预期，(5)比(1)差一些是合理的。

#mnist cnn 控制组  nodrop  minibatch=5     x100  100次迭代 0.164350 95.32 lr=0.0001  momentum=0.999 (1)
#mnist cnn 控制组  nodrop  minibatch=1     x100  100次迭代 0.120086 96.25 lr=0.00001 momentum=0.999
#mnist cnn 控制组  nodrop  minibatch=1     x100  100次迭代 2.347241 10.10 lr=0.0001  momentum=0.999
#mnist cnn cross   nodrop  minibatch=1(10) x100  100次迭代 0.337935 89.74 lr=0.0001  momentum=0.999
#mnist cnn cross   nodrop  minibatch=1(5)  x100  100次迭代 0.214818 94.15 lr=0.0001  momentum=0.999 (4)
#mnist cnn cross   nodrop  minibatch=1(5)  x100  500次迭代 0.235046 94.52 lr=0.0001  momentum=0.999 (5)
#mnist cnn cross   nodrop  minibatch=1(5)  x100  100次迭代 0.137648 95.73 lr=0.0001  momentum=0.995 (2)
#mnist cnn cross   nodrop  minibatch=1(5)  x100  500次迭代 0.064822 97.97 lr=0.0001  momentum=0.995 (3)
#mnist cnn cross   nodrop  minibatch=1(5)  x100  100次迭代 0.421069 86.62 lr=0.00001 momentum=0.999

#mnist cnn 控制组  nodrop  minibatch=5     x100  100次迭代 ? ? lr=0.001  momentum=0.999
#mnist cnn cross   nodrop  minibatch=1(5)  x100  100次迭代 ? ? lr=0.001  momentum=0.995

from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne import utils
import lasagne

def get_or_compute_grads(loss_or_grads, params):
    """Helper function returning a list of gradients

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to return the gradients for

    Returns
    -------
    list of expressions
        If `loss_or_grads` is a list, it is assumed to be a list of
        gradients and returned as is, unless it does not match the length
        of `params`, in which case a `ValueError` is raised.
        Otherwise, `loss_or_grads` is assumed to be a cost expression and
        the function returns `theano.grad(loss_or_grads, params)`.

    Raises
    ------
    ValueError
        If `loss_or_grads` is a list of a different length than `params`, or if
        any element of `params` is not a shared variable (while we could still
        compute its gradient, we can never update it and want to fail early).
    """
    if any(not isinstance(p, theano.compile.SharedVariable) for p in params):
        raise ValueError("params must contain shared variables only. If it "
                         "contains arbitrary parameter expressions, then "
                         "lasagne.utils.collect_shared_vars() may help you.")
    if isinstance(loss_or_grads, list):
        if not len(loss_or_grads) == len(params):
            raise ValueError("Got %d gradient expressions for %d parameters" %
                             (len(loss_or_grads), len(params)))
        return loss_or_grads
    else:
        return theano.grad(loss_or_grads, params, disconnected_inputs='warn')


def sgd(loss_or_grads, params, learning_rate, grads_clip=False):
    """Stochastic Gradient Descent (SGD) updates

    Generates update expressions of the form:

    * ``param := param - learning_rate * gradient``

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    """
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        if type(grads_clip)==float and grads_clip>0:
            grad = T.clip(grad,utils.floatX(-grads_clip),utils.floatX(grads_clip))
        elif grads_clip:
            grad = T.clip(grad,utils.floatX(-1.0),utils.floatX(1.0))
        updates[param] = param - learning_rate * grad

    return updates

def apply_momentum(updates, params=None, momentum=0.9, average=1):

    N=average

    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)

    #count = theano.shared(np.zeros((1,), dtype='int8'))
    #updates[count]=(count+1) % N
    for param in params:
        value = param.get_value(borrow=True)
        sumval = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
        count = theano.shared(np.zeros(value.shape, dtype='int8'),
                                 broadcastable=param.broadcastable)
        updates[count]=(count+1) % N

        avg=(sumval+updates[param])/N

        x = momentum * velocity + avg

        updates[sumval] = T.switch(T.eq(count,N-1),T.zeros_like(sumval),sumval+updates[param])
        updates[velocity] = T.switch(T.eq(count,N-1),x - param,velocity)
        updates[param] = T.switch(T.eq(count,N-1),x,param)

    return updates

def momentum(loss_or_grads, params, learning_rate, momentum=0.9, average=1, grads_clip=False):

    updates = sgd(loss_or_grads, params, learning_rate)
    return apply_momentum(updates, momentum=momentum, average=average)

def adamax(loss_or_grads, params, learning_rate=0.002, beta1=0.9,
           beta2=0.999, epsilon=1e-8, average=1, grads_clip=False, noise=False):

    N=average

    all_grads = get_or_compute_grads(loss_or_grads, params)
    t_prev = theano.shared(utils.floatX(0.))
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    t = t_prev + 1
    a_t = learning_rate/(one-beta1**t)

    srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))

    for param, g_t in zip(params, all_grads):
        value = param.get_value(borrow=True)
        if type(noise)==float and noise>0:
            g_t = g_t+srng.normal(value.shape,avg=0.0,std=noise)
        elif noise:
            g_t = g_t+srng.normal(value.shape,avg=0.0,std=1e-8)

        if type(grads_clip)==float and grads_clip>0:
            g_t = T.clip(g_t,utils.floatX(-grads_clip),utils.floatX(grads_clip))
        elif grads_clip:
            g_t = T.clip(g_t,utils.floatX(-1.0),utils.floatX(1.0))
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        u_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        sumval = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
        count = theano.shared(np.zeros(value.shape, dtype='int8'),
                                 broadcastable=param.broadcastable)
        updates[count]=(count+1) % N

        avg=(sumval+g_t)/N

        m_t = beta1*m_prev + (one-beta1)*avg
        u_t = T.maximum(beta2*u_prev, abs(avg))
        step = a_t*m_t/(u_t + epsilon)

        updates[sumval] = T.switch(T.eq(count,N-1),T.zeros_like(sumval),sumval+g_t)
        updates[m_prev] = T.switch(T.eq(count,N-1),m_t,m_prev)
        updates[u_prev] = T.switch(T.eq(count,N-1),u_t,u_prev)
        updates[param] = T.switch(T.eq(count,N-1),param - step,param)

    count = theano.shared(0)
    updates[count]=(count+1) % N
    updates[t_prev] = T.switch(T.eq(count,N-1),t,t_prev)
    return updates
