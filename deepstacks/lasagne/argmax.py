import theano
import theano.tensor as T
import numpy as np
from .. import utils

floatX = theano.config.floatX


def goroshin_max(z, axis=(1, ), beta=3, keepdims=False):
    return (z*T.exp(beta*z)/T.exp(beta*z)
            .sum(axis=axis, keepdims=True)).sum(axis=axis, keepdims=keepdims)


def goroshin_argmax(z, shape, axis=(1, ), beta=3, epsilon=0.0001):
    z = z/(abs(T.max(z))+utils.floatX(epsilon))
    a = ()
    for t in axis:
        a += (slice(0, shape[t]), )
    xyshape = list(shape)+[]
    for i in range(len(shape)):
        if i not in axis:
            xyshape[i] = 1
    xy = T.mgrid[a]
    b = T.exp(beta*z)/T.exp(beta*z).sum(axis, keepdims=True)
    res = []
    for i in range(len(axis)):
        x = ((xy[i].astype(floatX)).reshape(xyshape)*b).sum(axis=axis)
        res += [x]
    return T.stack(res, axis=1)


def goroshin_unargmax(z, shape, axis=(1, ), sigma=1.0, epsilon=0.0001):
    assert len(set([shape[ax] for ax in axis])) == 1
    assert len(shape) >= axis[-1]
    scale = utils.floatX(shape[axis[0]])
    sigma = utils.floatX(sigma)
    sigma /= scale
    z = z/scale
    a = ()
    for t in axis:
        a += (slice(0, shape[t]), )
    xyshape = list(shape)+[]
    zshape = list(shape)+[]
    for i in range(len(shape)):
        if i not in axis:
            xyshape[i] = 1
        else:
            zshape[i] = 1
    xy = T.mgrid[a]
    s = T.zeros(shape, dtype=floatX)
    for i in range(len(axis)):
        x = (xy[i].astype(floatX)/scale).reshape(xyshape).repeat(shape[0], 0)
        s += ((x-z[:, i].reshape(zshape))**2)
    d = utils.floatX(1.0)/(utils.floatX(1.0)+s/(sigma**utils.floatX(2)))
    return d

if __name__ == '__main__':
    z = np.zeros((16, 3, 8, 8, 2, 2), dtype='float32')
    z[:, :, :, :, 1, 0] = 1
    print z[0, 0, 0, 0, :, :]
    xy = goroshin_argmax(z, (16, 3, 8, 8, 2, 2), axis=(4, 5), beta=32)
    print xy.eval()
    print xy.shape.eval()
    recon = goroshin_unargmax(xy, (16, 3, 8, 8, 2, 2), axis=(4, 5))
    print recon.shape.eval()
    print recon[0, 0, 0, 0, :, :].eval()
