import theano
import numpy as np
def floatXconst(val):
    if theano.config.floatX=='float32':
        return np.float32(val)
    elif  theano.config.floatX=='float64':
        return np.float64(val)
    elif  theano.config.floatX=='float16':
        return np.float16(val)
    else:
        return val

