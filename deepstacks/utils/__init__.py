import theano
import numpy as np

def floatX(val):
    return vars(np)[theano.config.floatX](val)

