#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Example that trains a small multi-layer perceptron with multiple branches on MNIST data.

Branch nodes are used to indicate points at which different layer sequences diverge

The topology of the network is:

 cost1      cost3
  |          /
 m_l4      b2_l2
  |        /
  | ___b2_l1
  |/
 m_l3       cost2
  |          /
 m_l2      b1_l2
  |        /
  | ___b1_l1
  |/
  |
 m_l1
  |
  |
 data

Usage:

    python examples/mnist_branch.py

"""

from neon import logger as neon_logger
from neon.callbacks.callbacks import Callbacks
from neon.data import ArrayIterator, MNIST
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine, BranchNode, Multicost, SingleOutputTree, Tree, Sequential
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, Softmax
from neon.transforms import CrossEntropyBinary, CrossEntropyMulti, Misclassification
from neon.util.argparser import NeonArgparser

import deepstacks
from deepstacks.macros import *
from deepstacks.neon import curr_layer,curr_stacks,curr_flags,curr_model

# parse the command line arguments
parser = NeonArgparser(__doc__)

args = parser.parse_args()

# load up the mnist data set
dataset = MNIST(path=args.data_dir)
train_set = dataset.train_iter
valid_set = dataset.valid_iter

#(X_train, y_train), (X_test, y_test), nclass = dataset.load_data()
#train_set = ArrayIterator([X_train], y_train, nclass=nclass)
#valid_set = ArrayIterator([X_test], y_test, nclass=nclass)

# setup weight initialization function
init_norm = Gaussian(loc=0.0, scale=0.01)

normrelu = dict(init=init_norm, activation=Rectlin())
normsigm = dict(init=init_norm, activation=Logistic(shortcut=True))
normsoft = dict(init=init_norm, activation=Softmax())

# setup model layers
b1 = BranchNode(name="b1")
b2 = BranchNode(name="b2")

def insert_branch_layer(network,b):
    return Sequential(layers=(network,b))

# NOTE: neon's orig layers dose not add bias, so it's much faster than us
l_in = deepstacks.neon.InputLayer((None,)+train_set.shape,'image')
l_y = deepstacks.neon.InputLayer((None,),'y')
network,stacks,paramlayers,errors,watchpoints=deepstacks.neon.build_network(l_in,(
        (0,100,0,0,'m_l1',0,{'dense'}),
        #(0,0,0,0,0,0,{'layer':(insert_branch_layer,curr_layer,b1)}),
        (0,32,0,0,'m_l2',0,{'dense'}),
        (0,16,0,0,'m_l3',0,{'dense'}),
        #(0,0,0,0,0,0,{'layer':(insert_branch_layer,curr_layer,b2)}),
        (0,10,0,0,'m_l4',0,{'dense':True,'nonlinearity':Softmax()}),
        ('m_l1',0,0,0,0,0,{}),
        #(0,0,0,0,0,0,{'layer':b1}),
        (0,16,0,0,'b1_l1',0,{'dense'}),
        (0,10,0,0,'b1_l2',0,{'dense':True,'nonlinearity':Logistic(shortcut=True),'equal':['target','b1',CrossEntropyBinary()]}),
        ('m_l3',0,0,0,0,0,{}),
        #(0,0,0,0,0,0,{'layer':b2}),
        (0,16,0,0,'b2_l1',0,{'dense'}),
        (0,10,0,0,'b2_l2',0,{'dense':True,'nonlinearity':Logistic(shortcut=True),'equal':['target','b2',CrossEntropyBinary()]}),
        ('m_l4',)
        ),{
            'target':l_y
            })

cost = GeneralizedCost(costfunc=CrossEntropyBinary())
cost,layers,tagslice = deepstacks.neon.get_loss(errors,watchpoints,cost)

print 'network:',network
print 'extra layers:',layers

network = Tree([network]+layers)

layers = [network]

inputs = deepstacks.neon.get_inputs(network)
targets = deepstacks.neon.get_targets(cost)

print 'inputs,targets:',inputs,targets
#assert tuple(inputs)==('image',)

#p1 = [l_in,
#        Affine(nout=100, name="m_l1", **normrelu),
#      b1,
#      Affine(nout=32, name="m_l2", **normrelu),
#      Affine(nout=16, name="m_l3", **normrelu),
#      b2,
#      Affine(nout=10, name="m_l4", **normsoft)]
#
#p2 = [b1,
#      Affine(nout=16, name="b1_l1", **normrelu),
#      Affine(nout=10, name="b1_l2", **normsigm)]
#
#p3 = [b2,
#      Affine(nout=16, name="b2_l1", **normrelu),
#      Affine(nout=10, name="b2_l2", **normsigm)]
#
#network=Tree([p1, p2, p3])
#inputs = deepstacks.neon.get_inputs(network)
#print inputs
#exit(0)


# setup cost function as CrossEntropy
#cost = Multicost(costs=[GeneralizedCost(costfunc=CrossEntropyMulti()),
#                        GeneralizedCost(costfunc=CrossEntropyBinary()),
#                        GeneralizedCost(costfunc=CrossEntropyBinary())],
#                 weights=[1, 0., 0.])

# setup optimizer
optimizer = GradientDescentMomentum(
    0.1, momentum_coef=0.9, stochastic_round=args.rounding)

# initialize model object
#alphas = [1, 0.25, 0.25]
#mlp = Model(layers=SingleOutputTree([p1, p2, p3], alphas=alphas))

mlp = Model(layers=layers)

# setup standard fit callbacks
callbacks = Callbacks(mlp, eval_set=valid_set, multicost=True, **args.callback_args)

# run fit
mlp.fit(train_set, optimizer=optimizer,
        num_epochs=args.epochs, cost=cost, callbacks=callbacks)

# TODO: introduce Multicost metric support.  The line below currently fails
# since the Misclassification metric expects a single Tensor not a list of
# Tensors
neon_logger.display('Misclassification error = %.1f%%' %
                    (mlp.eval(valid_set, metric=Misclassification()) * 100))
