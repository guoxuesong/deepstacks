#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import sys
sys.setrecursionlimit(50000)

from recurrent import *

import deepstacks
from deepstacks.macros import *
from deepstacks.lasagne import curr_layer,curr_stacks,curr_flags,curr_model
from operator import *

#import crossbatch_momentum

#def add(x,y):
#    return x+y
#def sub(x,y):
#    return x-y

def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, max sequence length, number of features)
    l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, 2))
    # The network also needs a way to provide a mask for each sequence.  We'll
    # use a separate input layer for that.  Since the mask only determines
    # which indices are part of the sequence for each batch entry, they are
    # supplied as matrices of dimensionality (N_BATCH, MAX_LENGTH)
    l_mask = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))

    l_out,stacks,layers,errors,watchpoints=deepstacks.lasagne.build_network(l_in,(
        (loop,MAX_LENGTH-1,( 
            (share,'rnn1',(
                ('input',(slice,(curr_loop_iterators,0),(add,(curr_loop_iterators,0),1)),
                    0,0,0,0,{}),
                (0,(2,),0,0,0,0,{}),
                (0,N_HIDDEN,0,0,0,0,{'dense','linear'}),
                ('hide1',),
                (0,N_HIDDEN,0,0,0,0,{'dense','linear'}),
                ((0,2),0,0,0,'hide1next',0,{'add':True,'nonlinearity':lambda x:lasagne.nonlinearities.tanh(theano.gradient.grad_clip(x,-100,100))}),
                ('mask',(slice,(curr_loop_iterators,0),(add,(curr_loop_iterators,0),1)),
                    0,0,0,0,{}),
                ((0,'hide1next','hide1'),0,0,0,'hide1',0,{'nonlinearity':lambda x:T.switch(T.repeat(x[:,0:1],N_HIDDEN,1),x[:,1:1+N_HIDDEN],x[:,1+N_HIDDEN:1+N_HIDDEN*2]),'shape':(N_BATCH,N_HIDDEN)}),
            )),
            )),
        (loop,MAX_LENGTH-1,(
            (share,'rnn2',(
                ('input',(slice,(sub,(sub,MAX_LENGTH,(curr_loop_iterators,0)),1),(sub,MAX_LENGTH,(curr_loop_iterators,0))),
                    0,0,0,0,{}),
                (0,(2,),0,0,0,0,{}),
                (0,N_HIDDEN,0,0,0,0,{'dense','linear'}),
                ('hide2',),
                (0,N_HIDDEN,0,0,0,0,{'dense','linear'}),
                ((0,2),0,0,0,'hide2next',0,{'add':True,'nonlinearity':lambda x:lasagne.nonlinearities.tanh(theano.gradient.grad_clip(x,-100,100))}),
                ('mask',(slice,(sub,(sub,MAX_LENGTH,(curr_loop_iterators,0)),1),(sub,MAX_LENGTH,(curr_loop_iterators,0))),
                    0,0,0,0,{}),
                ((0,'hide2next','hide2'),0,0,0,'hide2',0,{'nonlinearity':lambda x: T.switch(T.repeat(x[:,0:1],N_HIDDEN,1),x[:,1:1+N_HIDDEN],x[:,1+N_HIDDEN:1+N_HIDDEN*2]),'shape':(N_BATCH,N_HIDDEN)}),
            )),
            )),
        (('hide1','hide2'),1,0,0,0,0,{'dense':True,'nonlinearity':lasagne.nonlinearities.tanh}),
            ),{
                'mask':l_mask,
                'hide1':lasagne.layers.InputLayer(shape=(N_BATCH,N_HIDDEN),input_var=T.zeros((N_BATCH,N_HIDDEN))),
                'hide2':lasagne.layers.InputLayer(shape=(N_BATCH,N_HIDDEN),input_var=T.zeros((N_BATCH,N_HIDDEN))),
                })

#    # We're using a bidirectional network, which means we will combine two
#    # RecurrentLayers, one with the backwards=True keyword argument.
#    # Setting a value for grad_clipping will clip the gradients in the layer
#    # Setting only_return_final=True makes the layers only return their output
#    # for the final time step, which is all we need for this task
#    l_forward = lasagne.layers.RecurrentLayer(
#        l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
#        W_in_to_hid=lasagne.init.HeUniform(),
#        W_hid_to_hid=lasagne.init.HeUniform(),
#        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)
#    l_backward = lasagne.layers.RecurrentLayer(
#        l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
#        W_in_to_hid=lasagne.init.HeUniform(),
#        W_hid_to_hid=lasagne.init.HeUniform(),
#        nonlinearity=lasagne.nonlinearities.tanh,
#        only_return_final=True, backwards=True)
#    # Now, we'll concatenate the outputs to combine them.
#    l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])
#    # Our output layer is a simple dense connection, with 1 output unit
#    l_out = lasagne.layers.DenseLayer(
#        l_concat, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)

    target_values = T.vector('target_output')

    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)
    # The network output will have shape (n_batch, 1); let's flatten to get a
    # 1-dimensional vector of predicted values
    predicted_values = network_output.flatten()
    # Our cost will be mean-squared error
    cost = T.mean((predicted_values - target_values)**2)
    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out)
    # Compute SGD updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
    #updates = crossbatch_momentum.adamax(cost, all_params, learning_rate=2e-4, grads_clip=1.0)
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values, l_mask.input_var],
                            cost, updates=updates)
    compute_cost = theano.function(
        [l_in.input_var, target_values, l_mask.input_var], cost)

    # We'll use this "validation set" to periodically check progress
    X_val, y_val, mask_val = gen_data()

    print("Training ...")
    try:
        for epoch in range(num_epochs):
            for _ in range(EPOCH_SIZE):
                X, y, m = gen_data()
                train(X, y, m)
            cost_val = compute_cost(X_val, y_val, mask_val)
            print("Epoch {} validation cost = {}".format(epoch, cost_val))
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
