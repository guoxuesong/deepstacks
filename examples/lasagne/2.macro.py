#!/usr/bin/env python

from mnist import *

import deepstacks
from deepstacks.macros import *
from deepstacks.lasagne import curr_layer,curr_stacks,curr_flags,curr_model

# Macro 1 dropout: connect current layer to a DropoutLayer with p=p
def dropout(p):
    return ((0,0,0,0,0,0,{'layer':(lasagne.layers.DropoutLayer,curr_layer,{'p':p})}),)

def build_mlp(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(500, 1, 28, 28),
                                     input_var=input_var)

    network,stacks,layers,errors,watchpoints=deepstacks.lasagne.build_network(l_in,(
	# call macro dropout, this looks cleaner
        (dropout,0.2),
        (0,500,0,0,0,0,{'dense'}),
        (dropout,0.5),
        (0,500,0,0,0,0,{'dense'}),
        (dropout,0.5),
        (0,10,0,0,0,0,{'dense':True,'nonlinearity':lasagne.nonlinearities.softmax}),
        ))
    return network


# Macro 2 myloop: repeat l for n times, this equals to 'loop' in macros.py
def myloop(n,l):
    res=()
    for i in range(n):
        res+=macros(l)
    return res

# Macro 3 dropout_ifneed: dropout only if p!=0
def dropout_ifneed(p):
    return ((0,0,0,0,0,0,{'layer':(lasagne.layers.DropoutLayer,curr_layer,{'p':p})}) if p else (0,),)

def build_custom_mlp(input_var=None, depth=2, width=800, drop_input=.2,
                     drop_hidden=.5):
    l_in = lasagne.layers.InputLayer(shape=(500, 1, 28, 28),
                                     input_var=input_var)
    network,stacks,layers,errors,watchpoints=deepstacks.lasagne.build_network(l_in,(
        (dropout_ifneed,drop_input),
        (myloop,depth,(
            (0,width,0,0,0,0,{'dense'}),
            (dropout_ifneed,drop_hidden),
            )),
        (0,10,0,0,0,0,{'dense':True,'nonlinearity':lasagne.nonlinearities.softmax}),
        ))
    return network


def build_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(500, 1, 28, 28),
                                        input_var=input_var)
    network,stacks,layer,errors,watchpoints=deepstacks.lasagne.build_network(network,(
        (0,32,5,1,0,0,{}),
        (0,0,2,2,0,0,{'maxpool'}),
        (0,32,5,1,0,0,{}),
        (0,0,2,2,0,0,{'maxpool'}),
        (dropout,0.5),
        (0,256,0,0,0,0,{'dense'}),
        (dropout,0.5),
        (0,10,0,0,0,0,{'dense':True,'nonlinearity':lasagne.nonlinearities.softmax}),
        ))
    return network

# Following is just copied from mnist.py

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(model='mlp', num_epochs=500):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'mlp':
        network = build_mlp(input_var)
    elif model.startswith('custom_mlp:'):
        depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
        network = build_custom_mlp(input_var, int(depth), int(width),
                                   float(drop_in), float(drop_hid))
    elif model == 'cnn':
        network = build_cnn(input_var)
    else:
        print("Unrecognized model type %r." % model)
        return

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)
