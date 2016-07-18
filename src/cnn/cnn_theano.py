# -*- coding: utf-8 -*-
##
# @brief CNN for relation extraction, based on Theano and Lasagne
# @author ss

from __future__ import print_function
import os
import time
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import theano
import theano.tensor as T
import lasagne
from collections import OrderedDict


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """ Generate mini-batches for training and validating phase.
    """
    assert inputs.shape[0] == len(targets)
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0], batchsize):
        end_idx = min(start_idx + batchsize, inputs.shape[0])
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield inputs[excerpt], targets[excerpt]


def build_net(input_var, input_shape, windows):
    """ Build network using theano and lasagne.
    Args:
        input_var: T.tensor4
        input_shape(tuple): (channel, height, width)
        windows(list): list of window sizes

    Returns:
        network
    """
    channel, height, width = input_shape
    l_in = lasagne.layers.InputLayer(shape=(None, channel, height, width), input_var=input_var)
    conv_layers = []
    for window_size in windows:
        l_conv2d = lasagne.layers.Conv2DLayer(l_in, num_filters=150, filter_size=(window_size, width),
                                              stride=(1, 1),
                                              pad=0,
                                              untie_biases=False,
                                              W=lasagne.init.GlorotUniform(),
                                              b=lasagne.init.Constant(0.),
                                              nonlinearity=lasagne.nonlinearities.rectify)
        l_conv2d = lasagne.layers.MaxPool2DLayer(l_conv2d, pool_size=(height - window_size + 1, 1))
        conv_layers.append(l_conv2d)
    l_conv = lasagne.layers.ConcatLayer(conv_layers, axis=1)
    l_z = lasagne.layers.DropoutLayer(l_conv, p=0.5)
    l_dense = lasagne.layers.DenseLayer(l_z, num_units=19,
                                        nonlinearity=lasagne.nonlinearities.softmax,
                                        W=lasagne.init.GlorotUniform())
    return l_dense


class CNN(object):
    def __init__(self, param):
        self.batch_size = param['batch_size']
        self.learning_rate = param['learning_rate']
        self.rho = param['rho']
        self.epsilon = param['epsilon']
        self.num_epochs = param['num_epochs']
        self.predict_fn = None

    def fit(self, X_train, y_train, X_valid, y_valid):
        """ Train
        Args:
            X_train, X_valid: 4D-ndarray, whose shape is (number of samples, channel=1, sequence length,
                    vector length of word representation)
            y_train, y_valid: 1D-ndarray, labels

        Returns:
            self
        """
        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
        # Create Input tensors
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')
        # Create neural network model
        network = build_net(input_var, input_shape, windows=[2, 3, 4, 5])
        # Create train loss
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        # Create valid loss
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean()
        # Create valid accuracy
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
        # Get training parameters
        params = lasagne.layers.get_all_params(network, trainable=True)
        # Get gradients
        grads = theano.grad(loss, params)
        # Now create update rules for adadelta
        updates = OrderedDict()
        one = T.constant(1)  # (Using theano constant to prevent upcasting of float32)
        for param, grad in zip(params, grads):
            value = param.get_value(borrow=True)
            # accu: accumulate gradient magnitudes
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
            # delta_accu: accumulate update magnitudes (recursively!)
            delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                       broadcastable=param.broadcastable)
            # update accu (as in rmsprop)
            accu_new = self.rho * accu + (one - self.rho) * grad ** 2
            updates[accu] = accu_new
            # compute parameter update, using the 'old' delta_accu
            update = (grad * T.sqrt(delta_accu + self.epsilon) / T.sqrt(accu_new + self.epsilon))
            updates[param] = param - self.learning_rate * update
            """ It seems this norm constraint does not matter
            # if param == network.W:  # that is the param W in the final layer
            #     updates[param] = lasagne.updates.norm_constraint(updates[param], 3)
            """
            # update delta_accu (as accu, but accumulating updates)
            delta_accu_new = self.rho * delta_accu + (one - self.rho) * update ** 2
            updates[delta_accu] = delta_accu_new
        # Now create theano functions for training and validating
        train_fn = theano.function(inputs=[input_var, target_var], outputs=loss, updates=updates)
        valid_fn = theano.function(inputs=[input_var, target_var], outputs=[test_loss, test_acc])
        # Start training
        for epoch in range(self.num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, y_train, self.batch_size, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1
            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_valid, y_valid, self.batch_size, shuffle=False):
                inputs, targets = batch
                err, acc = valid_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, self.num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
        # Create theano function for model prediction
        self.predict_fn = theano.function(inputs=[input_var], outputs=prediction)
        return self

    def transform(self, inputs):
        """ Predict
        Args:
            inputs: 4D-ndarray, whose shape is (number of samples, channel=1, sequence length,
                    vector length of word representation)
        Returns:
            outputs(list): predictions
        """
        def _minibatches(inputs, batch_size):
            """ Generate mini-batches (data without labels) without shuffle.
            """
            for start_idx in range(0, inputs.shape[0], batch_size):
                end_idx = min(start_idx+batch_size, inputs.shape[0])
                excerpt = slice(start_idx, end_idx)
                yield inputs[excerpt]
        # Start predicting
        outputs = []
        for minibatch in _minibatches(inputs, batch_size=self.batch_size):
            output = self.predict_fn(minibatch)
            output = np.argmax(output, axis=1)
            outputs += output.tolist()
        return outputs
