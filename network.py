# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 09:54:56 2016

@author: fassois

network.py
~~~~~~~~~~

A module to implement the stochastic, mini-batch, momentum-based, gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation in a fully vetorized fashion and regularization is also used.
I've added some more features as: locally adapting weights, choice for hidden layers,
choice for output layer, choice for cost function,
choice for weight initialization, constricting the norm of the weight matrices,
decaying learning rate and early stopping.
"""

from __future__ import division

#### Libraries

import json
import random
import matplotlib.pyplot as plt
import psutil
import time
from numpy import tanh
from collections import namedtuple, Counter
import numpy as np


class OutputLayer(object):
    """Used when the final output layer
    is a softmax layer. The cost function
    then used for a softmax layer is
    the log-likelihood cost function.
    """
    @staticmethod
    def softmax(z):
        """Compute softmax values for each sets of scores in z"""
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y`` in the output softmax layer.
        """
        return -np.sum(np.nanmax(np.log(a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the softmax output layer."""
        return a-y

#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):
    """All functions necessary for the quadratic cost function combined."""
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return np.sum(np.asarray(a -y)**2)

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.
        Clips the sigmoid derivative at 0.05 in order to avoid
        decaying updates and dying neurons.
        """
        return np.multiply((a-y), Network.sigmoid_prime(z).clip(0.05))


class CrossEntropyCost(object):
    """All functions necessary for the cross-entropy cost function combined."""
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-np.multiply(y, np.log(a))-np.multiply((1-y), np.log(1-a))))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return a-y


class Network(object):
    """Single neural network object."""

    def __init__(self, sizes, k=1, activation='sigmoid', weight_initialization='normal',
                 cost_function=CrossEntropyCost, softmax=True):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.tm = time.time()
        self.k = k

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.zeros(y).reshape(y, 1).astype('float16') for y in sizes[1:]]
        self.weight_initializater(weight_initialization)
        self.activation_type(activation)

        # Select the cost function to use with the network
        self.cost_function = cost_function

        # Select how much to restrict the norm of the weight matrices
        self.explodeClip = 20

        # Select the output layer to be used
        self.outputLayer = None
        if softmax:
            self.outputLayer = OutputLayer
            self.cost_function = OutputLayer

    def activation_type(self, activation):
        """Selects the activation type for the
        hidden layers and their respective
        derivatives used for backpropagation.
        """

        Activation = namedtuple('Activation', 'activation_type activation_prime')
        activation_type = {
            'sigmoid': Activation(self.sigmoid, self.sigmoid_prime),
            'tanh': Activation(self.tan_h, self.tan_h_prime),
            'relu': Activation(self.relu, self.relu_prime)
        }
        self.activation = activation_type[activation]
        self.type = activation

    def weight_initializater(self, weight_initialization):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        Also included uniform distribution for initialization.
        """

        if weight_initialization == 'uniform':
            self.weights = [2*np.random.rand(y, x).astype('float16') - 1
                            for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        elif weight_initialization == 'normal':
            self.weights = [np.random.randn(y, x).astype('float16') / np.sqrt(x)
                            for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        elif weight_initialization == 'Relu':
            self.weights = [np.random.randn(y, x).astype('float16') / np.sqrt(x)
                            for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for ind, (b, w) in enumerate(zip(self.biases, self.weights)):
            if self.outputLayer and ind == self.num_layers - 2:
                a = self.outputLayer.softmax(np.dot(w, a)+b)
                break
            a = self.activation.activation_type(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, lmbda=0.0, n_epochs=10, verbose=True, local_weights=True):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the validation data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        Additional arguments innclude lmbda for decreasing learning rate
        and to select local weights depending on the size of the layer.
        """

        training_data = self.pre_process(training_data)

        self.eta = eta

        n = len(training_data)
        losses, accuracies = [], []

        if test_data:
            X_test, Y_test, T, n_test = self.process_test_data(test_data)

        momentum = [np.zeros(w.shape) for w in self.weights]

        if local_weights:
            self.local_weights()

        for j in xrange(epochs):

            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]

            for ind, mini_batch in enumerate(mini_batches):

                nabla_b, nabla_w, activations = self.update_mini_batch(mini_batch, j, momentum, lmbda, n, local_weights)

            if test_data:

                outputs, test_results, positives = self.evaluate(X_test, Y_test)
                print "Epoch {0}: {1} / {2}".format(
                    j, positives, n_test)
                if verbose:
                    self.validate(X_test, Y_test, test_results, outputs, j, positives, losses, accuracies, T)

                    if self.early_stopping(losses, n_epochs, local_weights):
                        break

    def update_mini_batch(self, mini_batch, j, momentum, lmbda, n, local_weights):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate. The momentum gradient approach is used
        to improve the learning.
        Reference: https://www.willamette.edu/~gorr/classes/cs449/momrate.html
        """

        # 3 dimensional matrices that hold the weight and bias differences for mini-batch samples
        batch_size = len(mini_batch)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        X, Y = zip(*mini_batch)
        X = np.asarray(X)
        X = np.asmatrix(X.T)

        Y = np.asarray(Y)
        Y = np.asmatrix(Y.T)

        # Perform a backpropagation step for the minibatch
        nabla_b, nabla_w, activations = self.backprop(X, Y, batch_size, nabla_b, nabla_w, j)

        # Update the weights
        if local_weights:

            self.weights = [(1-self.eta_local[hidden_layer]*(lmbda/n))*w -
                            (self.eta_local[hidden_layer]/batch_size)*nw + v
                            for hidden_layer, (w, nw, v) in enumerate(zip(self.weights, nabla_w, momentum))]
            self.biases = [b-(self.eta_local[hidden_layer]/batch_size)*nb
                           for hidden_layer, (b, nb) in enumerate(zip(self.biases, nabla_b))]
            momentum = [0.3 * v -(self.eta_local[hidden_layer]/batch_size)*nw
                        for hidden_layer, (v, nw) in enumerate(zip(momentum, nabla_w))]

        else:
            self.weights = [(1-self.eta*(lmbda/n))*w-(self.eta/batch_size)*nw + v
                            for w, nw, v in zip(self.weights, nabla_w, momentum)]
            self.biases = [b-(self.eta/batch_size)*nb
                           for b, nb in zip(self.biases, nabla_b)]
            momentum = [0.3 * v -(self.eta/batch_size)*nw
                        for v, nw in zip(momentum, nabla_w)]

        # Adjust the weights if the norm is larger than the limit
        self.weights = [(self.explodeClip / np.linalg.norm(w)) * w
                        if np.linalg.norm(w) > self.explodeClip
                        else w for w in self.weights]

        return nabla_b, nabla_w, activations



    def backprop(self, X, Y, no_inputs, nabla_b, nabla_w, j):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        # feedforward

        activation = X
        activations = [X] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for ind, (b, w) in enumerate(zip(self.biases, self.weights)):
            no_outputs = w.shape[0]
            sumproducts = w.dot(activation).reshape([1, no_outputs, -1])
            z = sumproducts + b
            zs.append(z)
            if self.outputLayer and ind == self.num_layers - 2:
                activation = self.outputLayer.softmax(z)
                activations.append(activation)
                break

            activation = self.activation.activation_type(z)
            activations.append(activation)

        # backward pass

        delta = self.cost_function.delta(z, activations[-1], Y)

        nabla_b[-1] = np.asmatrix(delta).sum(axis=1)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = self.activation.activation_prime(z).clip(0.05)
            delta = np.multiply(np.dot(self.weights[-l+1].transpose(), delta), sp)

            nabla_b[-l] = np.asmatrix(delta).sum(axis=1)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w, activations)

    def local_weights(self):
        """
        Set local learning rates per layer to
        achieve a well-conditioned network
        Reference: https://www.willamette.edu/~gorr/classes/cs449/precond.html
        """

        # list to store the error variance, layer by layer
        error_variance = [np.zeros(b.shape) for b in self.biases]

        # list to store all the local learning rates, layer by layer
        self.eta_local = [0 for x in xrange(len(self.sizes)-1)]

        error_variance[-1] = np.repeat(1/np.sqrt(self.sizes[-2]), self.sizes[-1])
        self.eta_local[-1] = self.eta/(np.sqrt(self.sizes[-2])*np.sqrt(np.mean(error_variance[-1])))

        for l in xrange(2, self.num_layers):
            error_variance[-l] = np.repeat(np.sum(error_variance[-l+1]) / np.sqrt(self.sizes[-l-1]), self.sizes[-l])
            self.eta_local[-l] = self.eta / (np.sqrt(self.sizes[-l-1]) * np.sqrt(np.mean(error_variance[-l])))

    def early_stopping(self, losses, n_epochs, local_weights):
        """
        Decrease the learning rate if after n_epochs
        the learning hasn't improved and stop early
        if the learning rate reaches a threshold.
        """
        if len(losses) > n_epochs:
            if losses[-n_epochs] <= losses[-1]:
                if local_weights:
                    self.eta_local = [x / 2.0 for x in self.eta_local]
                    if np.mean(self.eta_local) <= self.eta / 128.0:
                        return True
                else:
                    self.eta = self.eta / 2.0
                    if self.eta <= 1/128.0:
                        return True

    def evaluate(self, X_test, Y_test):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        outputs = self.feedforward(X_test)
        test_results = np.asmatrix(np.apply_along_axis(np.argmax, 0, outputs))

        return outputs, test_results, (Y_test == test_results).sum()

    def validate(self, X_test, Y_test, test_results, outputs, j, positives, losses, accuracies, T):
        """
        Compute the loss function on the validation data.
        Also output a random example of a digit after every epoch
        and the network's guess.
        """

        loss = self.cost_function.fn(outputs, T) / len(T.T)

        # Throttle code - DO NOT REMOVE THIS. Prevents system crashes.
        if psutil.virtual_memory().percent > 95 or psutil.cpu_times_percent().idle < 5:
            time.sleep(1)
        losses.append(loss)
        if self.type == 'sigmoid':
            pass
        else:
            testi = random.choice(range(Y_test.shape[1]))
            test = X_test.T[testi]
            plt.imshow(test.reshape(28, 28), cmap='gray')
            plt.show()
            cls = int(test_results.T[testi])
            print("Prediction: %d confidence=%0.2f" %
                  (cls, np.asarray(outputs).T[testi][cls]/np.sum(outputs.T[testi])))

        accpct = 100*positives/X_test.shape[1]
        accuracies.append(accpct)

        if j % 10 == 9:

            print("    Speed: %0.2f s/pass" % ((time.time() - self.tm)/(j+0.01)))
            print("    Accuracy: %d/%d, %0.3f%%" % (positives, X_test.shape[1], accpct))

            plt.plot(np.log(losses), color='blue')
            plt.title("Log loss")
            plt.show()
            plt.plot(accuracies, color='blue')
            plt.title("Accuracy")
            plt.show()

            '''
            import json
            with open("results-%d.json" % self.tm, "w") as f:
                f.write(json.dumps({'losses': losses, 'accuracies': accuracies}, indent=1))
            '''

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives
        for the output activations."""
        return output_activations-y

    #### Miscellaneous functions
    def sigmoid(self, z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(self.k*(-z)))

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.k * np.multiply(self.sigmoid(z), (1-self.sigmoid(z)))

    def tan_h(self, z):
        """The hyperbolic tangent function."""
        return tanh(z)

    def tan_h_prime(self, z):
        """Derivative of the hyperbolic tangent function."""
        return 1 - np.asarray(self.tan_h(z))**2

    def relu(self, z):
        """Relu activation function"""
        return np.maximum(z, 0)

    def vector(func):
        """ Vectorize functions"""
        def inner(*args):
            wrapped = np.vectorize(func)
            return wrapped(*args)
        return inner

    @vector
    def relu_prime(self, z):
        """Derivative of the relu activation function"""
        if z > 0:
            return 1
        else:
            return 0.05


    def pre_process(self, training_data):
        """ Preprocess the data in 3 dimensional matrices
        so as to implement vectorized gradient descent.
        Also for sigmoid activation function the input data
        is normalized and preprocessed for preconditioning."""

        # Seperate the X's and Y's in the training data and save them as matrices
        X, Y = zip(*training_data)
        X = np.asarray(X)
        X = np.asmatrix(X.T)

        Y = np.asarray(Y)
        Y = np.asmatrix(Y.T)

        if self.type == 'sigmoid':

            self.mean = np.mean(X, axis=1)
            X -= self.mean

            # get the data covariance matrix
            cov = np.dot(X, X.T) / X.shape[1]

            self.U, self.S, _ = np.linalg.svd(cov)

            # decorrelate the data
            Xrot = np.dot(self.U, X)

            # whiten the data:
            # divide by the eigenvalues (which are square roots of the singular values)
            Xwhite = Xrot.T / np.sqrt(self.S + 1e-5)
            X = Xwhite.T

        Y = np.hsplit(Y, 50000)
        X = np.hsplit(X, 50000)

        training_data = zip(X, Y)
        return training_data

    def process_test_data(self, test_data):
        """Preprocess the test data correspondingly."""
        n_test = len(test_data)

        X_test, Y_test = zip(*test_data)
        X_test = np.asarray(X_test)
        X_test = np.asmatrix(X_test.T)

        if self.type == 'sigmoid':

            X_test -= self.mean

            # decorrelate the data
            Xrot = np.dot(self.U, X_test)

            # whiten the data:
            # divide by the eigenvalues (which are square roots of the singular values)
            Xwhite = Xrot.T / np.sqrt(self.S + 1e-5)

            X_test = Xwhite.T

        Y_test = np.asarray(Y_test)
        Y_test = np.asmatrix(Y_test.T)

        T = np.zeros((len(Y_test.T), 10), dtype='uint8').T
        for i in range(len(Y_test.T)):
            T[Y_test.T[i], i] = 1

        return X_test, Y_test, T, n_test

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import cPickle
import gzip

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('data\\mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

training_data, validation_data, test_data = load_data_wrapper()



class Ensemble(object):
    """Creates an ensmble of neural networks
    and averages their predictions for
    classifying the digits.
    """

    def __init__(self, no_networks):
        """Initializes an ensemble of networks.
        """
        for i in range(no_networks):
            network = Network([784, 256, 128, 10], activation='tanh', weight_initialization='normal', softmax=True)
            setattr(self, "network%s" % (i+1), network)

    def get(self):
        """Returns the ensemble."""
        return self.__dict__

    def test_data(self):
        """Preprocess the test data data."""
        first_network = ensemble.get().values()[0]
        X_test, Y_test, _, n_test = first_network.process_test_data(test_data)
        return X_test, Y_test, n_test

    def train(self):
        """Preprocess the training data data.
        Train all networks of the ensemble."""
        X_test, Y_test, n_test = self.test_data()
        predictions = dict()
        for name, network in self.get().items():
            print ' /n Training /n %s' % (name)
            network.SGD(training_data, 200, 100, .5, test_data=validation_data, lmbda=.5, local_weights=False)
            _, test_results, _ = network.evaluate(X_test, Y_test)
            predictions[name] = test_results
        return predictions, n_test, Y_test

    def test_accuracy(self):
        """Pools all predictions to get the average prediction."""
        results = []
        predictions, n_test, Y_test = self.train()
        for i in range(n_test):
            vote = Counter([test_results.item(0, i) for test_results in predictions.values()])
            results.append(vote.most_common()[0][0])
        return len([1 for (a, b) in zip(Y_test.tolist()[0], results) if a == b])


# Run an ensemble of five networks.
ensemble = Ensemble(5)
accuracy = ensemble.test_accuracy()