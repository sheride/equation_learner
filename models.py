#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:44:11 2019 ish

@author: elijahsheridan
"""

from __future__ import division
import numpy as np
import sympy
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Input, Model
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LambdaCallback

from .keras_classes import EqlLayer, DivLayer, Connected


def getNonlinearInfo(numHiddenLayers, numBinary, unaryPerBinary):
    """
    Generates a 2D list to be used as a nonlinearInfo argument in building an
    EQL/EQL-div model

    # Arguments
        numHiddenLayers: integer, number of hidden layers (i.e. layers
            including nonlinear keras layer components)
        numBinary: list of integers, available numbers to be used as number of
            binary functions in a nonlinear layer component
        unaryPerBinary: integer, number of unary function per binary function
            in a nonlinear layer component

    # Returns
        A 2D list of integers with dimension numHiddenLayers x 2. Rows
        represent layers, first column is number of unary functions, second
        column is number of binary functions
    """

    nonlinearInfo = [0 for i in range(numHiddenLayers)]
    for i in range(numHiddenLayers):
        v = np.random.choice(numBinary)  # binary nodes
        u = unaryPerBinary * v  # unary nodes
        nonlinearInfo[i] = [u, v]
    return nonlinearInfo


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


def make_symbolic(n, m):
    """
    Generates a sympy matrix full of numbered variables. Not original: taken
    from Andrew Walker's post at https://stackoverflow.com/questions/23208838/
    sympy-substitute-sybolic-entries-in-a-matrix

    # Arguments
        n, m: number of rows, columns in matrix, respectively
    """

    rows = []
    for i in range(n):
        col = []
        for j in range(m):
            col.append(sympy.Symbol('x%d' % (j+1)))
        rows.append(col)
    return sympy.Matrix(rows)


def plotTogether(inputSize, outputSize, models, function, settings=dict(),
                 xmin=-2, xmax=2, step=0.01, colors=None,
                 width=10, height=10, save=False, legNames=None, name='EQL',
                 title='Title', xlabel='X-Axis', ylabel='Y-Axis',
                 titlefont=dict(), labelfont=dict(), tickfont=dict()):
    """
    Behaves like the plotSlice EQL/EQL-div class function, but can plot
    multiple models' learned functions on the same axes

    # Arguments
        inputSize: number of input variables to each model
        outputSize: number of output variables from each model (number of
            subplots)
        models: list of trained EQL/EQL-div models with same input/output
            sizes
        function: python function taking input of a list with length inputSize
            and outputs a list with length outputSize, represents the goal
            function
        settings: plt.rc_context dictionary to be used for visual settings
        xmin, xmax: minimum/maximum x-coordinates to be graphed
        step: x-axis spacing between sampled functional values
        colors: list of tuples containing normalized RGB values to be used as
            the colors of the plotted learned model functions
        width, height: dimensions of matplotlib plot
        save: boolean, indicates whether or not the plot should be saved to a
            .png file
        legNames: list of strings with same length as models, provides names of
            each of the models to be used for matplotlib legend
        name: name of saved .png file, if save == True
        title: string, title of matplotlib plot
        xlabel, ylabel: list of strings with length outputSize, matplotlib axes
            labels

    """

    # x values
    X = np.asarray(
            [[(i * step) + xmin for i in range(int((xmax - xmin)/step))]
                for j in range(inputSize)])
    # goal function values
    F_Y = np.apply_along_axis(function, 0, X)
    # model predictions
    models_Y = [model.model.predict(np.transpose(X)) for model in models]
    models_Y = np.transpose(models_Y, [0, 2, 1])

    if colors is None:
        colors = ['red', 'blue', 'purple', 'black', 'pink', 'brown']
    if legNames is None:
        legNames = tuple('Model ' + str(i+1) for i in range(len(models)))

    settings['figure.figsize'] = (width, height)
    with plt.rc_context(settings):
        fig, axs = plt.subplots(outputSize, figsize=(width, height))

        if outputSize == 1:
            axs = [axs]

        fig.suptitle(title)

        for i in range(outputSize):
            axs[i].plot(X[0], F_Y[i], color=colors[0], linestyle='-',
                        label='Goal Function')
            axs[i].set_xlabel(xlabel)
            axs[i].set_ylabel(ylabel)
            for j, _ in enumerate(models_Y):
                axs[i].plot(X[0], models_Y[j][i], color=colors[j+1],
                            linestyle=':', label=legNames[j])

        plt.legend()

    if save:
        plt.savefig(name + '.png', bbox_inches='tight', dpi=300)


class EQL:
    """
    EQL function learning network

    # Arguments
        inputSize: number of input variables to model. Integer.
        outputSize: number of variables outputted by model. Integer.
        numLayers: number of layers in model. A layer is either a fully-
            connected linear map and a nonlinear map (hidden layer), or just a
            fully-connected linear map (final layer). The Keras Input layer
            doesn't count.
        layers: list of Keras layers containing all of the layers in the
            EQL model (including the Keras Input layer).
        hypothesisSet: 2 x ? list, first row contains tensorflow R -> R
            function to be applied element-wise in nonlinear layer components,
            second row contains the corresponding sympy functions for use in
            printing out learned equations. In practice, usually contains
            identity, sine, cosine, and sigmoid.
        nonlinearInfo: list with rows equal to number of hidden layers and 2
            columns. First column is number of unary functions in each hidden
            layer, second column is number of binary functions in each hidden
            layer
        learningRate: optimizer learning rate.
        name: TensorFlow scope name (for TensorBoard)

    # References
        - [Extrapolation and learning equations](
            https://arxiv.org/abs/1610.02995)
    """

    def __init__(self, inputSize, outputSize, numLayers=4, hypothesisSet=None,
                 nonlinearInfo=None, name='EQL'):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.numLayers = numLayers
        self.layers = []
        self.hypothesisSet = hypothesisSet or [
                [tf.identity, tf.sin, tf.cos, tf.sigmoid],
                [sympy.Id, sympy.sin, sympy.cos, sympy.Function("sigm")]]
        self.nonlinearInfo = nonlinearInfo or getNonlinearInfo(
                self.numLayers-1, [4], 4)
        self.name = name
        self.unaryFunctions = [
                [j % len(self.hypothesisSet[0])
                    for j in range(self.nonlinearInfo[i][0])]
                for i in range(numLayers-1)]

    def build(self, learningRate=0.001, method=Adam, loss='mse',
              metrics=[rmse]):
        """
        __init__ defines all of the variables this model depends upon, this
        function actual creates the tensorflow Model variable

        # Arguments
            learningRate: model learning rate
            method: model learning method
            loss: loss function used to penalize model based on accuracy
            metrics: metric used for gauging model performance when evaluated
                on data points post-training
        """

        self.layers.append(Input((self.inputSize,)))
        inp = int(self.layers[-1].shape[1])

        for i in range(self.numLayers - 1):
            u, v = self.nonlinearInfo[i]
            out = u + 2 * v
            stddev = np.sqrt(1 / (inp * out))
            randNorm = RandomNormal(0, stddev=stddev)
            self.layers.append(EqlLayer(self.nonlinearInfo[i],
                                        self.hypothesisSet[0],
                                        self.unaryFunctions[i],
                                        kernel_initializer=randNorm,
                                        )(self.layers[-1]))
            inp = u + v

        stddev = np.sqrt(1 / (self.outputSize * inp))
        randNorm = RandomNormal(0, stddev=stddev)
        self.layers.append(Connected(self.outputSize,
                                     kernel_initializer=randNorm
                                     )(self.layers[-1]))

        # Optimizer
        optimizer = method(lr=learningRate)

        # Model
        self.model = Model(inputs=self.layers[0],
                           outputs=self.layers[-1])

        # Compilation
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, predictors, labels, numEpoch, reg=10**-3, batchSize=20,
            threshold=0.1, verbose=0):
        """
        Trains EQL model on a dataset following the training schedule defined
        in the reference.

        # Arguments
            predictors: ? x inputSize array containing data to be trained on
            labels: ? x outputSize array containing corresponding correct
                output for predictors, compared with model output
            numEpoch: integer, number of epochs
            reg: regularization (in the range [10^-4, 10^-2.5] is usually
                ideal)
            batchSize: number of datapoints trained on per gradient descent
                update
            threshold: float, weight/bias elements below this value are kept
                at zero during the final training phase
        """

        # PHASE 1: NO REGULARIZATION (T/4)
        for i in range(1, self.numLayers+1):
            K.set_value(self.model.layers[i].regularization, 0.)
            K.set_value(self.model.layers[i].Wtrimmer,
                        np.ones(self.model.layers[i].W.shape))
            K.set_value(self.model.layers[i].btrimmer,
                        np.ones(self.model.layers[i].b.shape))
        self.model.fit(predictors, labels, epochs=int(numEpoch*(1/4)),
                       batch_size=batchSize, verbose=verbose)

        # PHASE 2: REGULARIZATION (7T/10)
        for i in range(1, self.numLayers+1):
            K.set_value(self.model.layers[i].regularization, reg)
        self.model.fit(predictors, labels, epochs=int(numEpoch*(7/10)),
                       batch_size=batchSize, verbose=verbose)

        # PHASE 3: NO REGULARIZATION, L0 NORM PRESERVATION (T/20)
        for i in range(1, self.numLayers+1):
            W, b = self.model.layers[i].get_weights()[:2]
            K.set_value(self.model.layers[i].regularization, 0.)
            K.set_value(self.model.layers[i].Wtrimmer,
                        (np.abs(W) > threshold).astype(float))
            K.set_value(self.model.layers[i].btrimmer,
                        (np.abs(b) > threshold).astype(float))
        self.model.fit(predictors, labels, epochs=int(numEpoch*(1/20)),
                       batch_size=batchSize, verbose=verbose)

    def evaluate(self, predictors, labels, batchSize=10, verbose=0):
        """Evaluates trained model on data"""
        return self.model.evaluate(predictors, labels, batch_size=batchSize,
                                   verbose=verbose)[1]

    def getEquation(self):
        """Prints learned equation of a trained model."""

        # prepares lists for weights and biases
        weights = []
        bias = []

        # pulls/separates weights and biases from a model
        for i in range(1, self.numLayers+1):
            weights.append(self.model.layers[i].get_weights()[0])
            bias.append(self.model.layers[i].get_weights()[1])

        print(np.array(weights).shape, np.array(self.nonlinearInfo).shape)

        # creates generic input vector
        X = make_symbolic(1, self.inputSize)

        for i, _ in enumerate(weights):
            # computes the result of the next linear layer
            W = sympy.Matrix(weights[i])
            b = sympy.Transpose(sympy.Matrix(bias[i]))
            Y = sympy.zeros(1, b.cols)
            X = X*W + b

            # computes the result of the next nonlinear layer, if applicable
            if i != (len(weights) - 1):
                u, v = self.nonlinearInfo[i]

                # computes the result of the unary component of the nonlinear
                # layer
                # iterating over unary input
                for j in range(u):
                    Y[0, j] = self.hypothesisSet[1][self.unaryFunctions[i][j]](
                            X[0, j])

                # computes the result of the binary component of the nonlinear
                # layer
                # iterating over binary input
                for j in range(v):
                    Y[0, j+u] = X[0, j * 2 + u] * X[0, j * 2 + u + 1]

                # removes final v rows which are now outdated
                for j in range(u + v, Y.cols):
                    Y.col_del(u + v)

                X = Y

        return X

    def plotSlice(self, function, xmin=-2, xmax=2, step=0.01, width=10,
                  height=10, settings=dict(), save=False):
        """
        Plots the x_1 = ... = x_n slice of the learned function in each output
        variable.
        """

        # x values
        X = np.asarray(
                [[(i * step) + xmin for i in range(int((xmax - xmin)/step))]
                    for j in range(self.inputSize)])
        # goal function values
        F_Y = np.apply_along_axis(function, 0, X)
        # model predictions
        model_Y = self.model.predict(np.transpose(X))
        model_Y = np.transpose(model_Y)

        settings['figure.figsize'] = (width, height)
        with plt.rc_context(settings):
            fig, axs = plt.subplots(self.outputSize, figsize=(width, height))

            if self.outputSize == 1:
                axs = [axs]

            for i in range(self.outputSize):
                axs[i].plot(X[0], F_Y[i], linestyle='-', label='Goal Function')
                axs[i].plot(X[0], model_Y[i], linestyle=':',
                            label='Learned Function')

            plt.legend()

        if save:
            plt.savefig(self.name + '.png', bbox_inches='tight', dpi=300)

    def percentError(self, predictors, labels):
        """
        Returns the average percent error in each variable of a trained model
        with respect to a testing data set
        """

        labels = np.reshape(labels, (-1, self.outputSize))
        predictions = self.model.predict(predictors)
        error = np.divide(np.abs(predictions - labels), np.abs(labels))
        error = np.sum(error, 0)
        error *= (100 / labels.shape[0])
        return error

    def sparsity(self, minMag=0.01):
        """Returns the sparsity of a trained model (number of active nodes)"""

        vec = np.ones((1, self.inputSize))
        weights = self.model.get_weights()
        sparsity = 0
        for i in range(self.numLayers - 1):
            u, v = self.nonlinearInfo[i]
            w, b = weights[i*2], weights[i*2+1]
            vec = np.dot(vec, w) + b
            vec = np.concatenate(
                    (vec[:, :u], vec[:, u::2] * vec[:, u+1::2]), axis=1)
            sparsity += np.sum(
                    (np.abs(vec) > np.full_like(vec, minMag)).astype(int))

        w, b = weights[len(weights)-2], weights[len(weights)-1]
        vec = np.dot(vec, w) + b
        sparsity += np.sum(
                (np.abs(vec) > np.full_like(vec, minMag)).astype(int))
        return sparsity

    def odecompat(self, t, x):
        """Wrapper for Keras' predict function, solve_ivp compatible"""
        prediction = self.model.predict(np.reshape(x, (1, len(x))))
        return prediction

    def printJacobian(self, x):
        """
        Prints the Jacobian of the learned function, evaluated at a point

        # Arguments
            x: point at which Jacobian is evaluated, array-like with
                self.inputSize elements
        """

        x = np.reshape(x, (1, 1, self.inputSize)).tolist()
        gradients = [tf.gradients(self.model.output[:, i], self.model.input)[0]
                     for i in range(4)]
        funcs = [K.function((self.model.input, ), [g]) for g in gradients]
        jacobian = np.concatenate([func(x)[0] for func in funcs], axis=0)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(jacobian)


class EQLDIV:
    """
    EQL-div function learning network

    # Arguments
        inputSize: number of input variables to model. Integer.
        outputSize: number of variables outputted by model. Integer.
        numLayers: number of layers in model. A layer is either a fully-
            connected linear map and a nonlinear map (hidden layer), or just a
            fully-connected linear map (final layer). The Keras Input layer
            doesn't count.
        layers: list of Keras layers containing all of the layer tensor outputs
            (not the layer object itself) in the EQL model (including the Keras
            Input layer).
        hypothesisSet: list of 2-tuples, first element of tuples is tensorflow
            R -> R function to be applied element-wise in nonlinear layer
            components, second element is the corresponding sympy function for
            use in printing out learned equations. In practice, usually
            contains identity, sine, cosine, and sigmoid.
        nonlinearInfo: list with rows equal to number of hidden layers and 2
            columns. First column is number of unary functions in each hidden
            layer, second column is number of binary functions in each hidden
            layer
        energyInfo: if energy regularization is to be used, is a list with
            length three containing (1) a python function which uses tensorflow
            methods to compute the Hamiltonian associated with each member of
            a batch of predicted state, (2) a float value giving the actual
            Hamilton of the data (NOTE: ENERGYINFO SHOULD ONLY BE USED WITH
            TIMESERIES DATA OR OTHER CONSTANT ENERGY DATA), and (3) a
            coefficient for scaling the energy error in the loss function
            (10^-5 recommended, only activated during the second training
            phase)
        learningRate: optimizer learning rate.
        divThreshold: float, value which denominator must be greater than in
            order for division to occur (division returns 0 otherwise)
        name: TensorFlow scope name (for TensorBoard)
        changeOfVariables: a list with length two containing (1) a python
            function which can transform an arbitrarily large numpy array
            training data set to a different set of variables and (2) the
            number of components per data point in the transformed data that
            the function in the first component outputs

    # References
        - [Learning Equations for Extrapolation and Control](
            https://arxiv.org/abs/1806.07259)
    """

    def __init__(self, inputSize, outputSize, numLayers=3, hypothesisSet=None,
                 nonlinearInfo=None, energyInfo=None, changeOfVariables=None,
                 name='EQLDIV'):
        self.inputSize = changeOfVariables[1] or inputSize
        self.outputSize = outputSize
        self.numLayers = numLayers
        self.layers = []
        self.hypothesisSet = hypothesisSet or [
                [tf.identity, tf.sin, tf.cos, tf.sigmoid],
                [sympy.Id, sympy.sin, sympy.cos, sympy.Function("sigm")]]
        self.nonlinearInfo = nonlinearInfo or getNonlinearInfo(
                self.numLayers-1, [4], 4)
        self.energyInfo = energyInfo
        self.changeOfVariables = changeOfVariables
        self.name = name
        self.unaryFunctions = [
                [j % len(self.hypothesisSet[0])
                 for j in range(self.nonlinearInfo[i][0])]
                for i in range(self.numLayers-1)]

    def build(self, divThreshold=0.001, learningRate=0.001, method=Adam,
              loss='mse', metrics=[rmse]):
        """
        __init__ defines all of the variables this model depends upon, this
        function actual creates the tensorflow Model variable

        # Arguments
            learningRate: model learning rate
            method: model learning method
            loss: loss function used to penalize model based on accuracy
            metrics: metric used for gauging model performance when evaluated
                on data points post-training
        """

        self.layers.append(Input((self.inputSize,)))
        inp = int(self.layers[-1].shape[1])

        for i in range(self.numLayers):
            if (i == self.numLayers - 1):
                stddev = np.sqrt(1 / (self.outputSize * 2 * inp))
                randNorm = RandomNormal(0, stddev=stddev)
                self.layers.append(DivLayer(self.outputSize,
                                            threshold=divThreshold,
                                            kernel_initializer=randNorm,
                                            )(self.layers[-1]))
                break
            u, v = self.nonlinearInfo[i] or None
            out = u + 2 * v or None
            stddev = np.sqrt(1 / (inp * out))
            randNorm = RandomNormal(0, stddev=stddev)
            self.layers.append(EqlLayer(self.nonlinearInfo[i],
                                        self.hypothesisSet[0],
                                        self.unaryFunctions[i],
                                        kernel_initializer=randNorm,
                                        )(self.layers[-1]))
            inp = u + v

        # Optimizer
        optimizer = method(lr=learningRate)

        # Model
        self.model = Model(inputs=self.layers[0],
                           outputs=self.layers[-1])

        # Compilation
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, predictors, labels, numEpoch, regStrength=10**-3,
            batchSize=20, normThreshold=0.001, verbose=0):
        """
        Trains EQL model on a dataset following the training schedule defined
        in the reference.

        # Arguments
            predictors: ? x inputSize array containing data to be trained on
            labels: ? x outputSize array containing corresponding correct
                output for predictors, compared with model output
            numEpoch: integer, number of epochs
            reg: regularization (in the range [10^-4, 10^-2.5] is usually
                ideal)
            batchSize: number of datapoints trained on per gradient descent
                update
            normThreshold: float, weight/bias elements below this value are
                kept at zero during the final training phase
            verbose: 0, 1, or 2, determines whether Keras is silent, prints a
                progress bar, or prints a line every epoch.
        """
        if self.changeOfVariables:
            predictors = self.changeOfVariables[0](predictors)

        # Division threshold callbacks
        def phase1DivisionThresholdSchedule(epoch, logs):
            K.set_value(self.model.layers[-1].threshold,
                        1 / np.sqrt(epoch + 1))

        def phase2DivisionThresholdSchedule(epoch, logs):
            K.set_value(self.model.layers[-1].threshold,
                        1 / np.sqrt(int(numEpoch * (1 / 4)) + epoch + 1))

        def phase3DivisionThresholdSchedule(epoch, logs):
            K.set_value(
                    self.model.layers[-1].threshold,
                    1 / np.sqrt(int(numEpoch * (7/10))
                                + int(numEpoch * (1/4))
                                + epoch + 1))

        # PHASE 1: NO REGULARIZATION (T/4)
        for i in range(1, self.numLayers+1):
            K.set_value(self.model.layers[i].regularization, 0.)
            K.set_value(self.model.layers[i].Wtrimmer,
                        np.ones(self.model.layers[i].W.shape))
            K.set_value(self.model.layers[i].btrimmer,
                        np.ones(self.model.layers[i].b.shape))
        dynamicDivisionThreshold = LambdaCallback(
                on_epoch_begin=phase1DivisionThresholdSchedule)
        self.model.fit(predictors, labels, epochs=int(numEpoch*(1/4)),
                       batch_size=batchSize, verbose=verbose,
                       callbacks=[dynamicDivisionThreshold])

        # PHASE 2: REGULARIZATION (7T/10)
        if self.energyInfo is not None:
            K.set_value(self.model.layers[-1].loss.coef, self.coef)
        for i in range(1, self.numLayers+1):
            K.set_value(self.model.layers[i].regularization, regStrength)
        dynamicDivisionThreshold = LambdaCallback(
                on_epoch_begin=phase2DivisionThresholdSchedule)
        self.model.fit(predictors, labels, epochs=int(numEpoch*(7/10)),
                       batch_size=batchSize, verbose=verbose,
                       callbacks=[dynamicDivisionThreshold])

        # PHASE 3: NO REGULARIZATION, L0 NORM PRESERVATION (T/20)
        for i in range(1, self.numLayers+1):
            W, b = self.model.layers[i].get_weights()[:2]
            K.set_value(self.model.layers[i].regularization, 0.)
            K.set_value(self.model.layers[i].Wtrimmer,
                        (np.abs(W) > normThreshold).astype(float))
            K.set_value(self.model.layers[i].btrimmer,
                        (np.abs(b) > normThreshold).astype(float))
        dynamicDivisionThreshold = LambdaCallback(
                on_epoch_begin=phase3DivisionThresholdSchedule)
        self.model.fit(predictors, labels, epochs=int(numEpoch*(1/20)),
                       batch_size=batchSize, verbose=verbose,
                       callbacks=[dynamicDivisionThreshold])

        # Final maintenance
        K.set_value(self.model.layers[-1].threshold, 0.001)

    def evaluate(self, predictors, labels, batchSize=10, verbose=0):
        """Evaluates trained model on data"""
        return self.model.evaluate(predictors, labels, batch_size=batchSize,
                                   verbose=verbose)[1]

    def getEquation(self):
        """Prints learned equation of a trained model."""

        # prepares lists for weights and biases
        weights = []
        bias = []

        # pulls/separates weights and biases from a model
        for i in range(1, self.numLayers+1):
            weights.append(self.model.layers[i].get_weights()[0])
            bias.append(self.model.layers[i].get_weights()[1])

        # creates generic input vector
        X = make_symbolic(1, self.inputSize)

        for i, _ in enumerate(weights):
            # computes the result of the next linear layer
            W = sympy.Matrix(weights[i])
            b = sympy.Transpose(sympy.Matrix(bias[i]))
            Y = sympy.zeros(1, b.cols)
            X = X*W + b

            # computes the result of the next nonlinear layer, if applicable
            if i != (len(weights) - 1):
                u, v = self.nonlinearInfo[i]

                # computes the result of the unary component of the nonlinear
                # layer
                # iterating over unary input
                for j in range(u):
                    Y[0, j] = self.hypothesisSet[1][self.unaryFunctions[i][j]](
                            X[0, j])

                # computes the result of the binary component of the nonlinear
                # layer
                # iterating over binary input
                for j in range(v):
                    Y[0, j+u] = X[0, j * 2 + u] * X[0, j * 2 + u + 1]

                # removes final v rows which are now outdated
                for j in range(u + v, Y.cols):
                    Y.col_del(u + v)

                X = Y

            if i == (len(weights) - 1):
                # computes the result of the binary component of the nonlinear
                # layer
                # iterating over binary input
                for j in range(int(X.cols/2)):
                    if sympy.Abs(X[0, j*2+1]) == 0:
                        Y[0, j] = 0
                    else:
                        Y[0, j] = X[0, j*2] / X[0, j*2+1]
                for j in range(int(X.cols/2)):
                    Y.col_del(int(X.cols/2))

                X = Y

        return X

    def plotSlice(self, function, xmin=-2, xmax=2, step=0.01, width=10,
                  height=10, settings=dict(), save=False):
        """
        Plots the x_1 = ... = x_n slice of the learned function in each output
        variable.
        """

        # x values
        X = np.asarray(
                [[(i * step) + xmin for i in range(int((xmax - xmin)/step))]
                    for j in range(self.inputSize)])
        # goal function values
        F_Y = np.apply_along_axis(function, 0, X)
        # model predictions
        model_Y = self.model.predict(np.transpose(X))
        model_Y = np.transpose(model_Y)

        settings['figure.figsize'] = (width, height)
        with plt.rc_context(settings):
            fig, axs = plt.subplots(self.outputSize, figsize=(width, height))

            if self.outputSize == 1:
                axs = [axs]

            for i in range(self.outputSize):
                axs[i].plot(X[0], F_Y[i], linestyle='-', label='Goal Function')
                axs[i].plot(X[0], model_Y[i], linestyle=':',
                            label='Learned Function')

            plt.legend()

        if save:
            plt.savefig(self.name + '.png', bbox_inches='tight', dpi=300)

    def percentError(self, predictors, labels):
        """
        Returns the average percent error in each variable of a trained model
        with respect to a testing data set
        """

        labels = np.reshape(labels, (-1, self.outputSize))
        predictions = self.model.predict(predictors)
        error = np.divide(np.abs(predictions - labels), np.abs(labels))
        error = np.sum(error, 0)
        error *= (100 / labels.shape[0])
        return error

    def sparsity(self, minMag=0.01):
        """Returns the sparsity of a trained model (number of active nodes)"""

        vec = np.ones((1, self.inputSize))
        weights = self.model.get_weights()
        sparsity = 0
        for i in range(self.numLayers - 1):
            u, v = self.nonlinearInfo[i]
            w, b = weights[i*2], weights[i*2+1]
            vec = np.dot(vec, w) + b
            vec = np.concatenate(
                    (vec[:, :u], vec[:, u::2] * vec[:, u+1::2]), axis=1)
            sparsity += np.sum(
                    (np.abs(vec) > np.full_like(vec, minMag)).astype(int))

        w, b = weights[len(weights)-2], weights[len(weights)-1]
        vec = np.dot(vec, w) + b
        vec = vec[0, ::2] * vec[0, 1::2]
        sparsity += np.sum(
                (np.abs(vec) > np.full_like(vec, minMag)).astype(int))
        return sparsity

    def setPipeline(self, pipeline):
        """
        Saves the scikit_learn pipeline used to modify training data so that
        the same pipeline can be applied to testing data
        """

        self.pipeline = pipeline

    def applyPipeline(self, x):
        """Applies a saved scikit_learn pipeline to a dataset"""

        if self.pipeline is not None:
            for op in self.pipeline:
                x = op.transform(x)
        return x

    def odecompat(self, t, x):
        """Wrapper for Keras' function, solve_ivp compatible"""

        # if statement is bad hack for experimentations with feature
        # engineering for double pendulum
#        if self.inputSize == 7:
#            if len(np.array(x).shape) == 2:
#                y = [x[0, 0] - x[0, 2], x[0, 1]**2, x[0, 3]**2]
#                x = np.append(x, y)
#            else:
#                y = [x[0] - x[2], x[1]**2, x[3]**2]
#                x = np.append(x, y)

        prediction = self.model.predict(np.reshape(x, (1, -1)))
        return prediction

    def printJacobian(self, x):
        """
        Prints the Jacobian of the learned function, evaluated at a point

        # Arguments
            x: point at which Jacobian is evaluated, array-like with
                self.inputSize elements
        """
        x = np.reshape(x, (1, 1, self.inputSize)).tolist()
        gradients = [tf.gradients(self.model.output[:, i], self.model.input)[0]
                     for i in range(4)]
        funcs = [K.function((self.model.input, ), [g]) for g in gradients]
        jacobian = np.concatenate([func(x)[0] for func in funcs], axis=0)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(jacobian)
