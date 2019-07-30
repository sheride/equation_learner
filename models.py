from __future__ import division
import numpy as np
import sympy
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.backend as K
from keras.initializers import RandomNormal as RandNorm
from keras.engine.input_layer import Input
from keras.layers import Dense
from keras.callbacks import LambdaCallback as LambCall
from . import keras_classes as my
from .keras_classes import Nonlinear as Nonlin
from .keras_classes import DynamReg
from .keras_classes import ConstantL0
from .keras_classes import DenominatorPenalty as DenPen


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
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# NOT MINE: function for making n x m matrix of symbols
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
            for j in range(len(models_Y)):
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

    def __init__(self, inputSize, outputSize, numLayers, hypothesisSet,
                 nonlinearInfo, learningRate=0.01, name='EQL'):

        self.inputSize = inputSize
        self.outputSize = outputSize
        self.numLayers = numLayers
        self.layers = [None for i in range(numLayers * 2)]
        self.hypothesisSet = hypothesisSet
        self.nonlinearInfo = nonlinearInfo
        self.learningRate = learningRate
        self.name = name

        with tf.name_scope(self.name) as scope:
            # Number of Keras layers: length of self.layers
            numKerLay = len(self.layers)
            self.unaryFunctions = [[j % len(hypothesisSet[0])
                                    for j in range(self.nonlinearInfo[i][0])]
                                   for i in range(numLayers-1)]
            self.layers[0] = Input((inputSize,), name='input')

            # Create all hidden layers (linear and nonlinear components)
            for i in range(1, (self.numLayers-1) * 2, 2):
                # Size of input to layer
                linIn = int(self.layers[i-1].shape[1])
                # Unary, binary width of layer
                u, v = self.nonlinearInfo[int((i-1)/2)]
                # Dense/linear component of layer 'i'
                stddev = np.sqrt(1 / (linIn * (u + 2 * v)))
                randNorm = RandNorm(0, stddev=stddev, seed=2000)
                # Prepping weight, bias tensors for ConstL0
                wZeros = tf.cast(K.zeros((linIn, u + 2 * v)), tf.bool)
                bZeros = tf.cast(K.zeros((u + 2 * v, )), tf.bool)
                self.layers[i] = Dense(
                    u + 2 * v,
                    kernel_initializer=randNorm,
                    kernel_regularizer=DynamReg(0),
                    bias_regularizer=DynamReg(0),
                    kernel_constraint=ConstantL0(wZeros),
                    bias_constraint=ConstantL0(bZeros)
                    )(self.layers[i-1])
                # Non-linear component of layer 'i'
                self.layers[i+1] = Nonlin(self.nonlinearInfo[int((i-1)/2)],
                                          self.hypothesisSet[0],
                                          self.unaryFunctions[int((i-1)/2)],
                                          )(self.layers[i])
            # Final layer
            linIn = int(self.layers[numKerLay-2].shape[1])
            stddev = np.sqrt(1 / (self.outputSize * linIn))
            randNorm = RandNorm(0, stddev=stddev, seed=2000)
            # Prepping weight, bias tensors for ConstL0
            wZeros = tf.cast(K.zeros((linIn, self.outputSize)), tf.bool)
            bZeros = tf.cast(K.zeros((self.outputSize, )), tf.bool)
            self.layers[numKerLay - 1] = Dense(
                self.outputSize,
                kernel_initializer=randNorm,
                kernel_regularizer=DynamReg(0),
                bias_regularizer=DynamReg(0),
                kernel_constraint=ConstantL0(wZeros),
                bias_constraint=ConstantL0(bZeros),
                )(self.layers[numKerLay-2])

            # Optimizer
            optimizer = keras.optimizers.Adam(lr=self.learningRate)

            # Model
            self.model = keras.Model(inputs=self.layers[0],
                                     outputs=self.layers[self.numLayers*2-1])

            # Compilation
            self.model.compile(optimizer=optimizer, loss='mse', metrics=[rmse])

    def fit(self, predictors, labels, numEpoch, reg, batchSize=20,
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
        self.model.fit(predictors, labels, epochs=int(numEpoch*(1/4)),
                       batch_size=batchSize, verbose=verbose)

        # PHASE 2: REGULARIZATION (7T/10)
        # updating weight, bias regularization
        for i in range(1, len(self.model.layers), 2):
            K.set_value(self.model.layers[i].kernel_regularizer.l1,
                        reg)
            K.set_value(self.model.layers[i].bias_regularizer.l1,
                        reg)
        self.model.fit(predictors, labels, epochs=int(numEpoch*(7/10)),
                       batch_size=batchSize, verbose=verbose)

        # PHASE 3: NO REGULARIZATION, L0 NORM PRESERVATION (T/20)
        # updating weight, bias regularization
        for i in range(1, len(self.model.layers), 2):
            K.set_value(self.model.layers[i].kernel_regularizer.l1, 0)
            K.set_value(self.model.layers[i].bias_regularizer.l1, 0)
            weight, bias = self.model.layers[i].get_weights()
            K.set_value(self.model.layers[i].kernel_constraint.toZero,
                        np.less(np.abs(weight),
                                np.full(weight.shape, threshold)))
            K.set_value(self.model.layers[i].bias_constraint.toZero,
                        np.less(np.abs(bias), np.full(bias.shape, threshold)))
        self.model.fit(predictors, labels, epochs=int(numEpoch*(1/20)),
                       batch_size=batchSize, verbose=verbose)

    def evaluate(self, predictors, labels, batchSize=10, verbose=0):
        """
        Evaluates trained model on data
        """
        return self.model.evaluate(predictors, labels, batch_size=batchSize,
                                   verbose=verbose)[1]

    def getEquation(self):
        """
        Prints learned equation of a trained model.
        """

        # prepares lists for weights and biases
        weights = [0 for i in range(int(len(self.model.get_weights())/2))]
        bias = [0 for i in range(int(len(self.model.get_weights())/2))]

        # pulls/separates weights and biases from a model
        for i in range(0, len(self.model.get_weights()), 2):
            weights[int(i/2)] = np.asarray(self.model.get_weights()[i])
            bias[int(i/2)] = np.asarray(self.model.get_weights()[i+1])

        # creates generic input vector
        X = make_symbolic(1, self.inputSize)

        for i in range(0, len(weights)):
            # computes the result of the next linear layer
            W = sympy.Matrix(weights[i])
            b = sympy.Transpose(sympy.Matrix(bias[i]))
            X = X*W + b

            # computes the result of the next nonlinear layer, if applicable
            if i != (len(weights) - 1):
                Y = sympy.zeros(1, b.cols)
                # computes the result of the unary component of the nonlinear
                # layer
                # iterating over unary input
                for j in range(self.nonlinearInfo[i][0]):
                    Y[0, j] = self.hypothesisSet[1][self.unaryFunctions[i][j]](
                            X[0, j])

                # computes the result of the binary component of the nonlinear
                # layer
                # iterating over binary input
                for j in range(self.nonlinearInfo[i][1]):
                    Y[0, j+self.nonlinearInfo[i][0]] = X[
                        0, j * 2 + self.nonlinearInfo[i][0]] * X[
                            0, j * 2 + self.nonlinearInfo[i][0] + 1]

                # removes final v rows which are now outdated
                for j in range(
                        self.nonlinearInfo[i][0] + self.nonlinearInfo[i][1],
                        Y.cols):
                    Y.col_del(self.nonlinearInfo[i][0]
                              + self.nonlinearInfo[i][1])

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

    # percent error function
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
        """
        Returns the sparsity of a trained model (number of active nodes)
        """

        # list of lists where ith element is list containing activity (in the
        # form of a binary value) of outputs of ith layer
        layerSparsity = [[]
                         for i in range(int(len(self.model.get_weights())/2))]

        # iterating over layers
        for i in range(0, len(self.model.get_weights()), 2):

            # linear component
            # weights
            # iterating over columns (due to how the matrix mult. works)
            for j in range(len(self.model.get_weights()[i][0])):
                layerSparsity[int(i/2)].append(0)
                # iterating over rows
                for k in range(len(self.model.get_weights()[i])):
                    if ((i == 0 or layerSparsity[int(i/2) - 1][k] != 0) and
                            abs(self.model.get_weights()[i][k][j]) > minMag):
                        # if weight surpasses minimum magnitude and input was
                        # active, output is active
                        layerSparsity[int(i/2)][j] = 1

            # biases
            for j in range(len(self.model.get_weights()[i+1])):
                if abs(self.model.get_weights()[i+1][j]) > minMag:
                    # if bias surpasses minimum magnitude, output is active
                    layerSparsity[int(i/2)][j] = 1

        # handling binary units in nonlinear layers where necessary
        # if not last layer (which has no nonlinear component)
        if i != len(self.model.get_weights()) - 2:
            u, v = self.nonlinearInfo[int(i/2)]
            # active if both units active
            for j in range(v):
                layerSparsity[int(i/2)][u + j] = layerSparsity[int(i/2)][
                    u + 2 * j] * layerSparsity[int(i/2)][u + 2 * j + 1]
            del layerSparsity[int(i/2)][u + v:]  # get rid of old values now

        # return sum of flattened layerSparsity list (number of active oututs)
        return sum([item for sublist in layerSparsity for item in sublist])

    def odecompat(self, t, x):
        """
        Wrapper for Keras' predict function
        """

        prediction = self.model.predict(np.reshape(x, (1, len(x))))
        return prediction


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
        layers: list of Keras layers containing all of the layers in the
            EQL model (including the Keras Input layer).
        hypothesisSet: list of 2-tuples, first element of tuples is tensorflow
            R -> R function to be applied element-wise in nonlinear layer
            components, second element is the corresponding sympy function for
            use in printing out learned equations. In practice, usually
            contains identity, sine, cosine, and sigmoid.
        nonlinearInfo: list with rows equal to number of hidden layers and 2
            columns. First column is number of unary functions in each hidden
            layer, second column is number of binary functions in each hidden
            layer
        learningRate: optimizer learning rate.
        divThreshold: float, value which denominator must be greater than in
            order for division to occur (division returns 0 otherwise)
        name: TensorFlow scope name (for TensorBoard)

    # References
        - [Learning Equations for Extrapolation and Control](
           https://arxiv.org/abs/1806.07259)

    """

    def __init__(self, inputSize, outputSize, numLayers, hypothesisSet,
                 nonlinearInfo, learningRate=0.01, divThreshold=0.001,
                 name='EQL'):

        self.inputSize = inputSize
        self.outputSize = outputSize
        self.numLayers = numLayers
        self.layers = [None for i in range(numLayers * 2 + 1)]
        self.hypothesisSet = hypothesisSet
        self.nonlinearInfo = nonlinearInfo
        self.learningRate = learningRate
        self.divThreshold = divThreshold
        self.name = name
        self.pipeline = None

        with tf.name_scope(self.name) as scope:
            # Number of Keras layers: length of self.layers
            numKerLay = len(self.layers)
            self.unaryFunctions = [[j % len(hypothesisSet[0])
                                    for j in range(self.nonlinearInfo[i][0])]
                                   for i in range(self.numLayers-1)]
            self.layers[0] = Input((self.inputSize,), name='input')

            # Create all hidden layers (linear and nonlinear components)
            for i in range(1, (self.numLayers-1) * 2, 2):
                # Size of input to layer
                linIn = int(self.layers[i-1].shape[1])
                # Unary, binary width of layer
                u, v = self.nonlinearInfo[int((i-1)/2)]
                # Dense/linear component of layer 'i'
                stddev = np.sqrt(1 / (linIn * (u + 2 * v)))
                randNorm = RandNorm(0, stddev=stddev, seed=2000)
                # Prepping weight, bias tensors for ConstL0
                wZeros = tf.cast(K.zeros((linIn, u + 2 * v)), tf.bool)
                bZeros = tf.cast(K.zeros((u + 2 * v, )), tf.bool)
                self.layers[i] = Dense(u + 2 * v,
                                       kernel_initializer=randNorm,
                                       kernel_regularizer=DynamReg(0),
                                       bias_regularizer=DynamReg(0),
                                       kernel_constraint=ConstantL0(wZeros),
                                       bias_constraint=ConstantL0(bZeros),
                                       )(self.layers[i-1])

                # Non-linear component of layer 'i'
                self.layers[i+1] = Nonlin(self.nonlinearInfo[int((i-1)/2)],
                                          self.hypothesisSet[0],
                                          self.unaryFunctions[int((i-1)/2)],
                                          )(self.layers[i])

            # Final layer
            linIn = int(self.layers[numKerLay - 3].shape[1])
            stddev = np.sqrt(1 / (self.outputSize * linIn))
            randNorm = RandNorm(0, stddev=stddev, seed=2000)
            # Prepping weight, bias tensors for ConstL0
            wZeros = tf.cast(K.zeros((linIn, self.outputSize*2)), tf.bool)
            bZeros = tf.cast(K.zeros((self.outputSize * 2,)), tf.bool)
            self.layers[numKerLay - 2] = Dense(
                outputSize * 2,
                kernel_initializer=randNorm,
                kernel_regularizer=DynamReg(0),
                bias_regularizer=DynamReg(0),
                activity_regularizer=DenPen(self.divThreshold),
                kernel_constraint=ConstantL0(wZeros),
                bias_constraint=ConstantL0(bZeros),
                )(self.layers[numKerLay-3])

            # Division final layer component
            self.layers[numKerLay - 1] = my.Division(
                    self.divThreshold)(self.layers[numKerLay - 2])

            # Optimizer
            optimizer = keras.optimizers.Adam(lr=self.learningRate)

            # Model
            self.model = keras.Model(inputs=self.layers[0],
                                     outputs=self.layers[self.numLayers * 2])

            # Compilation
            self.model.compile(optimizer=optimizer, loss='mse', metrics=[rmse])

    def fit(self, predictors, labels, numEpoch, regStrength, batchSize=20,
            normThreshold=0.001, verbose=0):
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

        n = self.numLayers*2
        # PHASE 1: NO REGULARIZATION (T/4)
        dynamicThreshold = LambCall(
            on_epoch_begin=lambda epoch, logs: K.set_value(
                self.model.layers[n].threshold, 1 / np.sqrt(
                    epoch + 1)))
        dynamicThreshold2 = LambCall(
            on_epoch_begin=lambda epoch, logs: K.set_value(
                self.model.layers[n-1].activity_regularizer.divThreshold,
                1 / np.sqrt(epoch + 1)))
        self.model.fit(predictors, labels, epochs=int(numEpoch*(1/4)),
                       batch_size=batchSize, verbose=verbose,
                       callbacks=[dynamicThreshold, dynamicThreshold2])

        # PHASE 2: REGULARIZATION (7T/10)
        dynamicThreshold = LambCall(
            on_epoch_begin=lambda epoch, logs: K.set_value(
                self.model.layers[n].threshold, 1 / np.sqrt(
                    int(numEpoch * (1 / 4)) + epoch + 1)))
        dynamicThreshold2 = LambCall(
            on_epoch_begin=lambda epoch, logs:
            K.set_value(
                self.model.layers[n-1].activity_regularizer.divThreshold,
                1 / np.sqrt(int(numEpoch * (1 / 4)) + epoch + 1)))
        for i in range(1, len(self.model.layers), 2):
            K.set_value(self.model.layers[i].kernel_regularizer.l1,
                        regStrength)
            K.set_value(self.model.layers[i].bias_regularizer.l1,
                        regStrength)
        self.model.fit(predictors, labels, epochs=int(numEpoch*(7/10)),
                       batch_size=batchSize, verbose=verbose,
                       callbacks=[dynamicThreshold, dynamicThreshold2])

        # PHASE 3: NO REGULARIZATION, L0 NORM PRESERVATION (T/20)
        dynamicThreshold = LambCall(
            on_epoch_begin=lambda epoch, logs: K.set_value(
                self.model.layers[n].threshold, 1 / np.sqrt(
                    int(numEpoch * (1 / 4))
                    + int(numEpoch * (7 / 10))
                    + epoch + 1)))
        dynamicThreshold2 = LambCall(
            on_epoch_begin=lambda epoch, logs: K.set_value(
                self.model.layers[n-1].activity_regularizer.divThreshold,
                1 / np.sqrt(
                    int(numEpoch * (1 / 4))
                    + int(numEpoch * (7 / 10))
                    + epoch + 1)))
        for i in range(1, len(self.model.layers), 2):
            K.set_value(self.model.layers[i].kernel_regularizer.l1, 0)
            K.set_value(self.model.layers[i].bias_regularizer.l1, 0)
            weight, bias = self.model.layers[i].get_weights()
            K.set_value(self.model.layers[i].kernel_constraint.toZero,
                        np.less(np.abs(weight),
                                np.full(weight.shape, normThreshold)))
            K.set_value(self.model.layers[i].bias_constraint.toZero,
                        np.less(np.abs(bias),
                                np.full(bias.shape, normThreshold)))
        self.model.fit(predictors, labels, epochs=int(numEpoch*(1/20)),
                       batch_size=batchSize, verbose=verbose,
                       callbacks=[dynamicThreshold, dynamicThreshold2])
        K.set_value(self.model.layers[self.numLayers*2].threshold, 0.001)

    def evaluate(self, predictors, labels, batchSize=10, verbose=0):
        """
        Evaluates trained model on data
        """
        return self.model.evaluate(predictors, labels, batch_size=batchSize,
                                   verbose=verbose)[1]

    # function for retrieving a model's intrinsic equation
    def getEquation(self):
        """
        Prints learned equation of a trained model.
        """

        # prepares lists for weights and biases
        weights = [0 for i in range(int(len(self.model.get_weights())/2))]
        bias = [0 for i in range(int(len(self.model.get_weights())/2))]

        # pulls/separates weights and biases from a model
        for i in range(0, len(self.model.get_weights()), 2):
            weights[int(i/2)] = np.asarray(self.model.get_weights()[i])
            bias[int(i/2)] = np.asarray(self.model.get_weights()[i+1])

        # creates generic input vector
        X = make_symbolic(1, self.inputSize)

        for i in range(0, len(weights)):
            # computes the result of the next linear layer
            W = sympy.Matrix(weights[i])
            b = sympy.Transpose(sympy.Matrix(bias[i]))
            Y = sympy.zeros(1, b.cols)
            X = X*W + b

            # computes the result of the next nonlinear layer, if applicable
            if i != (len(weights) - 1):
                # computes the result of the unary component of the nonlinear
                # layer
                # iterating over unary input
                for j in range(self.nonlinearInfo[i][0]):
                    Y[0, j] = self.hypothesisSet[1][self.unaryFunctions[i][j]](
                            X[0, j])

                # computes the result of the binary component of the nonlinear
                # layer
                # iterating over binary input
                for j in range(self.nonlinearInfo[i][1]):
                    Y[0, j+self.nonlinearInfo[i][0]] = X[
                        0, j * 2 + self.nonlinearInfo[i][0]] * X[
                            0, j * 2 + self.nonlinearInfo[i][0] + 1]

                # removes final v rows which are now outdated
                for j in range(self.nonlinearInfo[i][0]
                               + self.nonlinearInfo[i][1], Y.cols):
                    Y.col_del(self.nonlinearInfo[i][0]
                              + self.nonlinearInfo[i][1])

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
        """
        Returns the sparsity of a trained model (number of active nodes)
        """

        # list of lists where ith element is list containing activity (in the
        # form of binary value) of outputs of ith layer
        layerSparsity = [[]
                         for i in range(int(len(self.model.get_weights())/2))]

        # iterating over layers
        for i in range(0, len(self.model.get_weights()), 2):

            # linear component
            # weights
            # iterating over columns (due to how the matrix mult. works)
            for j in range(len(self.model.get_weights()[i][0])):
                layerSparsity[int(i/2)].append(0)
                # iterating over rows
                for k in range(len(self.model.get_weights()[i])):
                    if ((i == 0 or layerSparsity[int(i/2) - 1][k] != 0) and
                            abs(self.model.get_weights()[i][k][j]) > minMag):
                        # if weight surpasses minimum magnitude and input was z
                        # active, output is active
                        layerSparsity[int(i/2)][j] = 1

            # biases
            for j in range(len(self.model.get_weights()[i+1])):
                if abs(self.model.get_weights()[i+1][j]) > minMag:
                    # if bias surpasses minimum magnitude, output is active
                    layerSparsity[int(i/2)][j] = 1

            # handling binary units in nonlinear layers where necessary
            # if not last layer (which has no nonlinear component)
            if i != len(self.model.get_weights()) - 2:
                u, v = self.nonlinearInfo[int(i/2)]
                # active if both units active
                for j in range(self.nonlinearInfo[int(i/2)][1]):
                    layerSparsity[int(i/2)][u + j] = layerSparsity[int(i/2)][
                        u + 2 * j] * layerSparsity[int(i/2)][
                            u + 2 * j + 1]
                del layerSparsity[int(i/2)][u + v:]
                # get rid of old values now

            # handling division layer
            # if last layer (i.e. has division component)
            if i == len(self.model.get_weights()) - 2:
                # active if both units active
                for j in range(int(len(layerSparsity[int(i/2)])/2)):
                    layerSparsity[int(i/2)][j] = layerSparsity[int(i/2)][
                        2 * j] * layerSparsity[int(i/2)][2 * j + 1]
                del layerSparsity[int(i/2)][
                    int(len(layerSparsity[int(i/2)])/2):]
                # get rid of old values now

        # return sum of flattened layerSparsity list (number of active oututs)
        return sum([item for sublist in layerSparsity for item in sublist])

    def setPipeline(self, pipeline):
        """
        Saves the scikit_learn pipeline used to modify training data so that
        the same pipeline can be applied to testing data
        """

        self.pipeline = pipeline

    def applyPipeline(self, x):
        """
        Applies a saved scikit_learn pipeline to a dataset
        """

        if self.pipeline is not None:
            for op in self.pipeline:
                x = op.transform(x)
        return x

    def odecompat(self, t, x):
        """
        Wrapper for Keras' predict function
        """

        x = np.reshape(x, (1, -1))
        if self.pipeline is not None:
            for op in self.pipeline:
                x = op.transform(x)

        prediction = self.model.predict(x)
        return prediction
