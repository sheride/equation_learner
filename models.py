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

"""
Hypothesis Set Functions

"""

# identity map
def f1(x):
    return x


# sine function
def f2(x):
    return K.sin(x)


# cosine function
def f3(x):
    return K.cos(x)


# sigmoid/logistic function
def f4(x):
    return 1 / (1 + K.exp(-1 * x))


# division with threshold (0.001)
def f5(x):
    return tf.where(K.less(x, K.zeros_like(x)+0.001),
                    K.zeros_like(x),
                    K.pow(x, -1))

"""
EQL/EQL-div Helper Functions

"""

def getNonlinearInfo(numHiddenLayers, numBinary, unaryPerBinary):
    nonlinearInfo = [0 for i in range(numHiddenLayers)]
    for i in range(numHiddenLayers):
        v = np.random.choice(numBinary)  # binary nodes
        u = unaryPerBinary * v  # unary nodes
        nonlinearInfo[i] = [u, v]
    return nonlinearInfo

# Root mean squared loss
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# NOT MINE: function for making n x m matrix of symbols
def make_symbolic(n, m):
    rows = []
    for i in range(n):
        col = []
        for j in range(m):
            col.append(sympy.Symbol('x%d' % (j+1)))
        rows.append(col)
    return sympy.Matrix(rows)

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
        hypothesisSet: list of unary (R -> R) functions (implemented using 
            Keras backend to enable element-wise application to tensors) to be 
            used for nonlinear map layers. In practice, usually contains 
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
            self.unaryFunctions = [[j % len(hypothesisSet) for j in range(self.nonlinearInfo[i][0])] for i in range(numLayers-1)]
            self.layers[0] = Input((inputSize,), name='input')

            # Create all hidden layers (linear and nonlinear components)
            for i in range(1, (self.numLayers-1)*2, 2):
                # Size of input to layer
                linIn = int(self.layers[i-1].shape[1])
                # Unary, binary width of layer
                u, v = self.nonlinearInfo[int((i-1)/2)]
                # Dense/linear component of layer 'i'
                stddev = np.sqrt(1 / (linIn * (u + 2 * v)))  #std. dev. of dist. for starting weight values
                randNorm = RandNorm(0, stddev=stddev, seed=2000)
                self.layers[i] = Dense(u + 2 * v,
                                       kernel_initializer=randNorm,
                                       bias_initializer='zeros',
                                       kernel_regularizer=my.dynamReg(0),
                                       bias_regularizer=my.dynamReg(0),
                                       kernel_constraint=my.ConstL0(tf.cast(K.zeros((linIn, u + 2 * v)), tf.bool)),
                                       bias_constraint=my.ConstL0(tf.cast(K.zeros((self.nonlinearInfo[int((i-1)/2)][0] + 2 * self.nonlinearInfo[int((i-1)/2)][1])), tf.bool)),
                                       name='linear'+str(int((i+1)/2)))(self.layers[i-1])  # creates the layer

                # Non-linear component of layer 'i'
                self.layers[i+1] = my.Nonlinear(self.nonlinearInfo[int((i-1)/2)],
                                                self.hypothesisSet,
                                                self.unaryFunctions[int((i-1)/2)],
                                                name='nonlinear'+str(int((i+1)/2)))(self.layers[i])
            # Final layer
            stddev = np.sqrt(1 / (self.outputSize * int(self.layers[self.numLayers*2-2].shape[1])))  # std. dev. of dist. for starting weight values
            randNorm = RandNorm(0, stddev=stddev, seed=2000)  # dist. for starting weight values
            self.layers[numLayers * 2 - 1] = Dense(self.outputSize, 
                                                   kernel_initializer=randNorm,
                                                   bias_initializer='zeros',
                                                   kernel_regularizer=my.dynamReg(0),
                                                   bias_regularizer=my.dynamReg(0),
                                                   kernel_constraint = my.ConstL0(tf.cast(K.zeros((int(self.layers[numLayers*2-2].shape[1]),self.outputSize)), tf.bool )),
                                                   bias_constraint = my.ConstL0(tf.cast(K.zeros((self.outputSize,)), tf.bool) ),
                                                   name='linear'+str(self.numLayers))(self.layers[self.numLayers*2-2])  # creates the layer

            # Optimizer
            optimizer = keras.optimizers.Adam(lr=self.learningRate)
    
            # Model
            self.model = keras.Model(inputs=self.layers[0], 
                                     outputs=self.layers[self.numLayers*2-1])
    
            # Compilation
            self.model.compile(optimizer=optimizer, loss='mse', metrics=[rmse])

    def fit(self, predictors, labels, numEpoch, reg, batchSize=20, 
            threshold=0.1, verbose=0):
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
                        np.less(np.abs(weight), np.full(weight.shape, 
                        threshold)))
            K.set_value(self.model.layers[i].bias_constraint.toZero, 
                        np.less(np.abs(bias), np.full(bias.shape, threshold)))
        self.model.fit(predictors, labels, epochs=int(numEpoch*(1/20)), 
                       batch_size=batchSize, verbose=verbose)

    def evaluate(self, predictors, labels, batchSize=10, verbose=0):
        return self.model.evaluate(predictors, labels, batch_size=batchSize, 
                                   verbose=verbose)[1]

    # function for retrieving a model's intrinsic equation
    def getEquation(self):
        # prepares lists for weights and biases
        weights = [0 for i in range(int(len(self.model.get_weights())/2))]
        bias = [0 for i in range(int(len(self.model.get_weights())/2))]

        # pulls/separates weights and biases from a model
        for i in range(0, len(self.model.get_weights()), 2):
            weights[int(i/2)] = np.asarray(self.model.get_weights()[i])
            bias[int(i/2)] = np.asarray(self.model.get_weights()[i+1])

        # creates generic input vector
        X = make_symbolic(1, self.inputSize)

        # defines sigmoid function
        sigm = sympy.Function("sigm")

        for i in range(0, len(weights)):
            # computes the result of the next linear layer
            W = sympy.Matrix(weights[i])
            b = sympy.Transpose(sympy.Matrix(bias[i]))
            X = X*W + b

            # computes the result of the next nonlinear layer, if applicable
            if i != (len(weights) - 1):
                Y = sympy.zeros(1, b.cols)
                # computes the result of the unary component of the nonlinear layer
                for j in range(self.nonlinearInfo[i][0]):  # iterating over unary input
                    if self.unaryFunctions[i][j] == 0:  # identity map
                        Y[0, j] = X[0, j]
                    elif self.unaryFunctions[i][j] == 1:  # sine function
                        Y[0, j] = sympy.sin(X[0, j])
                    elif self.unaryFunctions[i][j] == 2:  # cosine function
                        Y[0, j] = sympy.cos(X[0, j])
                    elif self.unaryFunctions[i][j] == 3:  # sigmoid function
                        Y[0, j] = sigm(X[0, j])

                # computes the result of the binary component of the nonlinear layer
                for j in range(self.nonlinearInfo[i][1]):  # iterating over binary input
                    Y[0, j+self.nonlinearInfo[i][0]] = X[0, j*2+self.nonlinearInfo[i][0]] * X[0, j*2+self.nonlinearInfo[i][0]+1]

                # removes final v rows which are now outdated
                for j in range(self.nonlinearInfo[i][0] + self.nonlinearInfo[i][1], Y.cols):
                    Y.col_del(self.nonlinearInfo[i][0] + self.nonlinearInfo[i][1])

                X = Y

        return X
   
    def plotSlice(self, function, xmin, xmax, step, width=10, height=10, 
                  save=False):
        # x values
        X = np.asarray([ [(i * step) + xmin for j in range(int(self.inputSize))] for i in range(int((xmax - xmin)/step))])
        # goal function values
        F_Y = np.apply_along_axis(function, 1, X)
        # model predictions
        model_Y = self.model.predict(X)

        # reshaping
        X = np.transpose(X)

        # graph colors
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'pink', 
                  'brown']

        # creating subplots
        font = {'family': 'normal', 'weight': 'bold', 'size': 22}

        fig, axs = plt.subplots(self.outputSize, figsize=(width, height))
        fig.suptitle('Title', **font)
        plt.xlabel('X-Axis', **font)
        plt.ylabel('Y-Axis', **font)
        axs.tick_params(axis='both', which='major', labelsize=18)
        axs.tick_params(axis='both', which='minor', labelsize=18)

        # graphing
        if self.outputSize == 1:
            axs.plot(X[0], F_Y, color=colors[0], linewidth=2.5, linestyle='-', 
                     label='Function')
            axs.plot(X[0], model_Y, color=colors[0], linewidth=1.5, 
                     linestyle=':', label='model')
        else:
            model_Y = np.transpose(model_Y)
            F_Y = np.transpose(F_Y)
            for i in range(self.outputSize):
                axs[i].plot(X[0], F_Y[i], color=colors[i], linewidth=2.5, 
                   linestyle='-', label='Function')
                axs[i].plot(X[0], model_Y[i], color=colors[i], linewidth=1.5, 
                   linestyle=':', label='model')

        if save:
            plt.savefig(self.name + '.png', bbox_inches='tight', dpi=300)

        plt.close()

    # percent error function
    def percentError(self, predictors, labels):
        labels = np.reshape(labels, (-1, self.outputSize))
        predictions = self.model.predict(predictors)
        error = np.divide(np.abs(predictions - labels), np.abs(labels))
        error = np.sum(error,0)
        error *= (100 / labels.shape[0])
        return error
    
    def sparsity(self, minMag = 0.01):
        # list of lists where ith element is list containing activity (in the 
        # form of a binary value) of outputs of ith layer
        layerSparsity = [ [] for i in range(int(len(self.model.get_weights())/2)) ]
        
        # iterating over layers
        for i in range(0,len(self.model.get_weights()),2):
 
            # linear component
            # weights
            # iterating over columns (due to how the matrix mult. works)
            for j in range(len(self.model.get_weights()[i][0])):
                layerSparsity[int(i/2)].append(0)
                # iterating over rows
                for k in range(len(self.model.get_weights()[i])):
                    if (i == 0 or layerSparsity[int(i/2) - 1][k] != 0) and abs(self.model.get_weights()[i][k][j]) > minMag:
                        layerSparsity[int(i/2)][j] = 1  # if weight surpasses minimum magnitude and input was active, output is active

            # biases
            for j in range(len(self.model.get_weights()[i+1])):
                if abs(self.model.get_weights()[i+1][j]) > minMag:
                    # if bias surpasses minimum magnitude, output is active
                    layerSparsity[int(i/2)][j] = 1
    
        # handling binary units in nonlinear layers where necessary
        # if not last layer (which has no nonlinear component)
        if i != len(self.model.get_weights()) - 2:
            # active if both units active
            for j in range(self.nonlinearInfo[int(i/2)][1]):
                layerSparsity[int(i/2)][self.nonlinearInfo[int(i/2)][0] + j] = layerSparsity[int(i/2)][self.nonlinearInfo[int(i/2)][0] + 2 * j] * layerSparsity[int(i/2)][self.nonlinearInfo[int(i/2)][0] + 2 * j + 1]
            del layerSparsity[int(i/2)][self.nonlinearInfo[int(i/2)][0] + self.nonlinearInfo[int(i/2)][1]:]  # get rid of old values now

        # return sum of flattened layerSparsity list (number of active oututs)
        return sum([item for sublist in layerSparsity for item in sublist])

#
# EQL-DIV KERAS MODEL
#


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
        hypothesisSet: list of unary (R -> R) functions (implemented using 
            Keras backend to enable element-wise application to tensors) to be 
            used for nonlinear map layers. In practice, usually contains 
            identity, sine, cosine, and sigmoid.
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

        with tf.name_scope(self.name) as scope:
            self.unaryFunctions = [ [ j % len(hypothesisSet) for j in range(self.nonlinearInfo[i][0]) ] for i in range(self.numLayers-1) ]
            self.layers[0] = Input((self.inputSize,), name='input')

            # Create all hidden layers (linear and nonlinear components)
            for i in range(1, (self.numLayers-1)*2, 2):
                # Size of input to layer
                linIn = int(self.layers[i-1].shape[1])
                # Unary, binary width of layer
                u, v = self.nonlinearInfo[int((i-1)/2)]

                # Dense/linear component of layer 'i'
                stddev = np.sqrt(1 / (linIn * (u + 2 * v)))
                randNorm = RandNorm(0, stddev=stddev, seed=2000)
                self.layers[i] = Dense(u + 2 * v,
                                       kernel_initializer=randNorm,
                                       bias_initializer='zeros',
                                       kernel_regularizer=my.dynamReg(0),
                                       bias_regularizer=my.dynamReg(0),
                                       kernel_constraint=my.ConstL0(tf.cast(K.zeros((linIn, u + 2 * v)), tf.bool)),
                                       bias_constraint=my.ConstL0(tf.cast(K.zeros((u + 2 * v,)), tf.bool)),
                                       name='linear'+str(int((i+1)/2)))(self.layers[i-1])

                # Non-linear component of layer 'i'
                self.layers[i+1] = my.Nonlinear(self.nonlinearInfo[int((i-1)/2)], 
                           self.hypothesisSet, 
                           self.unaryFunctions[int((i-1)/2)],
                           name='nonlinear'+str(int((i+1)/2)))(self.layers[i])
            
            # Final layer
            stddev = np.sqrt(1 / (self.outputSize * int(self.layers[self.numLayers*2-2].shape[1]))) 
            randNorm = RandNorm(0, stddev=stddev, seed=2000)
            self.layers[self.numLayers * 2 - 1] = Dense(outputSize * 2,
                                                        kernel_initializer=randNorm,
                                                        bias_initializer='zeros',
                                                        kernel_regularizer=my.dynamReg(0),
                                                        bias_regularizer=my.dynamReg(0),
                                                        activity_regularizer=my.CustomizedActivationRegularizer(self.divThreshold),
                                                        kernel_constraint=my.ConstL0(tf.cast(K.zeros((int(self.layers[numLayers*2-2].shape[1]), self.outputSize*2)), tf.bool)),
                                                        bias_constraint=my.ConstL0(tf.cast(K.zeros((self.outputSize * 2,)), tf.bool)),
                                                        name='linear'+str(self.numLayers))(self.layers[self.numLayers*2-2])  # creates the layer

            # Division final layer component
            self.layers[self.numLayers*2] = my.DivisionMap(self.divThreshold)(self.layers[self.numLayers*2-1])

            # Optimizer
            optimizer = keras.optimizers.Adam(lr=self.learningRate)

            # Model
            self.model = keras.Model(inputs=self.layers[0], 
                                     outputs=self.layers[self.numLayers * 2])

            # Compilation
            self.model.compile(optimizer=optimizer, loss='mse', metrics=[rmse])

    def fit(self, predictors, labels, numEpoch, regStrength, batchSize=20, 
            normThreshold=0.001, verbose=0):            
        dynamicThreshold = LambCall(on_epoch_begin = lambda epoch, logs: K.set_value(self.model.layers[self.numLayers*2].threshold, 1 / np.sqrt(epoch+1)))

        # PHASE 1: NO REGULARIZATION (T/4)
        self.model.fit(predictors, labels, epochs=int(numEpoch*(1/4)), batch_size=batchSize, verbose=verbose, callbacks=[dynamicThreshold])
        dynamicThreshold = LambCall(on_epoch_begin = lambda epoch, logs: K.set_value(self.model.layers[self.numLayers*2].threshold, 1 / np.sqrt(int(numEpoch*(1/4))+numEpoch+1)))

        # PHASE 2: REGULARIZATION (7T/10)
        for i in range(1,len(self.model.layers),2):
            K.set_value(self.model.layers[i].kernel_regularizer.l1, 
                        regStrength)
            K.set_value(self.model.layers[i].bias_regularizer.l1, 
                        regStrength)
        self.model.fit(predictors, labels, epochs=int(numEpoch*(7/10)), batch_size=batchSize, verbose=verbose, callbacks=[dynamicThreshold])
        dynamicThreshold = LambCall(on_epoch_begin = lambda epoch, logs: K.set_value(self.model.layers[self.numLayers*2].threshold, 1 / np.sqrt(int(numEpoch*(1/4))+int(numEpoch*(7/10))+numEpoch+1)))

        # PHASE 3: NO REGULARIZATION, L0 NORM PRESERVATION (T/20)
        for i in range(1, len(self.model.layers), 2):
            K.set_value(self.model.layers[i].kernel_regularizer.l1, 0)
            K.set_value(self.model.layers[i].bias_regularizer.l1, 0)
            weight, bias = self.model.layers[i].get_weights()
            K.set_value(self.model.layers[i].kernel_constraint.toZero,
                        np.less(np.abs(weight), np.full(weight.shape,
                        normThreshold)))
            K.set_value(self.model.layers[i].bias_constraint.toZero,
                        np.less(np.abs(bias), np.full(bias.shape,
                        normThreshold)))
        self.model.fit(predictors, labels, epochs=int(numEpoch*(1/20)), 
                       batch_size=batchSize, verbose=verbose, 
                       callbacks=[dynamicThreshold])
        K.set_value(self.model.layers[self.numLayers*2].threshold, 0.001)

    def evaluate(self, predictors, labels, batchSize=10, verbose=0):
        return self.model.evaluate(predictors, labels, batch_size=batchSize, 
                                   verbose=verbose)[1]

    #function for retrieving a model's intrinsic equation
    def getEquation(self):        
        # prepares lists for weights and biases
        weights = [0 for i in range(int(len(self.model.get_weights())/2))]
        bias = [0 for i in range(int(len(self.model.get_weights())/2))]

        # pulls/separates weights and biases from a model
        for i in range(0, len(self.model.get_weights()), 2):
            weights[int(i/2)] = np.asarray(self.model.get_weights()[i])
            bias[int(i/2)] = np.asarray(self.model.get_weights()[i+1])

        # creates generic input vector
        X = make_symbolic(1, self.inputSize)

        # defines sigmoid function
        sigm = sympy.Function("sigm")

        for i in range(0, len(weights)):
            # computes the result of the next linear layer
            W = sympy.Matrix(weights[i])
            b = sympy.Transpose(sympy.Matrix(bias[i]))
            Y = sympy.zeros(1, b.cols)
            X = X*W + b

            # computes the result of the next nonlinear layer, if applicable
            if i != (len(weights) - 1):
                # computes the result of the unary component of the nonlinear layer
                for j in range(self.nonlinearInfo[i][0]):  # iterating over unary input
                    if self.unaryFunctions[i][j] == 0:  # identity map
                        Y[0, j] = X[0, j]
                    elif self.unaryFunctions[i][j] == 1:  # sine function
                        Y[0, j] = sympy.sin(X[0, j])
                    elif self.unaryFunctions[i][j] == 2:  # cosine function
                        Y[0, j] = sympy.cos(X[0, j])
                    elif self.unaryFunctions[i][j] == 3:  # sigmoid function
                        Y[0, j] = sigm(X[0, j])

                # computes the result of the binary component of the nonlinear layer
                for j in range(self.nonlinearInfo[i][1]): #iterating over binary input
                    Y[0, j+self.nonlinearInfo[i][0]] = X[0, j*2+self.nonlinearInfo[i][0]] * X[0, j*2+self.nonlinearInfo[i][0]+1]

                # removes final v rows which are now outdated
                for j in range(self.nonlinearInfo[i][0] + self.nonlinearInfo[i][1], Y.cols):
                    Y.col_del(self.nonlinearInfo[i][0] + self.nonlinearInfo[i][1])

                X = Y

            if i == (len(weights) - 1):
                # computes the result of the binary component of the nonlinear layer
                for j in range(int(X.cols/2)):  # iterating over binary input
                    if sympy.Abs(X[0, j*2+1]) == 0:
                        Y[0, j] = 0
                    else:
                        Y[0, j] = X[0, j*2] / X[0, j*2+1]
                for j in range(int(X.cols/2)):
                    Y.col_del(int(X.cols/2))

                X = Y

        return X

    def plotSlice(self, function, xmin, xmax, step, width=10, height=10, save=False):
        # x values
        X = np.asarray([ [(i * step) + xmin for j in range(int(self.inputSize))] for i in range(int((xmax - xmin)/step))])
        # goal function values
        F_Y = np.apply_along_axis(function, 1, X)
        # model predictions
        model_Y = self.model.predict(X)
        
        # reshaping
        X = np.transpose(X)
        
        # graph colors
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'pink', 
                  'brown']

        # creating subplots
        font = {'family': 'normal', 'weight': 'bold', 'size': 22}

        fig, axs = plt.subplots(self.outputSize, figsize=(width, height))
        fig.suptitle('Title', **font)
        plt.xlabel('X-Axis', **font)
        plt.ylabel('Y-Axis', **font)

        # graphing
        if self.outputSize == 1:
            axs.plot(X[0], F_Y, color=colors[0], linewidth=2.5, linestyle='-', 
                     label='Function')
            axs.plot(X[0], model_Y, color=colors[0], linewidth=1.5, 
                     linestyle=':', label='model')
        else:
            model_Y = np.transpose(model_Y)
            F_Y = np.transpose(F_Y)
            for i in range(self.outputSize):
                axs[i].plot(X[0], F_Y[i], color=colors[i], linewidth=2.5, 
                   linestyle='-', label='Function')
                axs[i].plot(X[0], model_Y[i], color=colors[i], linewidth=1.5, 
                   linestyle=':', label='model')
        
        if save:
            plt.savefig(self.name + str(np.random.randint(100)) + '.png', 
                        bbox_inches='tight', dpi=300)

        plt.close()

    #percent error function
    def percentError(self, predictors, labels):
        error = [0 for i in range(self.outputSize)]

        if self.outputSize == 1:
            for i in range(len(predictors)):
                sample = np.reshape(predictors[i], (1, self.inputSize))
                error[0] += (np.abs(self.model.predict(sample)[0][0] - labels[i]) / abs(labels[i]))
            for i in range(len(error)):
                error[i] *= (100 / len(labels))
        else:
            for i in range(len(predictors)):
                for j in range(len(error)):
                    sample = np.reshape(predictors[i], (1, self.inputSize))
                    error[j] += (np.abs(self.model.predict(sample)[0][j] - labels[i][j]) / abs(labels[i][j]))
                for i in range(len(error)):
                    error[i] *= (100 / len(labels))

        return error

    def sparsity(self, minMag = 0.01):
        # list of lists where ith element is list containing activity (in the form of binary value) of outputs of ith layer
        layerSparsity = [ [] for i in range(int(len(self.model.get_weights())/2)) ]

        # iterating over layers
        for i in range(0,len(self.model.get_weights()),2):

            # linear component
            # weights
            for j in range(len(self.model.get_weights()[i][0])):  # iterating over columns (due to how the matrix mult. works)
                layerSparsity[int(i/2)].append(0)
                for k in range(len(self.model.get_weights()[i])):  # iterating over rows
                    if (i == 0 or layerSparsity[int(i/2) - 1][k] != 0) and abs(self.model.get_weights()[i][k][j]) > minMag:
                        layerSparsity[int(i/2)][j] = 1  # if weight surpasses minimum magnitude and input was active, output is active
        
            # biases
            for j in range(len(self.model.get_weights()[i+1])):
                if abs(self.model.get_weights()[i+1][j]) > minMag:
                    layerSparsity[int(i/2)][j] = 1  # if bias surpasses minimum magnitude, output is active

            # handling binary units in nonlinear layers where necessary
            if i != len(self.model.get_weights()) - 2:  # if not last layer (which has no nonlinear component)
                for j in range(self.nonlinearInfo[int(i/2)][1]):  # active if both units active
                    layerSparsity[int(i/2)][self.nonlinearInfo[int(i/2)][0] + j] = layerSparsity[int(i/2)][self.nonlinearInfo[int(i/2)][0] + 2 * j] * layerSparsity[int(i/2)][self.nonlinearInfo[int(i/2)][0] + 2 * j + 1]
                del layerSparsity[int(i/2)][self.nonlinearInfo[int(i/2)][0] + self.nonlinearInfo[int(i/2)][1]:]  # get rid of old values now

            # handling division layer
            if i == len(self.model.get_weights()) - 2:  # if last layer (i.e. has division component)
                for j in range(int(len(layerSparsity[int(i/2)])/2)):  # active if both units active
                    layerSparsity[int(i/2)][j] = layerSparsity[int(i/2)][2*j] * layerSparsity[int(i/2)][2*j + 1]
                del layerSparsity[int(i/2)][int(len(layerSparsity[int(i/2)])/2):]  # get rid of old values now

        return sum([item for sublist in layerSparsity for item in sublist])  # return sum of flattened layerSparsity list (number of active oututs)
    
    def solve_ivp_compat(self, t, x):
        deriv = self.model.predict(np.reshape(x,(1,len(x))))
        return deriv