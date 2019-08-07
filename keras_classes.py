#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:44:11 2019

@author: elijahsheridan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Layer


class eqlLayer(Layer):
    """
    WORK IN PROGRESS: COMBINED LINEAR/NONLINEAR LAYERS
    """

    def __init__(self, nodeInfo, hypSet, unaryFunc, kernel_initializer=None,
                 bias_initializer=None, **kwargs):
        self.nodeInfo = nodeInfo
        self.hypSet = hypSet
        self.unaryFunc = unaryFunc
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        super(eqlLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        u, v = self.nodeInfo
        self.W = self.add_weight(name='kernel',
                                 shape=(input_shape[1], u + 2 * v),
                                 initializer=self.kernel_initializer,
                                 trainable=True)
        self.b = self.add_weight(name='bias',
                                 shape=(u + 2 * v, ),
                                 initializer=self.kernel_initializer,
                                 trainable=True)
        super(eqlLayer, self).build(input_shape)

    def call(self, x):
        # linear component
        linOutput = tf.linalg.matmul(x, self.W) + self.b

        # nonlinear component
        # unary functions
        u, v = self.nodeInfo
        nonlinOutput = self.hypSet[self.unaryFunc[0]](linOutput[:, :1])
        for i in range(1, u):
            nonlinOutput = tf.concat(
                    [nonlinOutput,
                     self.hypSet[self.unaryFunc[i]](linOutput[:, i:i+1])],
                    axis=1)

        # binary functions (multiplication)
        for i in range(u, u + 2 * v, 2):
            nonlinOutput = tf.concat(
                    [nonlinOutput,
                     tf.multiply(linOutput[:, i:i+1], linOutput[:, i+1:i+2])],
                    axis=1)


class Nonlinear(Layer):
    """
    EQL/EQL-Div Nonlinear Keras Layer

    # Arguments
        nodeInfo: a list containing two integers, the first of which gives the
            number of unary functions in the layer, and the second of which
            gives the number of binary functions (multiplication units) in the
            layer
        hypSet: a list of Python function which apply tensor-compatible,
            element-wise, R -> R operations: the hypothesis set of the layer
        unaryFunc: a list of integers with length nodeInfo[0], each integer
            falls in range [0, len(hypSet) - 1], and gives the index of the
            hypothesis set function to be used at each node
    """

    def __init__(self, nodeInfo, hypSet, unaryFunc, **kwargs):
        self.nodeInfo = nodeInfo
        self.hypSet = hypSet
        self.unaryFunc = unaryFunc
        super(Nonlinear, self).__init__(**kwargs)

    def call(self, linOutput):
        u, v = self.nodeInfo
        nonlinOutput = self.hypSet[self.unaryFunc[0]](linOutput[:, :1])
        for i in range(1, u):
            nonlinOutput = tf.concat(
                    [nonlinOutput,
                     self.hypSet[self.unaryFunc[i]](linOutput[:, i:i+1])],
                    axis=1)

        for i in range(u, u + 2 * v, 2):
            nonlinOutput = tf.concat(
                    [nonlinOutput,
                     tf.multiply(linOutput[:, i:i+1], linOutput[:, i+1:i+2])],
                    axis=1)

        return nonlinOutput

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nodeInfo[0] + self.nodeInfo[1])


class EnergyConsReg(keras.regularizers.Regularizer):
    """
    Energy Conservation Keras Activity Regularizer

    Penalizes a training model for the difference betwee the Hamiltonian of
    predicted values and the actual Hamiltonian value associated with the
    training data.

    SHOULD ONLY BE USED WITH TIMESERIES DATA OR OTHER CONSTANT ENERGY DATA

    Arguments
        energyFunc: a python function which uses tensorflow methods to compute
            the Hamiltonian associated with each member of a batch of predicted
            state
        energy: a float value giving the actual Hamilton of the data
        coef: a coefficient for scaling the energy error in the loss function
            (10^-5 recommended)
    """

    def __init__(self, energyFunc, energy, coef):
        self.energyFunc = energyFunc
        self.energy = energy
        self.coef = K.variable(coef, name='energyFunc')

    def __call__(self, x):
        """
        Adds the sum of |E_pred - E_true| for each predicted vector in
        minibatch to the loss function
        """

        return self.coef * tf.reduce_sum(
                tf.abs(self.energyFunc(x) - self.energy))

    def get_config(self):
        return {'Energy Function': self.energyFunc, 'energy': self.energy,
                'Coefficient': self.coef}


class Division(Layer):
    """
    EQL-Div Division Keras Layer (IMPROVED??)

    # Arguments:
        threshold: float, denominators below this value are not accepted for
            division (0 is returned for that particular division instance
            instead)
        loss: keras loss function to be added to global loss using
            Layer.add_loss()
    """

    def __init__(self, threshold=0.001, loss=None, **kwargs):
        self.threshold = K.variable(threshold, name='threshold')
        self.loss = loss
        super(Division, self).__init__(**kwargs)

    def call(self, linOutput):
        numerators = linOutput[:, ::2]
        denominators = linOutput[:, 1::2]
        # following three lines adapted from
        # https://github.com/martius-lab/EQL_Tensorflow
        zeros = tf.cast(denominators > self.threshold, dtype=tf.float32)
        denominators = tf.reciprocal(tf.abs(denominators) + 1e-10)
        divOutput = numerators * denominators * zeros
        if self.loss is not None:
            self.add_loss(self.loss(divOutput))
        return divOutput

    def compute_output_shape(self, input_shape):
        return (input_shape[0], int(input_shape[1]/2))


class DynamReg(keras.regularizers.Regularizer):
    """
    Dynamic Keras Regularizer

    No change from Keras regularizer, except l1 and l2 are now tensorflow
    variables which can be changed during the training schedule.
    """

    def __init__(self, l1=0., l2=0.):
        # this is the important part: this has to be a variable (i.e.
        # modifiable)
        self.l1 = K.variable(l1, name='weightRegL1', dtype=tf.float32)
        self.l2 = K.variable(l2, name='weightRegL2', dtype=tf.float32)
        self.uses_learning_phase = True
        self.p = None

    def __call__(self, x):
        regularization = 0.
        if self.l1 != 0:
            regularization += tf.reduce_sum(self.l1 * K.abs(x))
        if self.l2 != 0:
            regularization += tf.reduce_sum(self.l2 * K.square(x))
        return regularization

    def get_config(self):
        return {'l1': self.l1, 'l2': self.l2}


class ConstantL0(keras.constraints.Constraint):
    """
    Constant L0 Norm Keras Constraint

    # Arguments
        toZero: boolean tensor with same shape as the weights of the layer the
            constraint is being applied to. Should contain "true" in all
            positions where weight elements are less than the normThreshold
            (and thus where weight elements should be set to zero to preserve
            L0 norm)
    """

    def __init__(self, toZero):
        self.toZero = K.variable(toZero, name='toZero', dtype=tf.bool)

    def __call__(self, w):
        return tf.where(self.toZero, tf.zeros_like(w), w)
        # ^^replaces weights matrix entries with original value if greater than
        # threshold, zero otherwise

    def get_config(self):
        return {'toZero': self.toZero}


class DenominatorPenalty(keras.regularizers.Regularizer):
    """
    Denominator Penalty Keras Activity Regularizer

    # Arguments
        divThreshold: float, denominators below this number are not accepted
        and are penalized.
    """

    def __init__(self, divThreshold=0.001):
        self.divThreshold = K.variable(divThreshold, name='divThreshold')

    def __call__(self, x):
        """
        Regularization penalty:

        Sum of max(divThreshold - x, 0) over all denominators x
        """

        x = tf.reshape(x, (-1, 2))
        output = K.sum(K.maximum(self.divThreshold - x, K.zeros_like(x)),
                       axis=0)[1]
        return output

    def get_config(self):
        return {"name": self.__class__.__name__,
                "threshold": self.divThreshold}
