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
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.initializers import Zeros


class eqlLayer(Layer):
    """
    WORK IN PROGRESS: COMBINED LINEAR/NONLINEAR LAYERS
    """

    def __init__(self, nodeInfo, hypSet, unaryFunc, kernel_init,
                 bias_init=Zeros, regularization=0., **kwargs):
        self.nodeInfo = nodeInfo
        self.hypSet = hypSet
        self.unaryFunc = unaryFunc
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.regularization = tf.Variable(regularization,
                                          name='regularization',
                                          trainable=False, dtype=tf.float32)

        super(eqlLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        u, v = self.nodeInfo
        self.W = self.add_weight(name='kernel',
                                 shape=(int(input_shape[1]), u + 2 * v),
                                 initializer=self.kernel_init,
                                 trainable=True)
        self.Wtrimmer = tf.Variable(tf.ones_like(self.W), name='Wtrimmer',
                                    trainable=False)
        self.b = self.add_weight(name='bias',
                                 shape=(u + 2 * v, ),
                                 initializer=self.bias_init,
                                 trainable=True)
        self.btrimmer = tf.Variable(tf.ones_like(self.b), name='Wtrimmer',
                                    trainable=False)
        super(eqlLayer, self).build(input_shape)

    def call(self, x):
        # linear computation
        self.W = self.W * self.Wtrimmer
        self.b = self.b * self.btrimmer
        linOutput = tf.linalg.matmul(x, self.W) + self.b

        # nonlinear computation
        # unary functions
        u, v = self.nodeInfo
        output = self.hypSet[self.unaryFunc[0]](linOutput[:, :1])
        for i in range(1, u):
            output = tf.concat(
                    [output,
                     self.hypSet[self.unaryFunc[i]](linOutput[:, i:i+1])],
                    axis=1)
        # binary functions (multiplication)
        for i in range(u, u + 2 * v, 2):
            output = tf.concat(
                    [output,
                     tf.multiply(linOutput[:, i:i+1], linOutput[:, i+1:i+2])],
                    axis=1)

        # regularization
        regularizationLoss = (
                tf.reduce_sum(self.regularization * tf.abs(self.W))
                + tf.reduce_sum(self.regularization * tf.abs(self.b)))
        self.add_loss(regularizationLoss)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nodeInfo[0] + self.nodeInfo[1])


class divLayer(Layer):
    """
    WORK IN PROGRESS: COMBINED LINEAR/DIVISION LAYERS
    """

    def __init__(self, outShape, kernel_init, threshold=0.001,
                 regularization=0., bias_init=Zeros, loss=None, **kwargs):
        self.outShape = outShape
        self.kernel_init = kernel_init
        self.threshold = tf.Variable(threshold, name='threshold',
                                     trainable=False)
        self.regularization = tf.Variable(regularization,
                                          name='regularization',
                                          trainable=False, dtype=tf.float32)
        self.bias_init = bias_init
        self.loss = loss
        super(divLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='kernel',
                                 shape=(int(input_shape[1]), self.outShape*2),
                                 initializer=self.kernel_init,
                                 trainable=True)
        self.Wtrimmer = tf.Variable(tf.ones_like(self.W), name='Wtrimmer',
                                    trainable=False)
        self.b = self.add_weight(name='bias',
                                 shape=(self.outShape * 2, ),
                                 initializer=self.bias_init,
                                 trainable=True)
        self.btrimmer = tf.Variable(tf.ones_like(self.b), name='btrimmer',
                                    trainable=False)
        super(divLayer, self).build(input_shape)

    def call(self, x):
        # linear computation
        self.W = self.W * self.Wtrimmer
        self.b = self.b * self.btrimmer
        linOutput = tf.linalg.matmul(x, self.W) + self.b

        # division computation
        numerators = linOutput[:, ::2]
        denominators = linOutput[:, 1::2]
        # following three lines adapted from
        # https://github.com/martius-lab/EQL_Tensorflow
        zeros = tf.cast(denominators > self.threshold, dtype=tf.float32)
        denominatorsInverse = tf.reciprocal(tf.abs(denominators) + 1e-10)
        output = numerators * denominatorsInverse * zeros

        # negative denominator penalty
        denominatorLoss = tf.reduce_sum(
                tf.maximum(self.threshold - denominators,
                           tf.zeros_like(denominators)))
        self.add_loss(denominatorLoss)

        # regularization
        regularizationLoss = (
                tf.reduce_sum(self.regularization * tf.abs(self.W))
                + tf.reduce_sum(self.regularization * tf.abs(self.b)))
        self.add_loss(regularizationLoss)

        # passed custom loss
        if self.loss is not None:
            self.add_loss(self.loss(output))

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.outShape)


class EnergyConsReg(Regularizer):
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
