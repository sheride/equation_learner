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
from tensorflow.keras.initializers import Zeros, RandomNormal


class Connected(Layer):
    def __init__(self, outputShape, kernel_initializer=RandomNormal,
                 bias_initializer=Zeros, regularization=0., **kwargs):
        self.outputShape = outputShape
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularization = tf.Variable(regularization,
                                          name='regularization',
                                          trainable=False, dtype=tf.float32)
        super(Connected, self).__init__(**kwargs)

    def Wconstraint(self, w):
        return w * self.Wtrimmer

    def bconstraint(self, b):
        return b * self.btrimmer

    def build(self, inputShape):
        self.W = self.add_weight(name='kernel',
                                 shape=(int(inputShape[1]), self.outputShape),
                                 initializer=self.kernel_initializer,
                                 constraint=self.Wconstraint,
                                 trainable=True)
        self.Wtrimmer = tf.Variable(tf.ones_like(self.W), name='Wtrimmer',
                                    trainable=False)
        self.b = self.add_weight(name='bias',
                                 shape=(self.outputShape),
                                 initializer=self.bias_initializer,
                                 constraint=self.bconstraint,
                                 trainable=True)
        self.btrimmer = tf.Variable(tf.ones_like(self.b), name='btrimmer',
                                    trainable=False)
        super(Connected, self).build(inputShape)

    def call(self, x):
        output = tf.matmul(x, self.W) + self.b

        # regularization
        regularizationLoss = self.regularization * (
                tf.reduce_sum(tf.abs(self.W)) + tf.reduce_sum(tf.abs(self.b)))
        self.add_loss(regularizationLoss)

        return output

    def compute_output_shape(self, inputShape):
        return (inputShape[0], self.outputShape)


class EqlLayer(Connected):
    """
    WORK IN PROGRESS: COMBINED LINEAR/NONLINEAR LAYERS
    """

    def __init__(self, nodeInfo, hypSet, unaryFunc, **kwargs):
        self.nodeInfo = nodeInfo
        self.hypSet = hypSet
        self.unaryFunc = unaryFunc
        super(EqlLayer, self).__init__(nodeInfo[0] + 2 * nodeInfo[1], **kwargs)

    def build(self, inputShape):
        super(EqlLayer, self).build(inputShape)

    def call(self, x):
        linOutput = super(EqlLayer, self).call(x)

        u, v = self.nodeInfo
        output = [self.hypSet[self.unaryFunc[i]](linOutput[:, i:i+1])
                  for i in range(u)]
        output.extend([linOutput[:, i:i+1] * linOutput[:, i+1:i+2]
                       for i in range(u, u+2*v, 2)])
        output = tf.concat(output, axis=1)

        return output

    def compute_output_shape(self, inputShape):
        return (inputShape[0], self.nodeInfo[0] + self.nodeInfo[1])


class DivLayer(Connected):
    """
    WORK IN PROGRESS: COMBINED LINEAR/DIVISION LAYERS
    """

    def __init__(self, outputShape, threshold=0.001, loss=None, **kwargs):
        self.outputShape = outputShape
        self.threshold = tf.Variable(threshold, name='threshold',
                                     trainable=False)
        self.loss = loss
        super(DivLayer, self).__init__(outputShape*2, **kwargs)

    def build(self, inputShape):
        super(DivLayer, self).build(inputShape)

    def call(self, x):
        linOutput = super(DivLayer, self).call(x)

        numerators = linOutput[:, ::2]
        denominators = linOutput[:, 1::2]
        # following three lines adapted from
        # https://github.com/martius-lab/EQL_Tensorflow
        zeros = tf.cast(denominators > self.threshold, dtype=tf.float32)
        denominatorsInverse = tf.math.reciprocal(tf.abs(denominators) + 1e-10)
        output = numerators * denominatorsInverse * zeros

        # negative denominator penalty
        denominatorLoss = tf.reduce_sum(
                tf.maximum(self.threshold - denominators,
                           tf.zeros_like(denominators)))
        self.add_loss(denominatorLoss)

        # passed custom loss
        if self.loss is not None:
            self.add_loss(self.loss(output))

        return output

    def compute_output_shape(self, inputShape):
        return (inputShape[0], self.outputShape)


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
