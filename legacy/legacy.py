#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:04:31 2019

@author: elijahsheridan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.constraints import Constraint


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
        self.threshold = tf.Variable(threshold, name='threshold',
                                     trainable=False)
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


class DynamReg(Regularizer):
    """
    Dynamic Keras Regularizer

    No change from Keras regularizer, except l1 and l2 are now tensorflow
    variables which can be changed during the training schedule.
    """

    def __init__(self, l1=0., l2=0.):
        self.l1 = K.variable(l1, name='weightRegL1', dtype=tf.float32)
        self.l2 = K.variable(l2, name='weightRegL2', dtype=tf.float32)
        self.uses_learning_phase = True
        self.p = None

    def __call__(self, x):
        regularization = 0.
        if self.l1 != 0:
            regularization += tf.reduce_sum(self.l1 * tf.abs(x))
        if self.l2 != 0:
            regularization += tf.reduce_sum(self.l2 * tf.square(x))
        return regularization

    def get_config(self):
        return {'l1': self.l1, 'l2': self.l2}


class ConstantL0(Constraint):
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


class DenominatorPenalty(Regularizer):
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
        output = tf.reduce_sum(
                tf.maximum(self.divThreshold - x, tf.zeros_like(x)), axis=0)[1]
        return output

    def get_config(self):
        return {"name": self.__class__.__name__,
                "threshold": self.divThreshold}


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
