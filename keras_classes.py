#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:44:11 2019

@author: elijahsheridan
"""

#
# CUSTOM KERAS NONLINEAR MAP LAYER
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras import backend as K
from keras.layers import Layer
import tensorflow as tf

class Nonlinear(Layer):

    # initializing with values
    def __init__(self, nodeInfo, hypSet, unaryFunc, **kwargs):
        self.nodeInfo = nodeInfo
        self.hypSet = hypSet
        self.unaryFunc = unaryFunc
        super(Nonlinear, self).__init__(**kwargs)

    # behavior of non-linear layer
    def call(self, linOutput):        
        # renaming num of unary, binary functions for simplicity
        u = self.nodeInfo[0]
        v = self.nodeInfo[1]

        # splitting input into inputs for unary, binary sections
        unaryPart, binaryPart = tf.split(linOutput, [u, 2 * v], 1)

        # handling unary part
        # need to iterate over 'u' elements in axis '1' (the columns), 
        # reshaping to make this axis '0' (the rows)
        unaryPart = tf.transpose(unaryPart)
        # applying non-linear function to first row
        unaryOutput = self.hypSet[self.unaryFunc[0]](unaryPart[0])
        # iterating over remaning rows
        for i in range(1,u):
            unaryOutput = tf.concat([unaryOutput,
                                     self.hypSet[self.unaryFunc[i]](unaryPart[i])],
                                    0) 

        # ^^concatenating non-linear function result for row 'i' to all 
        # previous results
        # after loop, unaryOutput.shape = (?,) where ? % u = 0 (it's just one 
        # long row), this separates it out
        unaryOutput = tf.reshape(unaryOutput,(u,-1)) 
        unaryOutput = tf.transpose(unaryOutput)

        # handing binary part
        # need to iterate over '2*v' elements in axis '1' (the columns), 
        # reshaping to make this axis '0' (the rows)
        binaryPart = tf.transpose(binaryPart)
        # applying multiplication to first two rows
        binaryOutput = tf.math.multiply(binaryPart[0], binaryPart[1])
        # iterating over remaning row pairs
        for i in range(2,v*2,2):
            binaryOutput = tf.concat([binaryOutput,
                                      tf.math.multiply(binaryPart[i], 
                                                       binaryPart[i+1])],
                                     0)
                                      
        # ^^concatenating row multiplication result for rows 'i', 'i+1' to 
        # all previous results
        binaryOutput = tf.reshape(binaryOutput,(v,-1)) 
        # ^^after loop, binaryOutput.shape = (?,) where ? % v = 0 (it's just 
        # one long row), this separates it out
        binaryOutput = tf.transpose(binaryOutput)

        # combining unary, binary outputs
        # concatenating unary, binary
        nonLinOutput = tf.concat([unaryOutput,binaryOutput],1)  
        # reshaping to proper form
        nonLinOutput = tf.reshape(nonLinOutput,(-1,u+v)) 

        return nonLinOutput

    # returns the shape of a non-linear layer using the nodeInfo list
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nodeInfo[0] + self.nodeInfo[1])

class DivisionMap(Layer):
    # initializing with values
    def __init__(self, threshold, **kwargs):
        self.threshold = K.variable(threshold, name='threshold')
        super(DivisionMap, self).__init__(**kwargs)

    def call(self, linOutput):
        linOutputTrans = tf.transpose(linOutput)
        divOutput = tf.where(K.less(linOutputTrans[1],
                                    K.zeros_like(linOutputTrans[1]) + self.threshold), 
                             K.zeros_like(linOutputTrans[1]), 
                             tf.math.divide(linOutputTrans[0], 
                                            linOutputTrans[1]))

        for i in range(2, linOutputTrans.shape[0], 2):
            divOutput = tf.concat([divOutput, 
                                   tf.where(K.less(linOutputTrans[i+1],
                                                   K.zeros_like(linOutputTrans[i+1]) + self.threshold), 
                                            K.zeros_like(linOutputTrans[i+1]), 
                                            tf.math.divide(linOutputTrans[i], 
                                                           linOutputTrans[i+1]))], 
                                  0)

        divOutput = tf.reshape(divOutput, 
                               (int(int(linOutputTrans.shape[0])/2), -1))
        divOutput = tf.transpose(divOutput)

        return divOutput

    # returns the shape of a non-linear layer using the nodeInfo list
    def compute_output_shape(self, input_shape):
        return (input_shape[0], int(input_shape[1]/2))
    
#
# CUSTOM KERAS REGULARIZATION CLASS (REGULARIZATION COEFFICIENTS AS MUTABLE 
# TENSORFLOW VARIABLES)
#

class dynamReg(keras.regularizers.Regularizer):
    def __init__(self, l1=0., l2=0.):
        # this is the important part: this has to be a variable (i.e. 
        # modifiable)
        self.l1 = K.variable(l1, name='weightReg')
        self.l2 = K.cast_to_floatx(l2)
        self.uses_learning_phase = True
        self.p = None

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(x))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(x))
        return regularization  

#    
# CUSTOM KERAS L0 NORM PRESERVATION CONSTRAINT CLASS
#
            
class ConstL0(keras.constraints.Constraint):
    def __init__(self, toZero):
        self.toZero = K.variable(toZero, name='toZero', dtype=tf.bool )
        
    def __call__(self, w):
        return tf.where(self.toZero, K.zeros_like(w), w)
        # ^^replaces weights matrix entries with original value if greater than 
        # threshold, zero otherwise
        
#
# CUSTOM KERAS REGULARIZATION CLASS (EQLDIV NEGATIVE DENOMINATOR PENALTY)
#

class CustomizedActivationRegularizer(keras.regularizers.Regularizer):
    def __init__(self, divThreshold = 0.001):
        self.divThreshold = divThreshold

    def __call__(self, x):
        x = tf.reshape(x, (-1, 2))
        x = tf.transpose(x)
        output = K.sum(K.maximum(self.divThreshold - x, K.zeros_like(x)), 
                       axis=1)[1]
        return output

    def get_config(self):
        return {"name": self.__class__.__name__,
                "threshold": self.divThreshold}