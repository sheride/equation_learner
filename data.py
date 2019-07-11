#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:02:34 2019

@author: elijahsheridan
"""

import numpy as np
from scipy.integrate import solve_ivp as slv

#
# SINGLE PENDULUM DATA GENERATION
#

#Pendulum data parameters
h = 2 #hypercube dimensions
dataPoints = 1000 #number of data points per test/training set

#returns either 1 or -1, randomly
def genSign():
    return 1 if np.random.rand() < 0.5 else -1

#generates a random number in [-width, width]
def genNum(width):
    return np.random.rand() * 2 * width - width

#label function
def pendulumDerivatives(x):
    g = 9.8 #acceleration due to gravity
    return [x[1]/g, -np.sin(x[0])]

#function for generating pendulum angle/ang. vel. data
#w is width of hypercube of sampled points for training data
#n is number of data points
def genPendulumData(w, n):
    training_predictors = [[genNum(w),genNum(w)] for i in range(n)]
    training_labels = [pendulumDerivatives(x) for x in training_predictors]
    interpolation_predictors = [[genNum(w),genNum(w)] for i in range(n)]
    interpolation_labels = [pendulumDerivatives(x) for x in interpolation_predictors]
    extrapolation_near_predictors = [[genNum(w/4)+genSign()*(5*w/4),genNum(w/4)+genSign()*(5*w/4)] for i in range(n)]
    extrapolation_near_labels = [pendulumDerivatives(x) for x in extrapolation_near_predictors]
    extrapolation_far_predictors = [[genNum(w/2)+genSign()*(3*w/2),genNum(w/2)+genSign()*(3*w/2)] for i in range(n)]
    extrapolation_far_labels = [pendulumDerivatives(x) for x in extrapolation_far_predictors]
    all_data = [training_predictors, training_labels, interpolation_predictors, interpolation_labels, extrapolation_near_predictors, extrapolation_near_labels, extrapolation_far_predictors, extrapolation_far_labels]
    for i in range(len(all_data)):
        all_data[i] = np.asarray(all_data[i])
    return all_data

#
# DOUBLE PENDULUM COORDINATE DATA GENERATION
#

g = 9.8 #acc. due to gravity
firstInput = [np.pi/2,0,np.pi/2,0] #initial x,x_dot values
#secondInput = [np.pi/2,-1,3 *np.pi/2,-1]
secondInput = [np.pi/4,0,np.pi/4,0]

#function for ODE solver
def doublePendulumDerivativesSolver(t,x):
    return [
                x[1],
                (
                    -1*(x[1]**2)*np.sin(x[0] - x[2])*np.cos(x[0] - x[2]) 
                    +g * np.sin(x[2]) * np.cos(x[0] - x[2]) 
                    + -1 * (x[3]**2) * np.sin(x[0] - x[2]) 
                    + -1 * 2 * g * np.sin(x[0])
                ) / (
                    2 - ((np.cos(x[0] - x[2]))**2)
                ), 
                x[3], 
                (
                    (x[3]**2)*np.sin(x[0] - x[2])*np.cos(x[0] - x[2]) 
                    +g * 2 * np.sin(x[0]) * np.cos(x[0] - x[2]) 
                    +2 * x[1]**2 * np.sin(x[0] - x[2]) 
                    + -1*2 * g * np.sin(x[2])
                ) / (
                    2 - (np.cos(x[0] - x[2]))**2
                )
           ]

def fixRadians(x):
    return x % (2*np.pi) if x % (2*np.pi) < np.pi else (x % (2*np.pi) - 2 * np.pi)

def doublePendulumCoordinate(x):
    return [np.sin(x[0]), -np.cos(x[0]), np.sin(x[0]) + np.sin(x[1]), -np.cos(x[0]) - np.cos(x[1])]

def getDoublePendulumCoordinateData():
    firstOutput = slv(doublePendulumDerivativesSolver, [0,40], firstInput, first_step = 0.05, max_step = 0.05) #ODE solver
    secondOutput = slv(doublePendulumDerivativesSolver, [0,40], secondInput, first_step = 0.05, max_step = 0.05)
    
    interpolation_predictors = [ [ fixRadians(firstOutput.y[0][i]), fixRadians(firstOutput.y[2][i]) ] for i in range(len(firstOutput.y[0])) ]
    interpolation_labels = [ doublePendulumCoordinate(x) for x in interpolation_predictors ]
    extrapolation_predictors = [ [ fixRadians(secondOutput.y[0][i]), fixRadians(secondOutput.y[2][i]) ] for i in range(len(secondOutput.y[0])) ]
    extrapolation_labels = [ doublePendulumCoordinate(x) for x in extrapolation_predictors ]
    
    all_data = [interpolation_predictors, interpolation_labels, extrapolation_predictors, extrapolation_labels] 
    for i in range(len(all_data)):
        all_data[i] = np.asarray(all_data[i])
        
    return all_data

#
# EQUATION DATA GENERATION
#

# F-1 from paper
def Function1(x):
    return (1/3) * ( np.sin(np.pi * x[0]) + np.sin(2 * np.pi * x[1] + (np.pi/8)) + x[1] - x[2]*x[3] )

# F-2 from paper
def Function2(x):
    return (1/3) * ( np.sin(np.pi * x[0]) + x[1] * np.cos(2 * np.pi * x[0] + (np.pi/4)) + x[2] - x[3]**2 )

# F-3 from paper
def Function3(x):
    return (1/3) * ( (1 + x[1]) * np.sin(np.pi * x[0]) + x[1]*x[2]*x[3] )

# 
def getEquationData(w,n,f):
    training_predictors = [[genNum(w),genNum(w),genNum(w),genNum(w)] for i in range(n)]
    training_labels = [ f(training_predictors[i]) for i in range(n) ]
    interpolation_predictors = [[genNum(w),genNum(w),genNum(w),genNum(w)] for i in range(int(n/2))]
    interpolation_labels = [ f(interpolation_predictors[i]) for i in range(int(n/2)) ]
    extrapolation_near_predictors = [[genNum(w/4)+genSign()*(5*w/4),genNum(w/4)+genSign()*(5*w/4),genNum(w/4)+genSign()*(5*w/4),genNum(w/4)+genSign()*(5*w/4)] for i in range(int(n/2))]
    extrapolation_near_labels = [ f(extrapolation_near_predictors[i]) for i in range(int(n/2)) ]
    extrapolation_far_predictors = [[genNum(w/2)+genSign()*(3*w/2),genNum(w/2)+genSign()*(3*w/2),genNum(w/2)+genSign()*(3*w/2),genNum(w/2)+genSign()*(3*w/2)] for i in range(int(n/2))]
    extrapolation_far_labels = [ f(extrapolation_far_predictors[i]) for i in range(int(n/2)) ]
    all_data = [training_predictors, training_labels, interpolation_predictors, interpolation_labels, extrapolation_near_predictors, extrapolation_near_labels, extrapolation_far_predictors, extrapolation_far_labels]
    for i in range(len(all_data)):
        all_data[i] = np.asarray(all_data[i])
    return all_data

#
# DOUBLE PENDULUM DIFF EQ DATA GENERATION
#

g = 9.8 #acc. due to gravity

#label function 
def doublePendulumDerivatives(x):
    return [
                x[1],
                (
                    -1*(x[1]**2)*np.sin(x[0] - x[2])*np.cos(x[0] - x[2]) 
                    +g * np.sin(x[2]) * np.cos(x[0] - x[2]) 
                    + -1 * (x[3]**2) * np.sin(x[0] - x[2]) 
                    + -1 * 2 * g * np.sin(x[0])
                ) / (
                    (2 - ((np.cos(x[0] - x[2]))**2))
                ), 
                x[3], 
                (
                    (x[3]**2)*np.sin(x[0] - x[2])*np.cos(x[0] - x[2]) 
                    +g * 2 * np.sin(x[0]) * np.cos(x[0] - x[2]) 
                    +2 * x[1]**2 * np.sin(x[0] - x[2]) 
                    + -1*2 * g * np.sin(x[2])
                ) / (
                    (2 - (np.cos(x[0] - x[2]))**2)
                )
           ]
    
#label function 
def doublePendulumDerivativesSolveIVP(t,x):
    return [
                x[1],
                (
                    -1*(x[1]**2)*np.sin(x[0] - x[2])*np.cos(x[0] - x[2]) 
                    +g * np.sin(x[2]) * np.cos(x[0] - x[2]) 
                    + -1 * (x[3]**2) * np.sin(x[0] - x[2]) 
                    + -1 * 2 * g * np.sin(x[0])
                ) / (
                    (2 - ((np.cos(x[0] - x[2]))**2))
                ), 
                x[3], 
                (
                    (x[3]**2)*np.sin(x[0] - x[2])*np.cos(x[0] - x[2]) 
                    +g * 2 * np.sin(x[0]) * np.cos(x[0] - x[2]) 
                    +2 * x[1]**2 * np.sin(x[0] - x[2]) 
                    + -1*2 * g * np.sin(x[2])
                ) / (
                    (2 - (np.cos(x[0] - x[2]))**2)
                )
           ]

def genDoublePendulumDiffEqData(w, n):
    training_predictors = [ [ genNum(np.pi),genNum(w),genNum(np.pi),genNum(w) ] for i in range(n) ]
    training_labels = [ (np.asarray(doublePendulumDerivatives(training_predictors[i]))) for i in range(n) ]
    interpolation_predictors = [[genNum(np.pi),genNum(w),genNum(np.pi),genNum(w)] for i in range(n)]
    interpolation_labels = [ (np.asarray(doublePendulumDerivatives(interpolation_predictors[i]))) for i in range(n) ]
    extrapolation_near_predictors = [ [genNum(w/4)+genSign()*(5*w/4),genNum(w/4)+genSign()*(5*w/4),genNum(w/4)+genSign()*(5*w/4),genNum(w/4)+genSign()*(5*w/4)] for i in range(n)]
    extrapolation_near_labels = [ (np.asarray(doublePendulumDerivatives(extrapolation_near_predictors[i]))) for i in range(n) ]
    extrapolation_far_predictors = [ [genNum(w/2)+genSign()*(3*w/2),genNum(w/2)+genSign()*(3*w/2),genNum(w/2)+genSign()*(3*w/2),genNum(w/2)+genSign()*(3*w/2)] for i in range(n)]
    extrapolation_far_labels = [ (np.asarray(doublePendulumDerivatives(extrapolation_far_predictors[i]))) for i in range(n) ]
    all_data = [training_predictors, training_labels, interpolation_predictors, interpolation_labels, extrapolation_near_predictors, extrapolation_near_labels, extrapolation_far_predictors, extrapolation_far_labels]
    for i in range(len(all_data)):
        all_data[i] = np.asarray(all_data[i])
    return all_data

#
# 4-LATTICE DIFF EQ DATA GENERATION
#

N = 4 #number of masses
k = [50]*(N+1) #spring constants length
m = [1]*N #masses
#x0 = [np.pi/4,0]
x0 = [2,0,1,0,0.25,0,-1,0] #initial x,x_dot values

#even indices pos, odd inices vel
def NLatticeDerivatives(t,x):
    z = [0,0]
    z.extend(x)
    z.extend([0,0])
    return [ z[i+1] if i % 2 == 0 else (.1 / m[int((i-1)/2) - 1]) * ( k[int((i-1)/2) - 1] * z[i-3] - ( k[int((i+1)/2) - 1] + k[int((i+1)/2) - 1] ) * z[i-1] + k[int((i+1)/2) - 1] * z[i+1] ) for i in range(2,len(z)-2) ]            

#even indices pos, odd inices vel
def NLatticeDerivativesSimple(x):
    z = [0,0]
    z.extend(x)
    z.extend([0,0])
    return [ z[i+1] if i % 2 == 0 else (.1/m[int((i-1)/2) - 1]) * ( k[int((i-1)/2) - 1] * z[i-3] - ( k[int((i+1)/2) - 1] + k[int((i+1)/2) - 1] ) * z[i-1] + k[int((i+1)/2) - 1] * z[i+1] ) for i in range(2,len(z)-2) ]            


def gen4LatticeDiffEqData(w, n):
    training_predictors = [ [ genNum(w),genNum(w),genNum(w),genNum(w),genNum(w),genNum(w),genNum(w),genNum(w) ] for i in range(n) ]
    training_labels = [ (np.asarray(NLatticeDerivatives(0,training_predictors[i]))) for i in range(n) ]
    
    interpolation_predictors = [[genNum(w),genNum(w),genNum(w),genNum(w),genNum(w),genNum(w),genNum(w),genNum(w)] for i in range(n)]
    interpolation_labels = [ (np.asarray(NLatticeDerivatives(0,interpolation_predictors[i]))) for i in range(n) ]
    
    extrapolation_near_predictors = [ [genNum(w/4)+genSign()*(5*w/4),genNum(w/4)+genSign()*(5*w/4),genNum(w/4)+genSign()*(5*w/4),genNum(w/4)+genSign()*(5*w/4),genNum(w/4)+genSign()*(5*w/4),genNum(w/4)+genSign()*(5*w/4),genNum(w/4)+genSign()*(5*w/4),genNum(w/4)+genSign()*(5*w/4)] for i in range(n)]
    extrapolation_near_labels = [ (np.asarray(NLatticeDerivatives(0,extrapolation_near_predictors[i]))) for i in range(n) ]
    extrapolation_far_predictors = [ [genNum(w/2)+genSign()*(3*w/2),genNum(w/2)+genSign()*(3*w/2),genNum(w/2)+genSign()*(3*w/2),genNum(w/2)+genSign()*(3*w/2),genNum(w/2)+genSign()*(3*w/2),genNum(w/2)+genSign()*(3*w/2),genNum(w/2)+genSign()*(3*w/2),genNum(w/2)+genSign()*(3*w/2)] for i in range(n)]
    extrapolation_far_labels = [ (np.asarray(NLatticeDerivatives(0,extrapolation_far_predictors[i]))) for i in range(n) ]
    all_data = [training_predictors, training_labels, interpolation_predictors, interpolation_labels, extrapolation_near_predictors, extrapolation_near_labels, extrapolation_far_predictors, extrapolation_far_labels]
    for i in range(len(all_data)):
        all_data[i] = np.asarray(all_data[i])
    return all_data

def divisionFunction(x):
    return np.sin(np.pi * x[0]) / (x[1]**2+1)

def genDivisionData(w,n):
    training_predictors = [ [ genNum(w),genNum(w) ] for i in range(n) ]
    training_labels = [ (np.asarray(divisionFunction(training_predictors[i]))) for i in range(n) ]
    
    interpolation_predictors = [[genNum(w),genNum(w)] for i in range(n)]
    interpolation_labels = [ (np.asarray(divisionFunction(interpolation_predictors[i]))) for i in range(n) ]
    
    extrapolation_near_predictors = [ [genNum(w/4)+genSign()*(5*w/4),genNum(w/4)+genSign()*(5*w/4)] for i in range(n)]
    extrapolation_near_labels = [ (np.asarray(divisionFunction(extrapolation_near_predictors[i]))) for i in range(n) ]
    extrapolation_far_predictors = [ [genNum(w/2)+genSign()*(3*w/2),genNum(w/2)+genSign()*(3*w/2)] for i in range(n)]
    extrapolation_far_labels = [ (np.asarray(divisionFunction(extrapolation_far_predictors[i]))) for i in range(n) ]
    all_data = [training_predictors, training_labels, interpolation_predictors, interpolation_labels, extrapolation_near_predictors, extrapolation_near_labels, extrapolation_far_predictors, extrapolation_far_labels]
    for i in range(len(all_data)):
        all_data[i] = np.asarray(all_data[i])
    return all_data