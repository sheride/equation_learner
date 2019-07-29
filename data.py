#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:02:34 2019

@author: elijahsheridan
"""

import numpy as np
from scipy.integrate import solve_ivp as slv
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

"""
Helper Functions

"""


def genSign():
    """
    Returns either 1 or -1
    """

    return 1 if np.random.rand() < 0.5 else -1


def genNum(width):
    """
    Returns a number in the range [-width, width]
    """

    return np.random.rand() * 2 * width - width


def fixRadians(x):
    """
    Converts an angle to within the range [-pi, pi]
    """

    return (x % (2 * np.pi) if x % (2 * np.pi) < np.pi
            else (x % (2 * np.pi) - 2 * np.pi))


"""
Data Preparation

"""


def pipeline(model, x, rescale=True, stand=True, norm=True):
    pipeline = []
    if rescale:
        mm = MinMaxScaler(feature_range=(0, 1)).fit(x)
        x = mm.transform(x)
        pipeline.append(mm)

    if stand:
        ss = StandardScaler().fit(x)
        x = ss.transform(x)
        pipeline.append(ss)

    if norm:
        n = Normalizer().fit(x)
        x = n.transform(x)
        pipeline.append(n)

    model.setPipeline(pipeline)

    return x


"""
Single Pendulum Differential Equation Data

"""


def pendulumDerivatives(x):
    g = 9.8
    return [x[1]/g, -np.sin(x[0])]


# function for generating pendulum angle/ang. vel. data
# w is width of hypercube of sampled points for training data
# n is number of data points
def genPendulumDiffEqData(w, n):
    training_predictors = [[genNum(w) for j in range(2)] for i in range(n)]
    training_labels = [pendulumDerivatives(x) for x in training_predictors]
    interpolation_predictors = [
            [genNum(w) for j in range(2)] for i in range(n)]
    interpolation_labels = [
            pendulumDerivatives(x) for x in interpolation_predictors]
    extrapolation_near_predictors = [
            [genNum(w/4) + genSign() * (5 * w/4)
             for j in range(2)] for i in range(n)]
    extrapolation_near_labels = [
            pendulumDerivatives(x) for x in extrapolation_near_predictors]
    extrapolation_far_predictors = [
            [genNum(w/2) + genSign() * (3 * w/2)
             for j in range(2)] for i in range(n)]
    extrapolation_far_labels = [
            pendulumDerivatives(x) for x in extrapolation_far_predictors]
    all_data = np.asarray([training_predictors,
                           training_labels,
                           interpolation_predictors,
                           interpolation_labels,
                           extrapolation_near_predictors,
                           extrapolation_near_labels,
                           extrapolation_far_predictors,
                           extrapolation_far_labels])
    np.save('PendulumDiffEq_' + str(w) + '_' + str(n), all_data,
            allow_pickle=False)


"""
Double Pendulum Coordinate Data

"""


# function for ODE solver
def doublePendulumDerivativesSolver(t, x):
    g = 9.8
    return [x[1],
            (-1 * (x[1]**2) * np.sin(x[0] - x[2]) * np.cos(x[0] - x[2])
            + g * np.sin(x[2]) * np.cos(x[0] - x[2])
            + -1 * (x[3]**2) * np.sin(x[0] - x[2])
            + -1 * 2 * g * np.sin(x[0]))
            / (2 - ((np.cos(x[0] - x[2]))**2)),
            x[3],
            ((x[3]**2) * np.sin(x[0] - x[2]) * np.cos(x[0] - x[2])
            + g * 2 * np.sin(x[0]) * np.cos(x[0] - x[2])
            + 2 * x[1]**2 * np.sin(x[0] - x[2])
            + -1 * 2 * g * np.sin(x[2]))
            / (2 - (np.cos(x[0] - x[2]))**2)]


# given a pair of angles in a list x, returns x and y coordinates of each
# mass bob in a vector
def doublePendulumCoordinate(x):
    return [np.sin(x[0]),
            -np.cos(x[0]),
            np.sin(x[0]) + np.sin(x[1]),
            -np.cos(x[0]) - np.cos(x[1])]


# generates two double pendulum coordinate data sets using two trajectories
def genDoublePendulumCoordinateData():
    firstInput = [np.pi/2, 0, np.pi/2, 0]
    secondInput = [np.pi/4, 0, np.pi/4, 0]
    firstOutput = slv(doublePendulumDerivativesSolver, [0, 40], firstInput,
                      first_step=0.05, max_step=0.05)
    secondOutput = slv(doublePendulumDerivativesSolver, [0, 40], secondInput,
                       first_step=0.05, max_step=0.05)
    interpolation_predictors = [
            [fixRadians(firstOutput.y[0][i]), fixRadians(firstOutput.y[2][i])]
            for i in range(len(firstOutput.y[0]))]
    interpolation_labels = [
            doublePendulumCoordinate(x) for x in interpolation_predictors]
    extrapolation_predictors = [
            [fixRadians(secondOutput.y[0][i]),
             fixRadians(secondOutput.y[2][i])]
            for i in range(len(secondOutput.y[0]))]
    extrapolation_labels = [
            doublePendulumCoordinate(x) for x in extrapolation_predictors]
    all_predictors = [interpolation_predictors, extrapolation_predictors]
    all_labels = [interpolation_labels, extrapolation_labels]
    np.save('DoublePendulumCoordPredictors', all_predictors,
            allow_pickle=False)
    np.save('DoublePendulumCoordLabels', all_labels, allow_pickle=False)


"""
Arbitrary R^4 -> R Function Data

"""


# F-1 from paper
def Function1(x):
    return (1/3) * (
            np.sin(np.pi * x[0])
            + np.sin(2 * np.pi * x[1] + (np.pi/8))
            + x[1]
            - x[2]*x[3])


# F-2 from paper
def Function2(x):
    return (1/3) * (
            np.sin(np.pi * x[0])
            + x[1] * np.cos(2 * np.pi * x[0] + (np.pi/4))
            + x[2]
            - x[3]**2)


# F-3 from paper
def Function3(x):
    return (1/3) * (
            (1 + x[1]) * np.sin(np.pi * x[0])
            + x[1] * x[2] * x[3])


def genFunctionData(w, n, f):
    training_predictors = [[genNum(w) for j in range(4)] for i in range(n)]
    training_labels = [f(training_predictors[i]) for i in range(n)]
    interpolation_predictors = [
            [genNum(w) for j in range(4)] for i in range(int(n/2))]
    interpolation_labels = [
            f(interpolation_predictors[i]) for i in range(int(n/2))]
    extrapolation_near_predictors = [
            [genNum(w/4) + genSign() * (5 * w/4) for j in range(4)]
            for i in range(int(n/2))]
    extrapolation_near_labels = [
            f(extrapolation_near_predictors[i]) for i in range(int(n/2))]
    extrapolation_far_predictors = [
            [genNum(w/2) + genSign() * (3 * w/2) for j in range(4)]
            for i in range(int(n/2))]
    extrapolation_far_labels = [
            f(extrapolation_far_predictors[i]) for i in range(int(n/2))]
    all_predictors = [training_predictors, interpolation_predictors,
                      extrapolation_near_predictors,
                      extrapolation_far_predictors]
    all_labels = [training_labels, interpolation_labels,
                  extrapolation_near_labels, extrapolation_far_labels]
    np.save('FunctionDataPredictors_' + str(w) + '_' + str(n), all_predictors,
            allow_pickle=False)
    np.save('FunctionDataLabels_' + str(w) + '_' + str(n), all_labels,
            allow_pickle=False)


"""
Double Pendulum Differential Equation Data

"""


def doublePendulumDerivatives(x):
    g = 9.8
    return [x[1],
            (-1 * (x[1]**2) * np.sin(x[0] - x[2]) * np.cos(x[0] - x[2])
            + g * np.sin(x[2]) * np.cos(x[0] - x[2])
            + -1 * (x[3]**2) * np.sin(x[0] - x[2])
            + -1 * 2 * g * np.sin(x[0]))
            / ((2 - ((np.cos(x[0] - x[2]))**2))),
            x[3],
            ((x[3]**2) * np.sin(x[0] - x[2]) * np.cos(x[0] - x[2])
            + g * 2 * np.sin(x[0]) * np.cos(x[0] - x[2])
            + 2 * x[1]**2 * np.sin(x[0] - x[2])
            + -1 * 2 * g * np.sin(x[2]))
            / ((2 - (np.cos(x[0] - x[2]))**2))]


def genDoublePendulumDiffEqData(w, n):
    training_predictors = [[genNum(w) for j in range(4)] for i in range(n)]
    training_labels = [
            doublePendulumDerivatives(training_predictors[i])
            for i in range(n)]
    interpolation_predictors = [
            [genNum(w) for j in range(4)] for i in range(n)]
    interpolation_labels = [
            doublePendulumDerivatives(interpolation_predictors[i])
            for i in range(n)]
    extrapolation_near_predictors = [
            [genNum(w/4) + genSign() * (5 * w/4) for j in range(4)]
            for i in range(n)]
    extrapolation_near_labels = [
            doublePendulumDerivatives(extrapolation_near_predictors[i])
            for i in range(n)]
    extrapolation_far_predictors = [
            [genNum(w/2)+genSign()*(3*w/2) for j in range(4)]
            for i in range(n)]
    extrapolation_far_labels = [
            doublePendulumDerivatives(extrapolation_far_predictors[i])
            for i in range(n)]
    all_data = np.asarray([training_predictors,
                           training_labels,
                           interpolation_predictors,
                           interpolation_labels,
                           extrapolation_near_predictors,
                           extrapolation_near_labels,
                           extrapolation_far_predictors,
                           extrapolation_far_labels])
    np.save('DoublePendulumDiffEq_' + str(w)[:1] + '_' + str(n), all_data,
            allow_pickle=False)


"""
N-Lattice Differential Equation Data

"""

N = 4  # number of masses
k = [50] * (N + 1)  # spring constants
m = [1] * N  # masses


# for x, even indices pos, odd inices vel
def NLatticeDerivativesSolver(t, x):
    z = [0, 0]
    z.extend(x)
    z.extend([0, 0])
    return [z[i+1] if i % 2 == 0
            else ((.1 / m[int((i-1)/2) - 1]) * (
                    k[int((i-1)/2) - 1] * z[i-3]
                    - (k[int((i + 1)/2) - 1] + k[int((i + 1)/2) - 1]) * z[i-1]
                    + k[int((i + 1)/2) - 1] * z[i+1]))
            for i in range(2, len(z)-2)]


def NLatticeDerivatives(x):
    z = [0, 0]
    z.extend(x)
    z.extend([0, 0])
    return [z[i+1] if i % 2 == 0
            else ((.1 / m[int((i-1)/2) - 1]) * (
                    k[int((i-1)/2) - 1] * z[i-3]
                    - (k[int((i + 1)/2) - 1] + k[int((i + 1)/2) - 1]) * z[i-1]
                    + k[int((i + 1)/2) - 1] * z[i+1]))
            for i in range(2, len(z)-2)]


def gen4LatticeDiffEqData(w, n):
    training_predictors = [
            [genNum(w) for j in range(2 * N)] for i in range(n)]
    training_labels = [
            NLatticeDerivatives(training_predictors[i]) for i in range(n)]
    interpolation_predictors = [
            [genNum(w) for j in range(2 * N)] for i in range(n)]
    interpolation_labels = [
            NLatticeDerivatives(interpolation_predictors[i])
            for i in range(n)]
    extrapolation_near_predictors = [
            [genNum(w/4) + genSign() * (5 * w/4) for j in range(2 * N)]
            for i in range(n)]
    extrapolation_near_labels = [
            NLatticeDerivatives(extrapolation_near_predictors[i])
            for i in range(n)]
    extrapolation_far_predictors = [
            [genNum(w/2)+genSign()*(3*w/2) for j in range(2 * N)]
            for i in range(n)]
    extrapolation_far_labels = [
            NLatticeDerivatives(extrapolation_far_predictors[i])
            for i in range(n)]
    all_predictors = [training_predictors, interpolation_predictors,
                      extrapolation_near_predictors,
                      extrapolation_far_predictors]
    all_labels = [training_labels, interpolation_labels,
                  extrapolation_near_labels, extrapolation_far_labels]
    np.save('NLatticeDiffEqPredictors_' + str(w) + '_' + str(n),
            all_predictors, allow_pickle=False)
    np.save('NLatticeDiffEqLabels_' + str(w) + '_' + str(n), all_labels,
            allow_pickle=False)


"""
Arbitrary R^2 -> R Division Function Data

"""


def divisionFunction(x):
    return np.sin(np.pi * x[0]) / (x[1]**2 + 1)


def genDivisionFunctionData(w, n):
    training_predictors = np.asarray([[genNum(w) for j in range(2)]
                                      for i in range(n)])
    training_labels = np.asarray([
            divisionFunction(training_predictors[i]) for i in range(n)])
    interpolation_predictors = np.asarray([
            [genNum(w) for j in range(2)] for i in range(n)])
    interpolation_labels = np.asarray([
            divisionFunction(interpolation_predictors[i])
            for i in range(n)])
    extrapolation_near_predictors = np.asarray([
            [genNum(w/4) + genSign() * (5 * w/4) for j in range(2)]
            for i in range(n)])
    extrapolation_near_labels = np.asarray([
            divisionFunction(extrapolation_near_predictors[i])
            for i in range(n)])
    extrapolation_far_predictors = np.asarray([
            [genNum(w/2) + genSign() * (3 * w/2) for j in range(2)]
            for i in range(n)])
    extrapolation_far_labels = np.asarray([
            divisionFunction(extrapolation_far_predictors[i])
            for i in range(n)])
    all_predictors = [training_predictors, interpolation_predictors,
                      extrapolation_near_predictors,
                      extrapolation_far_predictors]
    all_labels = [training_labels, interpolation_labels,
                  extrapolation_near_labels, extrapolation_far_labels]

    np.save('DivisionFunctionPredictors_' + str(w) + '_' + str(n),
            all_predictors, allow_pickle=False)
    np.save('DivisionFunctionLabels_' + str(w) + '_' + str(n), all_labels,
            allow_pickle=False)


"""
Double Pendulum Differential Equation with Energy Data

"""


def doublePendulumEnergy(x):
    g = 9.8
    return (x[1]**2
            + 0.5 * x[3]**2
            + x[1] * x[3] * np.cos(x[0] - x[2])
            - 2 * g * np.cos(x[0])
            - g * np.cos(x[2]))


def genDoublePendulumPointEnergy(w, func):
    point = [func(w) for i in range(4)]
    point.append(doublePendulumEnergy(point))
    return point


def genDoublePendulumDiffEqEnergyData(w, n):
    def ext_near(w):
        return genNum(w/4) + genSign() * (5*w/4)

    def ext_far(w):
        return genNum(w/2)+genSign()*(3*w/2)

    training_predictors = [
            genDoublePendulumPointEnergy(w, genNum) for i in range(n)]
    training_labels = [
            doublePendulumDerivatives(training_predictors[i])
            for i in range(n)]
    interpolation_predictors = [
            genDoublePendulumPointEnergy(w, genNum) for i in range(n)]
    interpolation_labels = [
            doublePendulumDerivatives(interpolation_predictors[i])
            for i in range(n)]
    extrapolation_near_predictors = [
            genDoublePendulumPointEnergy(w, ext_near)
            for i in range(n)]
    extrapolation_near_labels = [
            doublePendulumDerivatives(extrapolation_near_predictors[i])
            for i in range(n)]
    extrapolation_far_predictors = [
            genDoublePendulumPointEnergy(w, ext_far) for i in range(n)]
    extrapolation_far_labels = [
            doublePendulumDerivatives(extrapolation_far_predictors[i])
            for i in range(n)]
    all_predictors = [training_predictors, interpolation_predictors,
                      extrapolation_near_predictors,
                      extrapolation_far_predictors]
    all_labels = [training_labels, interpolation_labels,
                  extrapolation_near_labels, extrapolation_far_labels]
    np.save('DoublePendulumDiffEqEnergyPredictors_' + str(w) + '_' + str(n),
            all_predictors, allow_pickle=False)
    np.save('DoublePendulumDiffEqEnergyLabels_' + str(w) + '_' + str(n),
            all_labels, allow_pickle=False)


"""
Double Pendulum Differential Equation with Kinetic Energy, Potential Energy
Data

"""


def doublePendulumKE(x):
    return (x[1]**2 + 0.5 * x[3]**2 + x[1] * x[3] * np.cos(x[0] - x[2])) / 100


def doublePendulumPE(x):
    g = 9.8
    return (-2 * g * np.cos(x[0]) - g * np.cos(x[2])) / 100


def genDoublePendulumPointKEPE(w, func):
    point = [func(w) for i in range(4)]
    point.append(doublePendulumKE(point))
    point.append(doublePendulumPE(point))
    return point


def genDoublePendulumDiffEqKEPEData(w, n):
    def ext_near(w):
        return genNum(w/4) + genSign() * (5*w/4)

    def ext_far(w):
        return genNum(w/2)+genSign()*(3*w/2)

    training_predictors = [
            genDoublePendulumPointKEPE(w, genNum) for i in range(n)]
    training_labels = [
            doublePendulumDerivatives(training_predictors[i])
            for i in range(n)]
    interpolation_predictors = [
            genDoublePendulumPointKEPE(w, genNum) for i in range(n)]
    interpolation_labels = [
            doublePendulumDerivatives(interpolation_predictors[i])
            for i in range(n)]
    extrapolation_near_predictors = [
            genDoublePendulumPointKEPE(w, ext_near) for i in range(n)]
    extrapolation_near_labels = [
            doublePendulumDerivatives(extrapolation_near_predictors[i])
            for i in range(n)]
    extrapolation_far_predictors = [
            genDoublePendulumPointKEPE(w, ext_far) for i in range(n)]
    extrapolation_far_labels = [
            doublePendulumDerivatives(extrapolation_far_predictors[i])
            for i in range(n)]
    all_predictors = [training_predictors, interpolation_predictors,
                      extrapolation_near_predictors,
                      extrapolation_far_predictors]
    all_labels = [training_labels, interpolation_labels,
                  extrapolation_near_labels, extrapolation_far_labels]
    np.save('DoublePendulumDiffEqKEPE_' + str(w) + '_' + str(n),
            all_predictors, allow_pickle=True)
    np.save('DoublePendulumDiffEqKEPE_' + str(w) + '_' + str(n), all_labels,
            allow_pickle=True)


"""
Regularization Demonstration R -> R Function Data
"""


def regDem(x):
    return ((np.cos(11 * x) - 3 * x**2)
            / (np.sin(x) + 4))


def genRegFunctionData(w, n):
    training_predictors = [genNum(w) for i in range(n)]
    training_labels = [
            regDem(training_predictors[i]) for i in range(n)]
    interpolation_predictors = [genNum(w) for i in range(n)]
    interpolation_labels = [
            regDem(interpolation_predictors[i])
            for i in range(n)]
    extrapolation_near_predictors = [genNum(w/4) + genSign() * (5 * w/4)
                                     for i in range(n)]
    extrapolation_near_labels = [
            regDem(extrapolation_near_predictors[i])
            for i in range(n)]
    extrapolation_far_predictors = [genNum(w/2) + genSign() * (3 * w/2)
                                    for i in range(n)]
    extrapolation_far_labels = [
            regDem(extrapolation_far_predictors[i])
            for i in range(n)]
    all_predictors = [training_predictors, interpolation_predictors,
                      extrapolation_near_predictors,
                      extrapolation_far_predictors]
    all_labels = [training_labels, interpolation_labels,
                  extrapolation_near_labels, extrapolation_far_labels]
    np.save('RegDemFunction_' + str(w) + '_' + str(n), all_predictors,
            allow_pickle=False)
    np.save('RegDemFunction_' + str(w) + '_' + str(n), all_labels,
            allow_pickle=False)


"""
Double Pendulum Differential Equation Data (Generated by time-series for
initial conditions with same energy)
"""


# Energy E should fall roughly [-3g, 3g]
# n: number of datapoints
# c: number of configurations
# step: interval between recorded time-series points
def genDoublePendulumTimeseries(E, n, c, step):
    g = 9.8
    t_eval = np.linspace(0, int((n * step)/(2*c)), int(n/(2*c)))
    T = np.arccos(-E / (3*g))
    x0 = np.asarray([T, 0, T, 0])
    xs = [x0, -x0]

    for i in range(2, int(c/2)):
        xs.append([T/i,
                   np.sqrt(6/5) * np.sqrt(g * (np.cos(T/i) - np.cos(T))),
                   T/i,
                   np.sqrt(6/5) * np.sqrt(g * (np.cos(T/i) - np.cos(T)))])
        xs.append([T/i,
                   -np.sqrt(6/5) * np.sqrt(g * (np.cos(T/i) - np.cos(T))),
                   T/i,
                   -np.sqrt(6/5) * np.sqrt(g * (np.cos(T/i) - np.cos(T)))])

    sols = []
    for x in xs:
        sols.append(slv(doublePendulumDerivativesSolver,
                        [0, len(t_eval) * step],
                        x,
                        t_eval=t_eval).y)

    data = np.concatenate(tuple(sols), axis=1)

    labels = np.transpose(doublePendulumDerivatives(data))
    data = np.transpose(data)

    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)

    all_data = np.asarray([data, labels])

    np.save('DoublePendulumTimeseries_{}_{}_{}_{}'.format(str(E),
                                                          str(n),
                                                          str(c),
                                                          str(step)),
            all_data, allow_pickle=False)


def randomDPEnergyState(E, maxAngle, maxVel):
    x0 = [0, 0, 0, 0]

    while x0[3] == 0:
        g = 9.8
        sign = genSign()
        t1 = np.random.uniform(0.5 * sign * maxAngle, sign * maxAngle)
        sign = genSign()
        t2 = np.random.uniform(0.5 * sign * maxAngle, sign * maxAngle)
        w1 = np.random.uniform(0, maxVel/2)
        disc = (4 * E
                + 8 * g * np.cos(t1)
                + 4 * g * np.cos(t2)
                - 3 * w1**2
                + w1**2 * np.cos(2*t1 - 2*t2))
        if disc > 0:
            w2 = -w1*np.cos(t1 - t2) - np.sqrt(0.5 * disc)
        else:
            w2 = 0
        x0 = [t1, w1, t2, w2]

    return x0


def genDoublePendulumTimeseriesRandom(E, n, c, step):
    g = 9.8
    t_eval = np.linspace(0, int((n * step)/(2*c)), int(n/(2*c)))
    maxAngle = np.arccos(-E / (3*g))
    maxVel = np.sqrt((E + 3*g) / (5/2))
    xs = []
    for i in range(c):
        xs.append(randomDPEnergyState(E, maxAngle, maxVel))

    sols = []
    for x in xs:
        sols.append(slv(doublePendulumDerivativesSolver,
                        [0, len(t_eval) * step],
                        x,
                        t_eval=t_eval).y)

    data = np.concatenate(tuple(sols), axis=1)

    labels = np.transpose(doublePendulumDerivatives(data))
    data = np.transpose(data)

    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)

    all_data = np.asarray([data, labels])

    np.save('DoublePendulumTimeseriesRandom_{}_{}_{}_{}'.format(str(E),
                                                                str(n),
                                                                str(c),
                                                                str(step)),
            all_data, allow_pickle=False)


def genDoublePendulumConstEnergy(E, n):
    g = 9.8
    maxAngle = np.arccos(-E / (3*g))
    maxVel = np.sqrt((E + 3*g) / (5/2))
    data = []
    for i in range(n):
        data.append(randomDPEnergyState(E, maxAngle, maxVel))

    labels = np.transpose(doublePendulumDerivatives(np.transpose(data)))

    all_data = np.asarray([data, labels])

    np.save('DoublePendulumConstEnergys_{}_{}'.format(str(E), str(n)),
            all_data, allow_pickle=False)
