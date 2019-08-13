#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:02:34 2019

@author: elijahsheridan
"""

import numpy as np
from scipy.integrate import solve_ivp as slv
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import tensorflow as tf

"""
# References
        - [Learning Equations for Extrapolation and Control](
           https://arxiv.org/abs/1806.07259)
        - [Extrapolation and learning equations](
           https://arxiv.org/abs/1610.02995)
"""


def genSign():
    """Returns either 1 or -1"""
    return 1 if np.random.rand() < 0.5 else -1


def genNum(width):
    """Returns a number in the range [-width, width]"""
    return np.random.rand() * 2 * width - width


def fixRadians(x):
    """Converts an angle to within the range [-pi, pi]"""
    return (x % (2 * np.pi) if x % (2 * np.pi) < np.pi
            else (x % (2 * np.pi) - 2 * np.pi))


def pipeline(model, x, rescale=True, stand=True, norm=True):
    """Applies a scikit_learn pipeline for data preprocessing"""
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


def pendulumDerivatives(x):
    """Returns time derivative of pendulum phase space vector (divided by g)"""
    g = 9.8
    return [x[1], -g*np.sin(x[0])]


def pendulumDerivativesSolver(t, x):
    """Returns time derivative of pendulum phase space vector (divided by g)"""
    g = 9.8
    return [x[1], -g*np.sin(x[0])]


def genPendulumDiffEqData(w, n):
    """
    Saves 1 training and 3 testing data sets (each with n data points)
    corresponding to the E-L equations of motion of the single pendulum,
    sampled from different phase space hypercubes centered on origin, as
    described in [1].

    training_predictors, interpolation_predictors: [-w, w]^2
    extrapolation_near_predictors: [-3w/2, -3w/2]^2 (setminus) [-w, w]^2
    extrapolation_far_predictors: [-2w, -2w]^2 (setminus) [-w, -w]^2
    """

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


def genPendulumDiffEqTrajectories(n):
    smallAngle = [np.pi/8, 0]
    mediumAngle = [np.pi/2, 0]
    largeAngle = [np.pi, 0.1]

    step = 0.05
    t_eval = np.linspace(0, n * step, n)

    smallAngleSol = slv(pendulumDerivativesSolver, [0, len(t_eval) * step],
                        smallAngle, t_eval=t_eval)
    smallAnglePredictors = np.transpose(smallAngleSol.y)
    smallAngleLabels = np.transpose(pendulumDerivatives(smallAngleSol.y))

    mediumAngleSol = slv(pendulumDerivativesSolver, [0, len(t_eval) * step],
                         mediumAngle, t_eval=t_eval)
    mediumAnglePredictors = np.transpose(mediumAngleSol.y)
    mediumAngleLabels = np.transpose(pendulumDerivatives(mediumAngleSol.y))

    largeAngleSol = slv(pendulumDerivativesSolver, [0, len(t_eval) * step],
                        largeAngle, t_eval=t_eval)
    largeAnglePredictors = np.transpose(largeAngleSol.y)
    largeAngleLabels = np.transpose(pendulumDerivatives(largeAngleSol.y))

    print(smallAnglePredictors.shape, smallAngleLabels.shape)

    smallAngleData = [smallAnglePredictors, smallAngleLabels]
    mediumAngleData = [mediumAnglePredictors, mediumAngleLabels]
    largeAngleData = [largeAnglePredictors, largeAngleLabels]

    np.save('PendulumDiffEqSmallTrajectory_' + str(n), smallAngleData,
            allow_pickle=False)
    np.save('PendulumDiffEqMediumTrajectory_' + str(n), mediumAngleData,
            allow_pickle=False)
    np.save('PendulumDiffEqLargeTrajectory_' + str(n), largeAngleData,
            allow_pickle=False)


def doublePendulumDerivativesSolver(t, x):
    """
    Returns time derivative of double pendulum phase space vector, solve_ivp
    compatible
    """

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


def doublePendulumCoordinate(x):
    """
    Returns vector containing Cartesian coordinates [x1, y1, x2, y2]
    describing given double pendulum configuration.

    # Arguments
        x: vector in R^2 of form [theta1, theta2], configuration of double
        pendulum
    """

    return [np.sin(x[0]),
            -np.cos(x[0]),
            np.sin(x[0]) + np.sin(x[1]),
            -np.cos(x[0]) - np.cos(x[1])]


def genDoublePendulumCoordinateData():
    """
    Saves 1 training and 1 testing data sets (each with n data points)
    corresponding to the coordinate transformation between two angles (theta1,
    theta2) and Cartesian coordinates (x1, y1, x2, y2) using double pendulum
    trajectories as described in [1].

    Training set (interpolation) comes from low-energy trajectory (see
    firstInput for initial condition) and testing set (extrapolation) comes
    from higher energy trajectory (see secondInput).
    """

    firstInput = [np.pi/2, 0, np.pi/2, 0]
    secondInput = [np.pi/4, 0, np.pi/4, 0]
    firstOutput = slv(doublePendulumDerivativesSolver, [0, 40], firstInput,
                      first_step=0.05, max_step=0.05, rtol=1e-10, atol=1e-10)
    secondOutput = slv(doublePendulumDerivativesSolver, [0, 40], secondInput,
                       first_step=0.05, max_step=0.05, rtol=1e-10, atol=1e-10)
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


def Function1(x):
    """F-1 from [1]"""
    return (1/3) * (
            np.sin(np.pi * x[0])
            + np.sin(2 * np.pi * x[1] + (np.pi/8))
            + x[1]
            - x[2]*x[3])


def Function2(x):
    """F-2 from [1]"""
    return (1/3) * (
            np.sin(np.pi * x[0])
            + x[1] * np.cos(2 * np.pi * x[0] + (np.pi/4))
            + x[2]
            - x[3]**2)


def Function3(x):
    """F-3 from [1]"""
    return (1/3) * (
            (1 + x[1]) * np.sin(np.pi * x[0])
            + x[1] * x[2] * x[3])


def genFunctionData(w, n, f):
    """
    Saves 1 training and 3 testing data sets (each with n data points)
    corresponding to an arbitrary function f R^4 -> R, sampled from different
    hypercubes centered on the origin, as described in [1].

    training_predictors, interpolation_predictors: [-w, w]^4
    extrapolation_near_predictors: [-3w/2, -3w/2]^4 (setminus) [-w, w]^4
    extrapolation_far_predictors: [-2w, -2w]^4 (setminus) [-w, -w]^4
    """

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


def doublePendulumDerivatives(x):
    """Returns time derivative of double pendulum phase space vector"""
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
    """
    Saves 1 training and 3 testing data sets (each with n data points)
    corresponding to the E-L equations of motion of the double pendulum,
    sampled from different phase space hypercubes centered on the origin, a
    procedure described [1].

    training_predictors, interpolation_predictors: [-w, w]^4
    extrapolation_near_predictors: [-3w/2, -3w/2]^4 (setminus) [-w, w]^4
    extrapolation_far_predictors: [-2w, -2w]^4 (setminus) [-w, -w]^4
    """

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


def NLatticeDerivativesSolver(t, x, k=([50] * 5), m=([1] * 4)):
    """
    Returns time derivative of N-lattice phase space vector (x1, v1, ...),
    solve_ivp compatible
    """

    z = [0, 0]
    z.extend(x)
    z.extend([0, 0])
    return [z[i+1] if i % 2 == 0
            else ((.1 / m[int((i-1)/2) - 1]) * (
                    k[int((i-1)/2) - 1] * z[i-3]
                    - (k[int((i + 1)/2) - 1] + k[int((i + 1)/2) - 1]) * z[i-1]
                    + k[int((i + 1)/2) - 1] * z[i+1]))
            for i in range(2, len(z)-2)]


def NLatticeDerivatives(x, k=([50] * 5), m=([1] * 4)):
    """Returns time derivative of N-lattice phase space vector (x1, v1, ...)"""
    z = [0, 0]
    z.extend(x)
    z.extend([0, 0])
    return [z[i+1] if i % 2 == 0
            else ((.1 / m[int((i-1)/2) - 1]) * (
                    k[int((i-1)/2) - 1] * z[i-3]
                    - (k[int((i + 1)/2) - 1] + k[int((i + 1)/2) - 1]) * z[i-1]
                    + k[int((i + 1)/2) - 1] * z[i+1]))
            for i in range(2, len(z)-2)]


def genNLatticeDiffEqData(w, n, N=4):
    """
    Saves 1 training and 3 testing data sets (each with n data points)
    corresponding to the E-L equations of motion of the N-lattice pendulum,
    sampled from different phase space hypercubes centered on the origin, a
    procedure described [1] (defaults to N=4).

    training_predictors, interpolation_predictors: [-w, w]^N
    extrapolation_near_predictors: [-3w/2, -3w/2]^N (setminus) [-w, w]^N
    extrapolation_far_predictors: [-2w, -2w]^N (setminus) [-w, -w]^N
    """

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


def divisionFunction(x):
    """Eq. 11 in [2]"""
    return np.sin(np.pi * x[0]) / (x[1]**2 + 1)


def genDivisionFunctionData(w, n):
    """
    Saves 1 training and 3 testing data sets (each with n data points)
    corresponding to the divisionFunction, itself given in [2], sampled from
    different hypercubes centered on the origin, a procedure described [1]

    training_predictors, interpolation_predictors: [-w, w]^2
    extrapolation_near_predictors: [-3w/2, -3w/2]^2 (setminus) [-w, w]^2
    extrapolation_far_predictors: [-2w, -2w]^2 (setminus) [-w, -w]^2
    """

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


def doublePendulumEnergy(x):
    """
    Returns the Hamiltonian of a vector in the double pendulum phase space
    """

    g = 9.8
    return (x[1]**2
            + 0.5 * x[3]**2
            + x[1] * x[3] * np.cos(x[0] - x[2])
            - 2 * g * np.cos(x[0])
            - g * np.cos(x[2]))


def genDoublePendulumPointEnergy(w, func):
    """
    Uses func, w to generate elements of a double pendulum phase space vector,
    and appends that vector's associated Hamiltonian
    """

    point = [func(w) for i in range(4)]
    point.append(doublePendulumEnergy(point))
    return point


def genDoublePendulumDiffEqEnergyData(w, n):
    """
    Saves 1 training and 3 testing data sets (each with n data points)
    corresponding to the E-L equations of motion of the double pendulum,
    sampled from different phase space hypercubes centered on the origin, a
    procedure described [1]. Input is appended with its Hamiltonian value.

    training_predictors, interpolation_predictors: [-w, w]^4
    extrapolation_near_predictors: [-3w/2, -3w/2]^4 (setminus) [-w, w]^4
    extrapolation_far_predictors: [-2w, -2w]^4 (setminus) [-w, -w]^4
    """

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


def doublePendulumKE(x):
    """
    Returns the kinetic energy of a vector in the double pendulum phase space
    """

    return (x[1]**2 + 0.5 * x[3]**2 + x[1] * x[3] * np.cos(x[0] - x[2])) / 100


def doublePendulumPE(x):
    """
    Returns the potential energy of a vector in the double pendulum phase space
    """

    g = 9.8
    return (-2 * g * np.cos(x[0]) - g * np.cos(x[2])) / 100


def genDoublePendulumPointKEPE(w, func):
    """
    Uses func, w to generate elements of a double pendulum phase space vector,
    and appends that vector's associated kinetic energy, potential energy
    """

    point = [func(w) for i in range(4)]
    point.append(doublePendulumKE(point))
    point.append(doublePendulumPE(point))
    return point


def genDoublePendulumDiffEqKEPEData(w, n):
    """
    Saves 1 training and 3 testing data sets (each with n data points)
    corresponding to the E-L equations of motion of the double pendulum,
    sampled from different phase space hypercubes centered on the origin, a
    procedure described [1]. Input is appended with its kinetic energy and
    potential energy values.

    training_predictors, interpolation_predictors: [-w, w]^4
    extrapolation_near_predictors: [-3w/2, -3w/2]^4 (setminus) [-w, w]^4
    extrapolation_far_predictors: [-2w, -2w]^4 (setminus) [-w, -w]^4
    """

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
    np.save('DoublePendulumDiffEqKEPEPredictor_' + str(w) + '_' + str(n),
            all_predictors, allow_pickle=True)
    np.save('DoublePendulumDiffEqKEPELabel_' + str(w) + '_' + str(n),
            all_labels, allow_pickle=True)


def regDem(x):
    """
    Arbitrary function R -> R created to demonstrate effects of regularization
    for poster presentation at MSU Mid-SURE 2019
    """

    return ((np.cos(11 * x) - 3 * x**2)
            / (np.sin(x) + 4))


def genRegFunctionData(w, n):
    """
    Saves 1 training and 3 testing data sets (each with n data points)
    corresponding to the regDem function, sampled from different hypercubes
    centered on the origin, a procedure described [1]

    training_predictors, interpolation_predictors: [-w, w]^2
    extrapolation_near_predictors: [-3w/2, -3w/2]^2 (setminus) [-w, w]^2
    extrapolation_far_predictors: [-2w, -2w]^2 (setminus) [-w, -w]^2
    """

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
    np.save('RegDemFunctionPredictor_' + str(w) + '_' + str(n), all_predictors,
            allow_pickle=False)
    np.save('RegDemFunctionLabel_' + str(w) + '_' + str(n), all_labels,
            allow_pickle=False)


def genDoublePendulumTimeseries(E, n, c, step):
    """
    Saves a training dataset corresponding to the to the E-L equations of
    motion of the double pendulum, collected using numerical integration of
    a variety of trajectories, each corresponding to the same Hamiltonian
    value. All configurations are of the form theta1 = theta2 = T, omega1 =
    omega2 = W. If theta1 = theta2 = T_max -> omega1 = omega2 = 0, then
    utilized T values are (T_max, -T_max, T_max/2, -T_max/2, ..., T_max/(c/2))

    # Arguments
        E: Hamiltonian value for sampled trajectories, should fall in [-3g, 3g]
        n: number of datapoints
        c: number of configurations, should be even integer
        step: temporal spacing between sampled datapoints in trajectories (not
            the step size of the integrator)
    """

    g = 9.8
    t_eval = np.linspace(0, int((n * step)/(c)), int(n/(c)))
    T = np.arccos(-E / (3*g))
    x0 = np.asarray([T, 0, T, 0])
    xs = [x0, -x0]

    for i in range(2, int(c/2)+1):
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
                        [0, len(t_eval) * step], x, t_eval=t_eval,
                        rtol=1e-10, atol=1e-10).y,)

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
    """
    Generates a random initial configuration for the double pendulum with a
    given Hamiltonian value.

    # Arguments
        E: desired Hamiltonian value
        maxAngle: value such that Hamiltonian for (maxAngle, 0, maxAngle, 0) is
            E ( arccos(-E / 3g) )
        maxVel: value such that Hamiltonian for (0, maxVel, 0, maxVel) is E
            ( sqrt( (E + 3g) / (5/2) ) )
    """

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
    """
    Saves a training dataset corresponding to the to the E-L equations of
    motion of the double pendulum, collected using numerical integration of
    a variety of trajectories, each corresponding to the same Hamiltonian
    value. Configurations generated randomly using randomDPEnergyState.

    # Arguments
        E: Hamiltonian value for sampled trajectories
        n: number of datapoints
        c: number of configurations
        step: temporal spacing between sampled datapoints in trajectories (not
            the step size of the integrator)
    """

    g = 9.8
    t_eval = np.linspace(0, int((n * step)/c), int(n/c))
    maxAngle = np.arccos(-E / (3*g))
    maxVel = np.sqrt((E + 3*g) / (5/2))
    xs = []
    for i in range(c):
        xs.append(randomDPEnergyState(E, maxAngle, maxVel))

    sols = []
    for x in xs:
        sols.append(slv(doublePendulumDerivativesSolver,
                        [0, len(t_eval) * step], x, t_eval=t_eval, rtol=1e-10,
                        atol=1e-10).y)

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
    """
    Saves a training dataset with n datapoints corresponding to the to the E-L
    equations of motion of the double pendulum, collected using random sampling
    of subset of phase space corresponding to particular Hamiltonian value E
    """

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


def DPEnergyTF(x):
    """
    Analog to doublePendulumEnergy(), compatible with minibatches of TF/Keras
    tensors
    """

    g = 9.8
    return (tf.square(x[:, 1:2])
            + 0.5 * tf.square(x[:, 3:4])
            + x[:, 1:2] * x[:, 3:4] * tf.math.cos(x[:, 0:1] - x[:, 2:3])
            - 2 * g * tf.math.cos(x[:, 0:1])
            - g * tf.math.cos(x[:, 2:3]))


def genDPFeatureEngPoint(w, func):
    """
    Uses func, w to generate elements of a double pendulum phase space vector,
    and appends (theta1 - theta2, omega1^2, omega2^2) due to their frequent
    appearance in the double pendulum equations of motion.
    """

    point = [func(w) for i in range(4)]
    point.append(point[0] - point[2])
    point.append(point[1]**2)
    point.append(point[3]**2)
    return point


def genDoublePendulumDiffEqFeatureEng(w, n):
    """
    Saves 1 training and 3 testing data sets (each with n data points)
    corresponding to the E-L equations of motion of the double pendulum,
    sampled from different phase space hypercubes centered on the origin, a
    procedure described [1]. Input is appended with
    (theta1 - theta2, omega1^2, omega2^2) due to their frequent appearance in
    the equations of motion.

    training_predictors, interpolation_predictors: [-w, w]^4
    extrapolation_near_predictors: [-3w/2, -3w/2]^4 (setminus) [-w, w]^4
    extrapolation_far_predictors: [-2w, -2w]^4 (setminus) [-w, -w]^4
    """

    def ext_near(w):
        return genNum(w/4) + genSign() * (5*w/4)

    def ext_far(w):
        return genNum(w/2)+genSign()*(3*w/2)

    training_predictors = [
            genDPFeatureEngPoint(w, genNum) for i in range(n)]
    training_labels = [
            doublePendulumDerivatives(training_predictors[i])
            for i in range(n)]
    interpolation_predictors = [
            genDPFeatureEngPoint(w, genNum) for i in range(n)]
    interpolation_labels = [
            doublePendulumDerivatives(interpolation_predictors[i])
            for i in range(n)]
    extrapolation_near_predictors = [
            genDPFeatureEngPoint(w, ext_near) for i in range(n)]
    extrapolation_near_labels = [
            doublePendulumDerivatives(extrapolation_near_predictors[i])
            for i in range(n)]
    extrapolation_far_predictors = [
            genDPFeatureEngPoint(w, ext_far) for i in range(n)]
    extrapolation_far_labels = [
            doublePendulumDerivatives(extrapolation_far_predictors[i])
            for i in range(n)]
    all_predictors = [training_predictors, interpolation_predictors,
                      extrapolation_near_predictors,
                      extrapolation_far_predictors]
    all_labels = [training_labels, interpolation_labels,
                  extrapolation_near_labels, extrapolation_far_labels]
    np.save('DoublePendulumDiffEqFeatureEngPredictor_' + str(w) + '_' + str(n),
            all_predictors, allow_pickle=True)
    np.save('DoublePendulumDiffEqFeatureEngLabel_' + str(w) + '_' + str(n),
            all_labels, allow_pickle=True)
