#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:50:40 2019

@author: elijahsheridan
"""

import numpy as np
from scipy.integrate import solve_ivp
import pyvie
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def window(size):
    """
    Helper function for plotting moving time averages of a dataset. Not mine:
    taken from mtadd at
    stackoverflow.com/questions/15959819/time-series-averaging-in-numpy-python
    """

    return np.ones(size)/float(size)


def odeSolve(models, odeFunction, initialCond, timeSpan, step):
    """
    Applies scipy.integrate.solve_ivp to a goal function and a list of trained
    models for comparison/visualization purposes

    # Arguments
        models: list of trained EQL/EQL-div models
        odeFunction: goal function
        initialCond: list of size models[i].inputSize defining an initial
            condition to begin integrating from
        timeSpan: list with two elements (most likely with the first being 0)
            indicating the range of times to integrate over
        step: spacing between sampled functional values for goal/learned
            functions (note: NOT the steps that solve_ivp uses)

    # Returns
        A list containing solve_ivp output for the goal function and another
        list containing solve_ivp output for each model's learned function
    """

    t_eval = [timeSpan[0] + i * step for i in
              range(int((timeSpan[1] - timeSpan[0])/step))]
    actualSol = solve_ivp(odeFunction, timeSpan, initialCond, t_eval=t_eval)
    modelsSol = [0 for i in range(len(models))]
    for i, _ in enumerate(models):
        modelsSol[i] = solve_ivp(models[i].odecompat, timeSpan, initialCond,
                                 t_eval=t_eval)
    return [actualSol, modelsSol]


def dpdiffPlot(actualSol, modelSol, figsize=(10, 10), ymax=3,
               name='Difference Plot', names=None, title='',
               xlabel='X-Axis', ylabel='Y-Axis', binSize=50, settings=dict()):
    """
    Plots (moving time average of) error over time for trajectories of a double
    pendulum generated via an EQL/EQL-div model being entered into odeSolve.
    Error is Euclidean distance between actual and predicted second mass bob
    position.

    # Arguments
        actualSol, modelSol: outputs of scipy.integrate.solve_ivp, precisely
            what is returned by odeSolve
        figSize: dimensions of plot
        ymax: maximum y-axis (error) value shown on plot
        name: file name for saved plot
        names: tuple of strings, names for lines on plot for pyplot legend
        title: title of plot
        xlabel, ylabel: labels for axes
        binSize: bin size for moving time averages of error
        settings: pyplot rc_context settings
    """

    n = len(modelSol)

    diff = [0 for i in range(n)]
    lines = [0 for i in range(n)]
    if names is None:
        names = tuple('Model ' + str(i+1) for i in range(n))

    actualX = np.sin(actualSol.y[0]) + np.sin(actualSol.y[2])
    actualY = -np.cos(actualSol.y[0]) - np.cos(actualSol.y[2])
    actualCoord = np.asarray([actualX, actualY])
    for i in range(n):
        modelX = np.sin(modelSol[i].y[0]) + np.sin(modelSol[i].y[2])
        modelY = -np.cos(modelSol[i].y[0]) - np.cos(modelSol[i].y[2])
        modelCoord = np.asarray([modelX, modelY])
        diff[i] = np.linalg.norm(np.transpose(actualCoord)
                                 - np.transpose(modelCoord), axis=1)

    settings['figure.figsize'] = figsize
    with plt.rc_context(settings):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(0, actualSol.t[len(actualSol.t) - binSize])
        plt.ylim(0, ymax)

        for i in range(n):
            lines[i], = plt.plot(
                    actualSol.t[:-binSize],
                    np.convolve(diff[i], window(binSize), 'same')[:-binSize],
                    linewidth=10)

        lines = tuple(lines)
        plt.legend(lines, names, loc=2)
        plt.savefig(name+'.png', bbox_inches='tight')


def linear(x, m, b):
    """
    Helper function to be used with scipy.optimize.curve_fit in order to find
    energy drift in a learned set of physically-interpretable ODEs
    """

    return m * x + b


def getEnergyDriftAndFluc(modelSol, energyFunc):
    """
    Returns the energy drift and energy fluctuation learned set of
    physically-interpretable ODEs

    # Arguments
        modelSol: scipy.integrate.solve_ivp output from learned functions of an
            EQL/EQL-div model
        energyFunc: a python function which takes in a phase space vector
            corresponding to the given physical system and outputs the energy
            corresponding to that vector

    # Returns
        List containing energy drift, energy fluctuation
    """

    timeseries = [[modelSol.y[i][t] for i in range(len(modelSol.y))]
                  for t in range(len(modelSol.t))]
    energies = np.asarray([energyFunc(timeseries[t])
                           for t in range(len(timeseries))])
    energiesScaled = (energies - energies[0]) / energies[0]
    fit = curve_fit(linear, modelSol.t, energiesScaled)
    fluc = np.sqrt(np.sum(np.square(energies - energies[0])))
    return [fit[0][0], fluc]


def plotDriftAndFluc(modelSols, energyFunc, title='', xlabel='xaxis',
                     ylabel=['Energy Drift', 'Energy Fluctuation'], names=None,
                     name='Model', figsize=(10, 10), save=False,
                     settings=dict()):
    """
    Plots energy drift, energy fluctuation for a list of trained models.

    # Arguments
        modelSol: output of scipy.integrate.solve_ivp, precisely second item in
            list returned by odeSolve
        energyFunc: function accepting phase space vectors and returning the
            energy associated with each phase space vector
        title: title of plot
        xlabel, ylabel: labels for axes
        names: tuple of strings, names for lines on plot for pyplot legend
        name: file name for saved plot
        figSize: dimensions of plot
        save: boolean, True if plot is to be saved as a .png file
        settings: pyplot rc_context settings
    """

    msuGray = (153/255, 162/255, 162/255)
    msuGreen = (24/255, 69/255, 59/255)
    n = len(modelSols)
    if names is None:
        names = tuple('Model ' + str(i+1) for i in range(n))

    settings['figure.figsize'] = figsize
    with plt.rc_context(settings):

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set_title(title, color=msuGreen)
        ax1.set_xlabel(xlabel)
        ax1.set_xmargin(0.002)
        ax1.set_ylabel(ylabel[0], color=msuGreen)
        ax1.tick_params(axis='y', colors=msuGreen)
        ax2.set_ylabel(ylabel[1], color=msuGray)
        ax2.tick_params(axis='y', colors=msuGray)

        drifts = [0 for i in range(n)]
        flucs = [0 for i in range(n)]
        orig = [i for i in range(n)]

        for i in range(n):
            drifts[i], flucs[i] = getEnergyDriftAndFluc(modelSols[i],
                                                        energyFunc)

        ax1.scatter(orig, drifts, s=5000, c=[msuGreen], marker='s')
        ax2.scatter(orig, flucs, s=5000, c=[msuGray], marker='^')
        plt.xticks(orig, names)

        if save:
            plt.savefig(name+'.png', bbox_inches='tight')
