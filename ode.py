#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:50:40 2019

@author: elijahsheridan
"""

from equation_learner import data as d
import numpy as np
from scipy.integrate import solve_ivp
import pyvie
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# NOT MINE, TAKEN FROM:
# stackoverflow.com/questions/15959819/time-series-averaging-in-numpy-python
def window(size):
    return np.ones(size)/float(size)

def odeSolve(models, odeFunction, initialCond, timeSpan, step):
    t_eval = [timeSpan[0] + i * step for i in
              range(int((timeSpan[1] - timeSpan[0])/step))]
    actualSol = solve_ivp(odeFunction, timeSpan, initialCond, t_eval=t_eval)
    modelsSol = [0 for i in range(len(models))]
    for i in range(len(models)):
        modelsSol[i] = solve_ivp(models[i].odecompat, timeSpan, initialCond,
                                 t_eval=t_eval)
    return [actualSol, modelsSol]


def diffPlot(actualSol, modelSol, figsize=(10, 10), ymax=3,
             name='Difference Plot', names=None, title='',
             xlabel='X-Axis', ylabel='Y-Axis'):
    n = len(modelSol)

    diff = [0 for i in range(n)]
    lines = [0 for i in range(n)]
    if names is None:
        names = tuple('Model ' + str(i+1) for i in range(n))

    # Good for stable (pi/4, etc) for simulations lasting 100s of sec
    # binSize = 250

    # Good for chaotic, shorter simulations
    binSize = 50

    actualX = np.sin(actualSol.y[0]) + np.sin(actualSol.y[2])
    actualY = -np.cos(actualSol.y[0]) - np.cos(actualSol.y[2])
    actualCoord = np.asarray([actualX, actualY])
    for i in range(n):
        modelX = np.sin(modelSol[i].y[0]) + np.sin(modelSol[i].y[2])
        modelY = -np.cos(modelSol[i].y[0]) - np.cos(modelSol[i].y[2])
        modelCoord = np.asarray([modelX, modelY])
        diff[i] = np.linalg.norm(np.transpose(actualCoord)
                                 - np.transpose(modelCoord), axis=1)

    msuGray = (153/255, 162/255, 162/255)
    msuGreen = (24/255, 69/255, 59/255)
    msuOrange = (240/255, 133/255, 33/255)
    msuPurple = (110/255, 0, 95/255)
    msuCyan = (0, 129/255, 131/255)
    colors = [msuGray, msuOrange, msuPurple, msuCyan,
              'red', 'pink', 'blue', 'green']

    titlefont = {'family': 'sans-serif', 'weight': 'bold', 'size': 72,
                 'color': msuGreen}
    labelfont = {'family': 'sans-serif', 'weight': 'bold', 'size': 48,
                 'color': msuGreen}
    tickfont = {'size': 24, 'color': msuGreen}

    with plt.rc_context({'axes.linewidth': 6, 'axes.edgecolor': msuGreen,
                         'xtick.color': msuGreen, 'ytick.color': msuGreen,
                         'axes.labelpad': 20, 'xtick.major.size': 20,
                         'xtick.minor.size': 20, 'xtick.major.width': 5,
                         'xtick.minor.width': 5, 'ytick.major.size': 20,
                         'ytick.minor.size': 20, 'ytick.major.width': 5,
                         'ytick.minor.width': 5, 'xtick.labelsize': 48,
                         'ytick.labelsize': 48, 'axes.titlepad': 25,
                         'figure.figsize': figsize, 'legend.fontsize': 36,
                         'axes.labelweight': 'bold',
                         'axes.titleweight': 'bold',
                         'font.family': 'sans-serif', 'font.weight': 'bold'}):
        plt.title(title, **titlefont)
        plt.xlabel(xlabel, **labelfont)
        plt.ylabel(ylabel, **labelfont)
        plt.xticks(**tickfont)
        plt.yticks(**tickfont)
        plt.xlim(0, actualSol.t[len(actualSol.t) - binSize])
        plt.ylim(0, ymax)

        for i in range(n):
            lines[i], = plt.plot(
                    actualSol.t[:-binSize],
                    np.convolve(diff[i],window(binSize),'same')[:-binSize],
                    color=colors[i+1], linewidth=10)

        lines = tuple(lines)
        plt.legend(lines, names, loc=2)
        plt.savefig(name+'.png', bbox_inches='tight')


def make2DMovie(actualSolCoords, modelSolCoords, xmin=-3, xmax=3, ymin=-3,
                ymax=3, figSize=(10, 10), lineWidth=4,
                name='Model-Reality Comparison'):
    actualT, actualSolX, actualSolY = actualSolCoords
    modelT, modelSolX, modelSolY = modelSolCoords
    movie = pyvie.Movie(name, framerate=20, file_type='.png',
                        movie_type='.avi')
    plt.figure(figsize=figSize)
    for t in range(min(len(actualT), len(modelT))):
        plt.clf()
        plt.scatter(actualSolX[t], actualSolY[t], linewidth=lineWidth, s=1000)
        plt.scatter(modelSolX[t], modelSolY[t], linewidth=lineWidth, s=1000)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.grid(alpha=.5)
        plt.title(name, fontsize=22)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        movie.gather()
    movie.finalize()

def linear(x, m, b):
    return m * x + b

def getDPEnergyDriftAndFluc(modelSol):
    timeseries = [[modelSol.y[i][t] for i in range(4)]
        for t in range(len(modelSol.t))]
    energies = np.asarray([d.doublePendulumEnergy(timeseries[t])
        for t in range(len(timeseries))])
    energiesScaled = (energies - energies[0]) / energies[0]
#    plt.plot(modelSol.t, energiesScaled)
    fit = curve_fit(linear, modelSol.t, energiesScaled)
    fluc = np.sqrt(np.sum(np.square(energies - energies[0])))
    return [fit[0][0], fluc]


def plotDriftAndFluc(modelSols, title='', xlabel='xaxis',
                     ylabel=['yaxis', 'yaxis'], names=None, name='Model',
                     figsize=(10,10), save=False):
    msuGray = (153/255, 162/255, 162/255)
    msuGreen = (24/255, 69/255, 59/255)
    n = len(modelSols)
    if names is None:
        names = tuple('Model ' + str(i+1) for i in range(n))

    with plt.rc_context({'axes.linewidth': 6, 'axes.edgecolor': msuGreen,
                         'xtick.color': msuGreen, 'ytick.color': msuGreen,
                         'axes.labelpad': 20, 'xtick.major.size': 20,
                         'xtick.minor.size': 20, 'xtick.major.width': 5,
                         'xtick.minor.width': 5, 'ytick.major.size': 20,
                         'ytick.minor.size': 20, 'ytick.major.width': 5,
                         'ytick.minor.width': 5, 'xtick.labelsize': 48,
                         'ytick.labelsize': 48, 'axes.titlepad': 25,
                         'figure.figsize': figsize,
                         'axes.labelcolor': msuGreen,
                         'font.family': 'sans-serif', 'font.weight': 'bold',
                         'axes.titlesize': 72, 'axes.labelsize': 48,
                         'axes.labelweight': 'bold',
                         'axes.titleweight': 'bold', 'legend.fontsize': 36}):

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
            drifts[i], flucs[i] = getDPEnergyDriftAndFluc(modelSols[i])

        ax1.scatter(orig, drifts, s=5000, c=[msuGreen], marker='s')
        ax2.scatter(orig, flucs, s=5000, c=[msuGray], marker='^')
        plt.xticks(orig, names)

        if save == True:
            plt.savefig(name+'.png', bbox_inches='tight')
