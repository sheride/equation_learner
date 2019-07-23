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
             name='Difference Plot', names=None, title='Title',
             xlabel='X-Axis', ylabel='Y-Axis'):
    n = len(modelSol)

    diff = [0 for i in range(n)]
    lines = [0 for i in range(n)]
    if names is None:
        names = tuple('Model ' + str(i+1) for i in range(n))

    # Good for stable (pi/4, etc) for simulations lasting 100s of sec
    # binSize = 250

    binSize = 50

    actualX = np.sin(actualSol.y[0]) + np.sin(actualSol.y[2])
    actualY = -np.cos(actualSol.y[0]) - np.cos(actualSol.y[2])
    actualCoord = np.asarray([actualX, actualY])
    for i in range(n):
#        diff[i] = np.linalg.norm(
#                np.transpose(actualSol.y) - np.transpose(modelSol[i].y),
#                axis=1)
        modelX = np.sin(modelSol[i].y[0]) + np.sin(modelSol[i].y[2])
        modelY = -np.cos(modelSol[i].y[0]) - np.cos(modelSol[i].y[2])
        modelCoord = np.asarray([modelX, modelY])
        diff[i] = np.linalg.norm(np.transpose(actualCoord) - np.transpose(modelCoord), axis=1)

    msuGray = (153/255, 162/255, 162/255)
    msuGreen = (24/255, 69/255, 59/255)
    # msuYellow = (209/255, 202/255, 63/255)
    msuOrange = (240/255, 133/255, 33/255)
    msuPurple = (110/255, 0, 95/255)
    # msuBlue = (144/255, 154/255, 184/255)
    # msuTan = (232/255, 217/255, 181/255)
    msuCyan = (0, 129/255, 131/255)
    colors = [msuGray, msuCyan, msuOrange, msuPurple,
              'red', 'pink', 'blue', 'green']

    titlefont = {'family': 'sans-serif', 'weight': 'bold', 'size': 72, 'color': msuGreen}
    labelfont = {'family': 'sans-serif', 'weight': 'bold', 'size': 48, 'color': msuGreen}
    tickfont = {'size': 24, 'color': msuGreen}

    plt.figure(figsize=figsize)
    axs = plt.gca()
    axs.spines['top'].set_linewidth(4)
    axs.spines['right'].set_linewidth(4)
    axs.spines['bottom'].set_linewidth(4)
    axs.spines['left'].set_linewidth(4)
    axs.spines['top'].set_color(msuGreen)
    axs.spines['right'].set_color(msuGreen)
    axs.spines['bottom'].set_color(msuGreen)
    axs.spines['left'].set_color(msuGreen)
    plt.title(title, **titlefont)
    plt.xlabel(xlabel, labelpad=20, **labelfont)
    plt.ylabel(ylabel, labelpad=20, **labelfont)
    plt.xticks(**tickfont)
    plt.yticks(**tickfont)
    plt.xlim(0, actualSol.t[len(actualSol.t)-binSize])
    plt.ylim(0, ymax)
    plt.tick_params(axis='both', which='major', labelsize=30, color=msuGreen)
    axs.tick_params(length=15, width=4, which='both', color=msuGreen)
    for i in range(n):
#        cumsum = np.cumsum(np.insert(diff[i], 0, 0))
#        movavg = (cumsum[binSize:] - cumsum[:-binSize]) / binSize
        lines[i], = plt.plot(actualSol.t[:-binSize], np.convolve(diff[i],window(binSize),'same')[:-binSize], color=colors[i+1], linewidth=10)
    lines = tuple(lines)
    plt.legend(lines, names, fontsize=36)
    plt.tick_params(axis='both', which='major', labelsize=36, width=2,
                    length=15)
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
    fluc = np.sum(np.square(energies - energies[0]))
    return [fit[0][0], fluc]


def plotDriftAndFluc(modelSols, title='title', xlabel='xaxis', ylabel=['yaxis', 'yaxis'], names=None, name='Model', figsize=(10,10), save=False):
    msuGray = (153/255, 162/255, 162/255)
    msuGreen = (24/255, 69/255, 59/255)
    # msuYellow = (209/255, 202/255, 63/255)
    msuOrange = (240/255, 133/255, 33/255)
    msuPurple = (110/255, 0, 95/255)
    # msuBlue = (144/255, 154/255, 184/255)
    # msuTan = (232/255, 217/255, 181/255)
    msuCyan = (0, 129/255, 131/255)
    colors = [msuGray, msuCyan, msuOrange, msuPurple,
              'red', 'pink', 'blue', 'green']

    titlefont = {'family': 'sans-serif', 'weight': 'bold', 'size': 72, 'color': msuGreen}
    labelfont = {'family': 'sans-serif', 'weight': 'bold', 'size': 48, 'color': msuGreen}
    tickfont = {'size': 24, 'color': msuGreen}

    n = len(modelSols)
    if names is None:
        names = tuple('Model ' + str(i+1) for i in range(n))

    with plt.rc_context({'axes.linewidth': 4, 'axes.edgecolor': msuGreen,
                         'xtick.color': msuGreen, 'ytick.color': msuGreen,
                         'axes.labelpad': 20, 'xtick.major.size': 15,
                         'xtick.minor.size': 15, 'xtick.major.width': 2,
                         'xtick.minor.width': 2, 'ytick.major.size': 15,
                         'ytick.minor.size': 15, 'ytick.major.width': 2,
                         'ytick.minor.width': 2, 'xtick.labelsize': 30,
                         'ytick.labelsize': 30, 'axes.titlepad': 25}):

        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()

        ax1.set_title(title, **titlefont)
        ax1.set_xlabel(xlabel, **labelfont)
        ax1.set_ylabel(ylabel[0], **labelfont)
        ax2.set_ylabel(ylabel[1], **labelfont)

        drifts = [0 for i in range(n)]
        flucs = [0 for i in range(n)]
        orig = [i for i in range(n)]

        for i in range(n):
            drifts[i], flucs[i] = getDPEnergyDriftAndFluc(modelSols[i])

        ax1.scatter(orig, drifts, s=2000, c=[msuGreen], marker='x')
        ax2.scatter(orig, flucs, s=2000, c=[msuGreen], marker='^')
        plt.xticks(orig, names, **tickfont)

#        ax1.set_ylim(-0.002, 0.002)
#        ax2.set_ylim(-0.002, 0.002)

        if save == True:
            plt.savefig(name+'.png', bbox_inches='tight')
