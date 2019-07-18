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


def odeSolve(models, odeFunction, initialCond, timeSpan, step):
    actualSol = solve_ivp(odeFunction, timeSpan, initialCond, first_step=step,
                          max_step=step)
    modelsSol = [0 for i in range(len(models))]
    for i in range(len(models)):
        modelsSol[i] = solve_ivp(models[i].odecompat, timeSpan, initialCond,
                                 first_step=step, max_step=step)
    return [actualSol, modelsSol]


def diffPlot(actualSol, modelSol, figsize=(10, 10), ymax=3,
             name='Difference Plot', names=None, title='Title',
             xlabel='X-Axis', ylabel='Y-Axis'):
    n = len(modelSol)

    diff = [0 for i in range(n)]
    lines = [0 for i in range(n)]
    if names is None:
        names = tuple('Model ' + str(i+1) for i in range(n))

    for i in range(n):
        diff[i] = np.linalg.norm(
                np.transpose(actualSol.y) - np.transpose(modelSol[i].y),
                axis=1)

    font = {'family': 'sans-serif', 'weight': 'bold', 'size': 48}
    plt.figure(figsize=figsize)
    plt.suptitle(title, **font)
    plt.xlabel(xlabel, **font)
    plt.ylabel(ylabel, **font)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(0, actualSol.t[len(actualSol.t)-1])
    plt.ylim(0, ymax)
    for i in range(n):
        lines[i], = plt.plot(actualSol.t, diff[i])
    lines = tuple(lines)
    plt.legend(lines, names, fontsize='xx-large')
    plt.savefig(name+'.png')


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
