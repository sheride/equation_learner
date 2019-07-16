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


def odeSolve(model, odeFunction, initialCond, timeSpan, step):
    actualSol = solve_ivp(odeFunction, timeSpan, initialCond, first_step=step,
                          max_step=step)
    modelSol = solve_ivp(model.odecompat, timeSpan, initialCond, 
                         first_step=step, max_step=step)
    return [actualSol, modelSol]


def diffPlot(actualSol, modelSol, figsize=(10,10), name='Difference Plot'):
    newTrue = 0
    newModel = 0

    if actualSol.y[0].shape[0] > modelSol.y[0].shape[0]:
        newTrue = actualSol.y[0][0:modelSol.y[0].shape[0]]
        newModel = modelSol.y[0]
        t = modelSol.t
    else:
        newModel = modelSol.y[0][0:actualSol.y[0].shape[0]]
        newTrue = actualSol.y[0]
        t = actualSol.t

    diff = np.abs(newTrue - newModel)

    font = {'family': 'normal', 'weight': 'bold', 'size': 22}
    plt.figure(figsize=figsize)
    plt.suptitle('Title', **font)
    plt.xlabel('X-Axis', **font)
    plt.ylabel('Y-Axis', **font)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    line, = plt.plot(t, diff)
    plt.legend((line), ("Model 1"))
    plt.savefig(name+'.png')


def make2DMovie(actualSolCoords, modelSolCoords, xmin=-3, xmax=3, ymin=-3, 
                ymax=3, figSize=(10,10), lineWidth=4, 
                name='Model-Reality Comparison'):
    actualT, actualSolX, actualSolY = actualSolCoords
    modelT, modelSolX, modelSolY = modelSolCoords
    movie = pyvie.Movie(name, framerate=20, file_type='.png', 
                        movie_type = '.avi')
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
