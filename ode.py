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
    actualSol = solve_ivp(odeFunction, timeSpan, initialCond,
                              first_step=step, max_step=step)
    modelSol = solve_ivp(model.odecompat, timeSpan, initialCond,
                              first_step=step, max_step=step)
    return [actualSol, modelSol]

# FIG SIZE IS TUPLE W/ LENGTH 2
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


def diffPlot(actualSol, modelSol):
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
    plt.plot(t, diff)
