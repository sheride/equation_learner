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
import vpython as vp
from vpython import vector as vec
from vpython.no_notebook import stop_server


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


# from https://trinket.io/glowscript/9bdab2cf88:
#
# double pendulum simulation
# based of an idea from @rjallain at
# http://www.wired.com/2009/12/pendulum-a-third-way/
def simulateDoublePendula(models, func, x0, tEnd=10, deltat=0.005):
    # constants
    t = 0

    theta1, omega1, theta2, omega2 = x0
    L1 = 1
    L2 = 1

    vp.canvas(width=1400, height=750, background=vp.color.white)

    # create the ceiling, masses, and strings
    ceiling = vp.box(pos=vec(0, 1, 0), size=vec(0.1, 0.05, 0.1),
                     color=vp.color.gray(0.5))

    rball1 = vp.sphere(
        pos=vec(ceiling.pos.x+L1*vp.sin(theta1),
                ceiling.pos.y-L1*vp.cos(theta1), 0),
        radius=0.05, color=vp.color.orange)
    rball1.color = vp.color.cyan
    rball1.radius = 0.05
    rball2 = vp.sphere(
        pos=vec(ceiling.pos.x+L1*vp.sin(theta1)+L2*vp.sin(theta2),
                ceiling.pos.y-L1*vp.cos(theta1)-L2*vp.cos(theta2),
                0),
        radius=0.05, color=vp.color.cyan, make_trail=True, interval=10,
        retain=15)
    rball2.color = vp.color.cyan
    rball2.radius = 0.05
    rstring1 = vp.cylinder(pos=ceiling.pos, axis=rball1.pos-ceiling.pos,
                           color=vp.color.gray(0.5), radius=0.008)
    rstring2 = vp.cylinder(pos=rball1.pos, axis=rball2.pos-rball1.pos,
                           color=vp.color.gray(0.5), radius=0.008)

    balls = [None for i in range(len(models) * 2)]
    strings = [None for i in range(len(models) * 2)]
    colors = [vp.color.magenta, vp.color.red, vp.color.orange, vp.color.yellow]

    for i in range(0, len(balls), 2):
        balls[i] = vp.sphere(
            pos=vec(ceiling.pos.x+L1*vp.sin(theta1),
                    ceiling.pos.y-L1*vp.cos(theta1),
                    0),
            radius=0.05, color=colors[int(i/2)])
        balls[i].color = colors[int(i/2)]
        balls[i].radius = 0.05
        balls[i+1] = vp.sphere(
            pos=vec(ceiling.pos.x+L1*vp.sin(theta1)+L2*vp.sin(theta2),
                    ceiling.pos.y-L1*vp.cos(theta1)-L2*vp.cos(theta2),
                    0),
            radius=0.05, color=colors[int(i/2)], make_trail=True, interval=10,
            retain=15)
        balls[i+1].color = colors[int(i/2)]
        balls[i+1].radius = 0.05
        strings[i] = vp.cylinder(pos=ceiling.pos,
                                 axis=balls[i].pos-ceiling.pos,
                                 color=vp.color.gray(0.5), radius=0.008)
        strings[i+1] = vp.cylinder(pos=balls[i].pos,
                                   axis=balls[i+1].pos-balls[i].pos,
                                   color=vp.color.gray(0.5), radius=0.008)

    actualSol, modelsSol = odeSolve(models, func, x0, [0, tEnd], deltat)

    # calculation loop
    while t < tEnd / deltat:
        vp.rate(1/deltat)

        rball1.pos = vec(vp.sin(actualSol.y[0][t]),
                         -vp.cos(actualSol.y[0][t]),
                         0) + ceiling.pos
        rstring1.axis = rball1.pos - ceiling.pos
        rball2.pos = rball1.pos + vec(vp.sin(actualSol.y[2][t]),
                                      -vp.cos(actualSol.y[2][t]),
                                      0)
        rstring2.axis = rball2.pos - rball1.pos
        rstring2.pos = rball1.pos

        for i in range(0, len(balls), 2):
            balls[i].pos = vec(vp.sin(modelsSol[int(i/2)].y[0][t]),
                               -vp.cos(modelsSol[int(i/2)].y[0][t]),
                               0) + ceiling.pos
            strings[i].axis = balls[i].pos - ceiling.pos
            balls[i+1].pos = (balls[i].pos
                              + vec(vp.sin(modelsSol[int(i/2)].y[2][t]),
                                    -vp.cos(modelsSol[int(i/2)].y[2][t]),
                                    0))
            strings[i+1].axis = balls[i+1].pos - balls[i].pos
            strings[i+1].pos = balls[i].pos

        t = t + 1

    stop_server()
