#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:03:21 2019

@author: elijahsheridan
"""

from . import ode
import vpython as vp
from vpython import vector as vec
from vpython.no_notebook import stop_server
import numpy as np
from scipy.integrate import solve_ivp

def justOne(func, x0, tEnd=10, deltat=0.005):
    # constants
    t = 0

    theta1, omega1, theta2, omega2 = x0
    L1 = 1
    L2 = 1

    vp.canvas(width=1400, height=750, background=vp.color.white, range=1.75,
              autoscale=False, userpan=False, userspin=False, userzoom = False)

    # create the ceiling, masses, and strings
    ceiling = vp.box(pos=vec(0, 1, 0), size=vec(0.1, 0.05, 0.1),
                     color=vp.color.gray(0.5))

    rball1 = vp.sphere(
        pos=vec(ceiling.pos.x+L1*vp.sin(theta1),
                ceiling.pos.y-L1*vp.cos(theta1), 0),
        radius=0.05, color=vp.color.orange)
    rball1.color = vp.color.black
    rball1.radius = 0.05
    rball2 = vp.sphere(
        pos=vec(ceiling.pos.x+L1*vp.sin(theta1)+L2*vp.sin(theta2),
                ceiling.pos.y-L1*vp.cos(theta1)-L2*vp.cos(theta2),
                0),
        radius=0.05, color=vp.color.black, make_trail=True, interval=10,
        retain=15)
    rball2.color = vp.color.black
    rball2.radius = 0.05
    rstring1 = vp.cylinder(pos=ceiling.pos, axis=rball1.pos-ceiling.pos,
                           color=vp.color.gray(0.5), radius=0.008)
    rstring2 = vp.cylinder(pos=rball1.pos, axis=rball2.pos-rball1.pos,
                           color=vp.color.gray(0.5), radius=0.008)

    actualSol = solve_ivp(func, [0, tEnd], x0,
                          t_eval=np.linspace(0, tEnd, int(tEnd/deltat)))

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

        t = t + 1

    stop_server()

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

    vp.canvas(width=1400, height=750, background=vp.color.white, range=1.75,
              autoscale=False, userpan=False, userspin=False, userzoom = False)

    # create the ceiling, masses, and strings
    ceiling = vp.box(pos=vec(0, 1, 0), size=vec(0.1, 0.05, 0.1),
                     color=vp.color.gray(0.5))

    rball1 = vp.sphere(
        pos=vec(ceiling.pos.x+L1*vp.sin(theta1),
                ceiling.pos.y-L1*vp.cos(theta1), 0),
        radius=0.05, color=vp.color.orange)
    rball1.color = vp.color.black
    rball1.radius = 0.05
    rball2 = vp.sphere(
        pos=vec(ceiling.pos.x+L1*vp.sin(theta1)+L2*vp.sin(theta2),
                ceiling.pos.y-L1*vp.cos(theta1)-L2*vp.cos(theta2),
                0),
        radius=0.05, color=vp.color.black, make_trail=True, interval=10,
        retain=15)
    rball2.color = vp.color.black
    rball2.radius = 0.05
    rstring1 = vp.cylinder(pos=ceiling.pos, axis=rball1.pos-ceiling.pos,
                           color=vp.color.gray(0.5), radius=0.008)
    rstring2 = vp.cylinder(pos=rball1.pos, axis=rball2.pos-rball1.pos,
                           color=vp.color.gray(0.5), radius=0.008)

    balls = [None for i in range(len(models) * 2)]
    strings = [None for i in range(len(models) * 2)]
    colors = [vec(240/255, 133/255, 33/255), vec(110/255, 0, 95/255),
              vec(0, 129/255, 131/255)]

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

    actualSol, modelsSol = ode.odeSolve(models, func, x0, [0, tEnd], deltat)

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