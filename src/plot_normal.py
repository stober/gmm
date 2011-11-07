#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: PLOT_NORMAL.PY
Date: Wednesday, October 26 2011
Description: Visualization of the normal distribution.
"""

import numpy as np
import numpy.linalg as la
import numpy.random as npr
import random as pr
import pylab as pl
import matplotlib
from matplotlib.ticker import NullFormatter
from matplotlib.widgets import Slider
import pdb
from normal import Normal

def draw2dnormal(norm, show = False, axes = None):
    """
    Just draw a simple 2d normal pdf.
    """
    # create a meshgrid centered at mu that takes into account the variance in x and y
    delta = 0.025

    lower_xlim = norm.mu[0] - (2.0 * norm.E[0,0])
    upper_xlim = norm.mu[0] + (2.0 * norm.E[0,0])
    lower_ylim = norm.mu[1] - (2.0 * norm.E[1,1])
    upper_ylim = norm.mu[1] + (2.0 * norm.E[1,1])

    x = np.arange(lower_xlim, upper_xlim, delta)
    y = np.arange(lower_ylim, upper_ylim, delta)

    X,Y = np.meshgrid(x,y)

    # remember sqrts!
    Z = matplotlib.mlab.bivariate_normal(X, Y, sigmax=np.sqrt(norm.E[0,0]), sigmay=np.sqrt(norm.E[1,1]), mux=norm.mu[0], muy=norm.mu[1], sigmaxy=norm.E[0,1])

    minlim = min(lower_xlim, lower_ylim)
    maxlim = max(upper_xlim, upper_ylim)

    # Plot the normalized faithful data points.
    if not axes:
        fig = pl.figure(num = 1, figsize=(4,4))
        pl.contour(X,Y,Z)
        #axes.set_xlim(minlim,maxlim)
        #axes.set_ylim(minlim,maxlim)
    else:
        axes.contour(X,Y,Z)
        #axes.set_xlim(minlim,maxlim)
        #axes.set_ylim(minlim,maxlim)

    if show:
        pl.show()

def evalpdf(norm):
    delta = 0.025
    mu = norm.mu[0]
    sigma = norm.E[0,0]
    lower_xlim = mu - (2.0 * sigma)
    upper_xlim = mu + (2.0 * sigma)
    x = np.arange(lower_xlim,upper_xlim, delta)
    y = matplotlib.mlab.normpdf(x, mu, np.sqrt(sigma))
    return x,y

def draw1dnormal(norm, show = False, axes = None):
    """
    Just draw a simple 1d normal pdf. Used for plotting the conditionals in simple test cases.
    """
    x,y = evalpdf(norm)
    if axes is None:
        pl.plot(x,y)
    else:
        return axes.plot(y,x)

    if show:
        pl.show()

def draw2d1dnormal(norm, cnorm, show = False):

    pl.figure(1, figsize=(8,8))

    nullfmt = NullFormatter()

    rect_2d = [0.1, 0.1, 0.65, 0.65]
    rect_1d = [0.1 + 0.65 + 0.02, 0.1, 0.2, 0.65]
    ax2d = pl.axes(rect_2d)
    ax1d = pl.axes(rect_1d)
    ax1d.xaxis.set_major_formatter(nullfmt)
    ax1d.yaxis.set_major_formatter(nullfmt)
    draw2dnormal(norm, axes = ax2d)
    draw1dnormal(cnorm, axes = ax1d)
    y = ax2d.get_ylim()
    x = [cnorm.cond['data'], cnorm.cond['data']]
    ax2d.plot(x,y)


def draw_slider_demo(norm):

    fig = pl.figure(1, figsize=(8,8))
        
    nullfmt = NullFormatter()

    cnorm = norm.condition([0],2.0)

    rect_slide = [0.1, 0.85, 0.65 + 0.1, 0.05]
    rect_2d = [0.1, 0.1, 0.65, 0.65]
    rect_1d = [0.1 + 0.65 + 0.02, 0.1, 0.2, 0.65]
    ax2d = pl.axes(rect_2d)
    ax1d = pl.axes(rect_1d)
    ax1d.xaxis.set_major_formatter(nullfmt)
    ax1d.yaxis.set_major_formatter(nullfmt)
    axslide = pl.axes(rect_slide)
    slider = Slider(axslide, 'Cond', -4.0,4.0,valinit=2.0)
        
    draw2dnormal(norm, axes = ax2d)
    l2, = draw1dnormal(cnorm, axes = ax1d)

    y = ax2d.get_ylim()
    x = [cnorm.cond['data'], cnorm.cond['data']]
    l1, = ax2d.plot(x,y)
    
    def update(val):
        cnorm = norm.condition([0],val)
        x = [cnorm.cond['data'], cnorm.cond['data']]
        l1.set_xdata(x)
        x,y = evalpdf(cnorm)
        print cnorm
        #print y
        l2.set_xdata(y)
        l2.set_ydata(x)
        pl.draw()
            

    slider.on_changed(update)
    
    return slider

if __name__ == '__main__':
    # Tests for the ConditionalNormal class...
    mu = [1.5, 0.5]
    sigma = [[1.0, 0.5], [0.5, 1.0]]
    n = Normal(2, mu = mu, sigma = sigma)
    sl = draw_slider_demo(n)
    pl.show()
