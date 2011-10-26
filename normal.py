#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: NORMAL.PY
Date: Friday, July 7, 2011
Description: Manipulating normal distributions.
"""
import numpy as np
import numpy.linalg as la
import numpy.random as npr
import random as pr
npa = np.array
ix  = np.ix_ # urgh - sometimes numpy is ugly!

import sys
import pdb

import matplotlib
import pylab as pl
from matplotlib.ticker import NullFormatter
from matplotlib.widgets import Slider

class Normal(object):
    """
    A class for storing the parameters of a multivariate normal
    distribution.

    cond : (parent, conditional value)
    """

    def __init__(self, dim, mu = None, sigma = None, data = None, parent = None, cond = None):

        self.dim = dim # full data dimension

        if not mu is None  and not sigma is None:
            pass
        elif not data is None:
            # estimate the parameters from data - rows are samples, cols are variables
            mu, sigma = self.estimate(data)
        else:
            # generate random means
            mu = npr.randn(dim)
            sigma = np.eye(dim)

        self.cond = cond
        self.parent = parent

        self.update(npa(mu),npa(sigma))


    def update(self, mu, sigma):
        self.mu = mu
        self.E = sigma

        det = None
        if self.dim == 1:
            self.A = 1.0 / self.E
            det = np.fabs(self.E[0])
        else:
            self.A = la.inv(self.E) # precision matrix
            det = np.fabs(la.det(self.E))

        self.factor = (2.0 * np.pi)**(self.dim / 2.0) * (det)**(0.5)

    def __str__(self):
        return "%s\n%s" % (str(self.mu), str(self.E))

    def mean(self):
        return self.mu

    def covariance(self):
        return self.E

    def pdf(self, x):
        dx = x - self.mu
        A = self.A
        fE = self.factor

        return np.exp(-0.5 * np.dot(np.dot(dx,A),dx)) / fE

    def simulate(self, ndata = 100):
        """
        Draw pts from the distribution.
        """
        return npr.multivariate_normal(self.mu, self.E, ndata)

    def estimate(self, data):
        mu = np.mean(data, axis=0)
        sigma = np.cov(data, rowvar=0)
        return mu, sigma

    def marginalize(self, indices):
        """
        Creates a new marginal normal distribution for ''indices''.
        """
        indices = npa(indices)
        return Normal(len(indices), mu = self.mu[indices], sigma = self.E[ix(indices,indices)])

    def condition(self, indices, x):
        """
        Creates a new normal distribution conditioned on the data x at indices.
        """

        idim = indices
        odim = npa([i for i in range(self.dim) if not i in indices])

        Aaa = self.A[ix(odim,odim)]
        Aab = self.A[ix(odim,idim)]
        iAaa = None
        det = None

        if len(odim) == 1: # linalg does not handle d1 arrays
            iAaa = 1.0 / Aaa
            det = np.fabs(iAaa[0])
        else:
            iAaa = la.inv(Aaa)
            det = np.fabs(la.det(iAaa))

        # compute the new mu
        premu = np.dot(iAaa, Aab)

        mub = self.mu[idim]
        mua = self.mu[odim]
        new_mu = mua + np.dot(premu, (x - mub))

        new_E = iAaa
        return Normal(len(odim), mu = new_mu, sigma = new_E, cond = {'data' : x, 'indices' : indices}, parent = self)

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
    Z = matplotlib.mlab.bivariate_normal(X, Y, sigmax=norm.E[0,0], sigmay=norm.E[1,1], mux=norm.mu[0], muy=norm.mu[1], sigmaxy=norm.E[0,1])


    minlim = min(lower_xlim, lower_ylim)
    maxlim = max(upper_xlim, upper_ylim)


    # Plot the normalized faithful data points.
    if not axes:
        fig = pl.figure(num = 1, figsize=(4,4))
        pl.contour(X,Y,Z)
        axes.set_xlim(minlim,maxlim)
        axes.set_ylim(minlim,maxlim)
    else:
        axes.contour(X,Y,Z)
        axes.set_xlim(minlim,maxlim)
        axes.set_ylim(minlim,maxlim)



    if show:
        pl.show()

def evalpdf(norm):
    delta = 0.025
    mu = norm.mu[0]
    sigma = norm.E[0,0]
    lower_xlim = mu - (2.0 * sigma)
    upper_xlim = mu + (2.0 * sigma)
    x = np.arange(lower_xlim,upper_xlim, delta)
    y = matplotlib.mlab.normpdf(x, mu, sigma)
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


if __name__ == '__main__':


    def draw_slider_demo(norm):

        
        #pl.subplot(111)
        fig = pl.figure(1, figsize=(8,8))
        
        nullfmt = NullFormatter()

        cnorm = norm.condition([0],2.0)

        #rect = [left, bottom, width, height]

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


        # TODO: change the range of the slider to match the 2d normal

        #pdb.set_trace()

        
        # def update(val):
        #     print "test"
        #     pl.draw()
        #     l.set_xdata([val,val])


    # Tests for the ConditionalNormal class...
    mu = [0.5, 0.5]
    # rot = [[np.cos(np.pi / 4.0), -np.sin(np.pi / 4.0)],
    #        [np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)]]
    sigma = [[1.0, 0.5], [0.5, 1.0]]
    #sigma = np.dot(np.array(rot), np.array(sigma))
    print sigma
    n = Normal(2, mu = mu, sigma = sigma)
    #draw2dnormal(n)

    # mu = [100, 100]
    # sigma = np.eye(2)
    # n = Normal(2, mu = mu, sigma = sigma)
    # draw2dnormal(n)

    data = n.simulate(1000)
    #pl.scatter(data[:,0],data[:,1])

    pdb.set_trace()
    newn = n.condition([0],2.0)
    #draw1dnormal(newn)

    #draw2d1dnormal(n,newn)
    sl = draw_slider_demo(n)
    pl.show()
