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
import pylab

class Normal(object):
    """
    A class for storing the parameters of a multivariate normal
    distribution.
    """

    def __init__(self, dim, mu = None, sigma = None, data = None):

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
        return Normal(len(odim), mu = new_mu, sigma = new_E)

def show2dnormal(norm, show = False):

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

    # Plot the normalized faithful data points.
    fig = pylab.figure(num = 1, figsize=(4,4))
    pylab.contour(X,Y,Z)

    if show:
        pylab.show()


if __name__ == '__main__':

    # Tests for the ConditionalNormal class...
    mu = [0.1, 0.3]
    sigma = [[0.9, -0.2], [-0.2, 0.9]]
    n = Normal(2, mu = mu, sigma = sigma)
    show2dnormal(n)

    mu = [100, 100]
    sigma = np.eye(2)
    n = Normal(2, mu = mu, sigma = sigma)
    show2dnormal(n)


    pylab.show()
