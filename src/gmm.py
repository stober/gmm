#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: GMM.PY
Date: Friday, June 24 2011/Volumes/NO NAME/seds/nodes/gmm.py
Description: A python class for creating and manipulating GMMs.
"""

import scipy.cluster.vq as vq
import numpy as np
import numpy.linalg as la
import numpy.random as npr
import random as pr
npa = np.array

import sys; sys.path.append('.')
import pdb

#import matplotlib
import pylab
from normal import Normal

class GMM(object):

    def __init__(self, dim = None, ncomps = None, data = None,  method = None, filename = None, params = None):

        if not filename is None:  # load from file
            self.load_model(filename)

        elif not params is None: # initialize with parameters directly
            self.comps = params['comps']
            self.ncomps = params['ncomps']
            self.dim = params['dim']
            self.priors = params['priors']

        elif not data is None: # initialize from data

            assert dim and ncomps, "Need to define dim and ncomps."

            self.dim = dim
            self.ncomps = ncomps
            self.comps = []

            if method is "uniform":
                # uniformly assign data points to components then estimate the parameters
                npr.shuffle(data)
                n = len(data)
                s = n / ncomps
                for i in range(ncomps):
                    self.comps.append(Normal(dim, data = data[i * s: (i+1) * s]))

                self.priors = np.ones(ncomps, dtype = "double") / ncomps

            elif method is "random":
                # choose ncomp points from data randomly then estimate the parameters
                mus = pr.sample(data,ncomps)
                clusters = [[] for i in range(ncomps)]
                for d in data:
                    i = np.argmin([la.norm(d - m) for m in mus])
                    clusters[i].append(d)

                for i in range(ncomps):
                    print mus[i], clusters[i]
                    self.comps.append(Normal(dim, mu = mus[i], sigma = np.cov(clusters[i], rowvar=0)))

                self.priors = np.ones(ncomps, dtype="double") / np.array([len(c) for c in clusters])

            elif method is "kmeans":
                # use kmeans to initialize the parameters
                (centroids, labels) = vq.kmeans2(data, ncomps, minit="points", iter=100)
                clusters = [[] for i in range(ncomps)]
                for (l,d) in zip(labels,data):
                    clusters[l].append(d)

                # will end up recomputing the cluster centers
                for cluster in clusters:
                    self.comps.append(Normal(dim, data = cluster))

                self.priors = np.ones(ncomps, dtype="double") / np.array([len(c) for c in clusters])

            else:
                raise ValueError, "Unknown method type!"

        else:

            # these need to be defined
            assert dim and ncomps, "Need to define dim and ncomps."

            self.dim = dim
            self.ncomps = ncomps

            self.comps = []

            for i in range(ncomps):
                self.comps.append(Normal(dim))

            self.priors = np.ones(ncomps,dtype='double') / ncomps

    def __str__(self):
        res = "%d" % self.dim
        res += "\n%s" % str(self.priors)
        for comp in self.comps:
            res += "\n%s" % str(comp)
        return res

    def save_model(self):
        pass

    def load_model(self):
        pass

    def mean(self):
        return np.sum([self.priors[i] * self.comps[i].mean() for i in range(self.ncomps)], axis=0)

    def covariance(self): # computed using Dan's method
        m = self.mean()
        s = -np.outer(m,m)

        for i in range(self.ncomps):
            cm = self.comps[i].mean()
            cvar = self.comps[i].covariance()
            s += self.priors[i] * (np.outer(cm,cm) + cvar)

        return s

    def pdf(self, x):
        responses = [comp.pdf(x) for comp in self.comps]
        return np.dot(self.priors, responses)

    def condition(self, indices, x):
        """
        Create a new GMM conditioned on data x at indices.
        """
        condition_comps = []
        marginal_comps = []

        for comp in self.comps:
            condition_comps.append(comp.condition(indices, x))
            marginal_comps.append(comp.marginalize(indices))

        new_priors = []
        for (i,prior) in enumerate(self.priors):
            new_priors.append(prior * marginal_comps[i].pdf(x))
        new_priors = npa(new_priors) / np.sum(new_priors)

        params = {'ncomps' : self.ncomps, 'comps' : condition_comps,
                  'priors' : new_priors, 'dim' : marginal_comps[0].dim}

        return GMM(params = params)

    def em(self, data, nsteps = 100):

        k = self.ncomps
        d = self.dim
        n = len(data)

        for l in range(nsteps):

            # E step

            responses = np.zeros((k,n))

            for j in range(n):
                for i in range(k):
                    responses[i,j] = self.priors[i] * self.comps[i].pdf(data[j])

            responses = responses / np.sum(responses,axis=0) # normalize the weights

            # M step

            N = np.sum(responses,axis=1)

            for i in range(k):
                mu = np.dot(responses[i,:],data) / N[i]
                sigma = np.zeros((d,d))

                for j in range(n):
                   sigma += responses[i,j] * np.outer(data[j,:] - mu, data[j,:] - mu)

                sigma = sigma / N[i]

                self.comps[i].update(mu,sigma) # update the normal with new parameters
                self.priors[i] = N[i] / np.sum(N) # normalize the new priors


def shownormal(data,gmm):

    xnorm = data[:,0]
    ynorm = data[:,1]

    # Plot the normalized faithful data points.
    fig = pylab.figure(num = 1, figsize=(4,4))
    axes = fig.add_subplot(111)
    axes.plot(xnorm,ynorm, '+')

    # Plot the ellipses representing the principle components of the normals.
    for comp in gmm.comps:
        comp.patch(axes)

    pylab.draw()
    pylab.show()


if __name__ == '__main__':

    """
    Tests for gmm module.
    """


    # x = npr.randn(20, 2)

    # print "No data"
    # gmm = GMM(2,1,2) # possibly also broken
    # print gmm

    # print "Uniform"
    # gmm = GMM(2,1,2,data = x, method = "uniform")
    # print gmm

    # print "Random"
    # gmm = GMM(2,1,2,data = x, method = "random") # broken
    # print gmm

    # print "Kmeans"
    # gmm = GMM(2,1,2,data = x, method = "kmeans") # possibly broken
    # print gmm


    x = np.arange(-10,30)
    #y = x ** 2 + npr.randn(20)
    y = x + npr.randn(40) # simple linear function
    #y = np.sin(x) + npr.randn(20)
    data = np.vstack([x,y]).T
    print data.shape


    gmm = GMM(dim = 2, ncomps = 4,data = data, method = "random")
    print gmm
    shownormal(data,gmm)

    gmm.em(data,nsteps=1000)
    shownormal(data,gmm)
    print gmm
    ngmm = gmm.condition([0],[-3])
    print ngmm.mean()
    print ngmm.covariance()
