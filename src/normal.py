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

class Normal(object):
    """
    A class for storing the parameters of a multivariate normal
    distribution. Supports evaluation, sampling, conditioning and
    marginalization.
    """

    def __init__(self, dim, mu = None, sigma = None, data = None,
                 parent = None, cond = None, margin = None):
        """
        Initialize a normal distribution.

        Parameters
        ----------
        dim : int
            Number of dimensions (e.g. number of components in the mu parameter).
        mu : array, optional
            The mean of the normal distribution.
        sigma : array, optional
            The covariance matrix of the normal distribution.
        data : array, optional
            If provided, the parameters of the distribution will be estimated from the data. Rows are observations, columns are components.
        parent : Normal, optional
            A reference to a parent distribution that was marginalized or conditioned.
        cond : dict, optional
            A dict of parameters describing how the parent distribution was conditioned.
        margin : dict, optional
            A dict of parameters describing how the parent distribution was marginalized.

        Examples
        --------
        >>> x = Normal(2,mu = np.array([0.1,0.7]), sigma = np.array([[ 0.6,  0.4], [ 0.4,  0.6]]))
        >>> print x
        [ 0.1  0.7]
        [[ 0.6  0.4]
        [ 0.4  0.6]]

        To condition on a value (and index):
        
        >>> condx = x.condition([0],0.1)
        >>> print condx
        [ 0.7]
        [[ 0.33333333]]
        
        """

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
        self.margin = margin
        self.parent = parent

        self.update(npa(mu),npa(sigma))


    def update(self, mu, sigma):
        """
        Update the distribution with new parameters.

        Parameters
        ----------
        mu : array
            The new mean parameters.
        sigma : array
            The new covariance matrix.

        Example
        -------

        >>> x = Normal(2,mu = np.array([0.1,0.7]), sigma = np.array([[ 0.6,  0.4], [ 0.4,  0.6]]))
        >>> print x
        [ 0.1  0.7]
        [[ 0.6  0.4]
        [ 0.4  0.6]]

        >>> x.update(np.array([0.0,0.0]), x.E)
        >>> print x
        [ 0.0  0.0]
        [[ 0.6  0.4]
        [ 0.4  0.6]]
        """

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

    def pdf_mesh(self, x, y):
        # for 2d meshgrids
        # use matplotlib.mlab.bivariate_normal -- faster (vectorized)

        z = np.zeros((len(y),len(x)))
        
        for (i,v) in enumerate(x):
            for (j,w) in enumerate(y):
                z[j,i] = self.pdf([v,w])
        
        return z

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
        return Normal(len(indices), mu = self.mu[indices], sigma = self.E[ix(indices,indices)], margin = {'indices' : indices}, parent = self)

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
        new_mu = mua - np.dot(premu, (x - mub))

        new_E = iAaa
        return Normal(len(odim), mu = new_mu, sigma = new_E,
                      cond = {'data' : x, 'indices' : indices},
                      parent = self)

