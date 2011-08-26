#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: GMR.PY
Date: Friday, June 24 2011
Description: An experimental python replacement for fgmm and ds_node.
"""

import roslib
roslib.load_manifest('seds')
import rospy
from seds.srv import DSLoad, DSSrv, DSLoadResponse, DSSrvResponse

import numpy as np
import numpy.linalg as la
import numpy.random as npr
npa = np.array

import sys
import pdb

class Normal(object):
    """
    A class for storing the parameters of a conditional multivariate
    normal distribution. Pre-computes as many parameters as possible
    for fast pdf estimation.
    """

    def __init__(self, idim, mu, sigma):
        self.idim = idim # dimension of the inputs for the conditional distribution
        self.mu = mu
        self.sigma = sigma

        Eaa = sigma[:self.idim,:self.idim]
        Ebb = sigma[self.idim:,self.idim:]
        Eab = sigma[:self.idim,self.idim:]
        Eba = sigma[self.idim:,:self.idim]

        # we are going to be computing b | a conditional probabilities
        # b : dx
        # a : x

        iEaa = la.inv(Eaa)
        det = la.det(Eaa)

        # constant factor in pdf estimation
        factor = (2.0 * np.pi)**(self.idim / 2.0) * (det)**(0.5)

        # precompute parameters for conditional variance calculation
        # from Bishop pg. 87
        Ebca = Ebb - np.dot(np.dot(Eba, iEaa), Eab)

        # precompute parameters for conditional mean calculation
        # from Bishop pg. 87
        mEbca = np.dot(Eba, iEaa)

        self.subs = {'E' : sigma, 'dEaa' : det,
                     'fEaa' : factor, 'Eaa' : Eaa, 'Ebb' : Ebb,
                     'Eab' : Eab, 'Eba' : Eba,'iEaa' : iEaa,
                     'Ebca' : Ebca, 'mEbca' : mEbca}

    def cpdf(self, x):

        iEaa = self.subs['iEaa']
        fEaa = self.subs['fEaa']
        mu = self.mu[:self.idim]

        answer = np.exp(-0.5 * np.dot(np.dot(x - mu, iEaa), x - mu)) / fEaa

        # note that for extreme values this differs from fgmm, but we match R!
        rospy.logdebug("x : %s mu : %s precision : %s  answer : %s" % (str(x), str(mu), str(iEaa), str(answer)))

        return answer

    def cmean(self, x):
        mua = self.mu[:self.idim]
        mub = self.mu[self.idim:]
        mEbca = self.subs['mEbca']

        # from Bishop pg. 87
        return mub + np.dot(mEbca, (x - mua))


class GMR(object):


    def __init__(self, filename):

        # load mus,sigmas,ncomp,dim from file
        self.load_model(filename)

        # do a bunch of precomputations for each component
        self.comps = []
        for i in range(self.ncomps):
            self.comps.append(Normal(self.dim, self.mus[:,i], self.sigmas[i]))

    def load_model(self, filename):
        fp = open(filename)
        raw_data = [] # store for parameters which will populate mu and sigma structures
        for (i,line) in enumerate(fp):

            if (i == 0):
                self.dim = int(line)
            elif (i == 1):
                self.ncomps = int(line)
            elif (i == 2):
                self.dT = float(line)
            elif (i == 4):
                self.offset = npa(line.split(),dtype='double')
            elif (i == 6):
                self.priors = npa(line.split(),dtype='double')
            elif (i > 6):
                # store all parameters in the raw data array -- use
                # this to populate mu and sigma objects -- otherwise
                # indexing gets too incomprehensible
                raw_data.extend([float(x) for x in line.split()])

        # now populate self.mus and self.sigmas based on the raw parameters
        size = 2 * self.dim
        self.mus = npa(raw_data[:self.ncomps*size]).reshape((size,self.ncomps))
        del raw_data[:self.ncomps * size]

        self.sigmas = []
        for i in range(self.ncomps):
            self.sigmas.append(npa(raw_data[:size**2]).reshape((size,size)))
            del raw_data[:size**2]
        self.sigmas = npa(self.sigmas)

    def compute_conditional_priors(self,x):
        self.cpriors = np.zeros(self.ncomps)

        for i in range(self.ncomps):
            self.cpriors[i] = self.priors[i] * self.comps[i].cpdf(x)

        # normalize cpriors
        self.cpriors = self.cpriors / np.sum(self.cpriors)

    def compute_conditional_means(self,x):
        self.cmeans = []

        for i in range(self.ncomps):
            self.cmeans.append(self.comps[i].cmean(x))

    def regression(self, x):
        #pdb.set_trace()
        self.compute_conditional_means(x)
        self.compute_conditional_priors(x)
        return np.dot(self.cpriors, self.cmeans)

def load_model(req):
    global gmr
    rospy.loginfo("Loading model %s" % req.filename)
    gmr = GMR(req.filename)
    rospy.loginfo("dT: %f" % gmr.dT)
    return DSLoadResponse()

def ds_server(req):
    global gmr
    if gmr == None:
        rospy.loginfo("No model loading!")
        return DSSrvResponse()
    else:
        nx = (npa(req.x) - gmr.offset[:gmr.dim]) * 1000.0
        dx = gmr.regression(nx) / 1000.0
        rospy.logdebug("x : %s dx : %s" % (str(req.x), str(dx)))
        return DSSrvResponse(dx)

def init():
    rospy.init_node('gmr')
    rospy.Service('load_model', DSLoad, load_model)
    rospy.Service('ds_server', DSSrv, ds_server)
    rospy.loginfo('ds_node ready.')
    rospy.spin()

if __name__ == '__main__':

    init()
