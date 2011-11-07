#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: TEST_NORMAL.PY
Date: Wednesday, October 26 2011
Description: Fit normal to old faithful data.
"""


from normal import Normal
from plot_normal import draw2dnormal
import pylab as pl
import numpy as np
npa = np.array
import pdb


if False:
    fp = open("faithful.txt")
    data = []
    for line in fp.readlines():
        x,y = line.split()
        data.append([float(x),float(y)])

    data = npa(data)
    pl.scatter(data[:,0],data[:,1])
    x = Normal(2, data=data)
    draw2dnormal(x,show=True,axes=pl.gca())

if True:
    x = Normal(2,mu = np.array([0.1,0.7]), sigma = np.array([[ 0.6,  0.4], [ 0.4,  0.6]]))
    s = x.simulate()
    draw2dnormal(x)
    pl.scatter(s[:,0],s[:,1])
    pl.show()
    print s

if False:
    x = Normal(2,mu = np.array([0.1,0.7]), sigma = np.array([[ 0.6,  0.4], [ 0.4,  0.6]]))
    #draw2dnormal(x,show=True)
    print x
    new = x.condition([0],0.1)
    print new

if False:

    from randcov import gencov
    import numpy.random as npr
    import numpy.linalg as la

    S = gencov(5)
    mu = npr.randn(5)

    x = Normal(5,mu = mu, sigma = S)
    newx = x.condition([0,1],np.array([0.1,0.3]))
    print newx

    A = la.inv(S)
    newS = la.inv(A[2:,2:])
    newmu = mu[2:] - np.dot(np.dot(newS, A[2:,:2]), (np.array([0.1,0.3])- mu[:2]))

    print newmu
    print newS # should match above

    
    
