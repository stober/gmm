#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: TEST_GMM.PY
Date: Thursday, November  3 2011
Description: Testing code for gmm with new normal distribution.
"""

import numpy as np
npa = np.array
import pylab as pl
from normal import Normal
from gmm import GMM
from plot_normal import draw2dnormal
from plot_gmm import draw2dgmm

if False:
    fp = open("../data/faithful.txt")
    data = []
    for line in fp.readlines():
        x,y = line.split()
        data.append([float(x),float(y)])

    data = npa(data)
    pl.scatter(data[:,0],data[:,1])
    gmm = GMM(dim = 2, ncomps = 2, data = data, method = "kmeans")

    #x = Normal(2, data=data)
    #draw2dnormal(x,show=True,axes=pl.gca())
    print gmm
    draw2dgmm(gmm)
    pl.show()

if False:

    from test_func import noisy_cosine

    x,y = noisy_cosine()
    data = np.vstack([x,y]).transpose()
    pl.scatter(data[:,0],data[:,1])

    gmm = GMM(dim = 2, ncomps = 2, data = data, method = "kmeans")

    draw2dgmm(gmm)
    pl.show()
    #print data
    

if True:

    from test_func import noisy_cosine

    x,y = noisy_cosine()
    data = np.vstack([x,y]).transpose()
    pl.scatter(data[:,0],data[:,1])

    gmm = GMM(dim = 2, ncomps = 2, data = data, method = "kmeans")

    draw2dgmm(gmm)

    #pl.show()

    nx = np.arange(0,2 * np.pi, 0.1)
    ny = []
    for i in nx:
        ngmm = gmm.condition([0],[i])
        ny.append(ngmm.mean()) 

    #ngmm = gmm.condition([0],[0.5])
    #print ngmm.mean()
    #print np.cos(0.5)
    pl.plot(nx,ny,color='red')
    pl.show()
    #print data
