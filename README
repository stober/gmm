Gaussian Mixture Models in Python

Author: Jeremy Stober
Contact: stober@gmail.com
Version: 0.01

This is a standalone Pythonic implementation of Gaussian Mixture
Models. Various initialization strategies are included along with a
standard EM algorithm for determining the model parameters based on
data.

Example code for the GMM and Normal classes can be found in the
src/test_*.py files. The GMM and the underlying Normal class both
support conditioning on data and marginalization for any subset of the
variables. This makes this implementation ideal for experimenting with
Gaussian Mixture Regression. For example, the following code learns
the cosine function:


import numpy as np
from gmm import GMM
from plot_gmm import draw2dgmm
from test_func import noisy_cosine
import pylab as pl

x,y = noisy_cosine()
data = np.vstack([x,y]).transpose()
pl.scatter(data[:,0],data[:,1])

gmm = GMM(dim = 2, ncomps = 2, data = data, method = "kmeans")
draw2dgmm(gmm)

nx = np.arange(0,2 * np.pi, 0.1)
ny = []
for i in nx:
    ngmm = gmm.condition([0],[i])
    ny.append(ngmm.mean()) 

pl.plot(nx,ny,color='red')
pl.show()


