#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: TEST_FUNC.PY
Date: Monday, November  7 2011
Description: 2-d noisy function for testing GMM/GMR.
"""

import numpy as np
import pylab as pl
import numpy.random as npr


def noisy_cosine():
    x = npr.rand(100) * np.pi * 2.0
    x.sort()
    y = np.cos(x) + 0.1 * npr.randn(100)
    return x,y

if __name__ == '__main__':
    #pl.plot(*noisy_cosine())
    x,y = noisy_cosine()
    pl.scatter(x,y)
    pl.show()
    
