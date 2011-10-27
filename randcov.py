#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: RANDCOV.PY
Date: Thursday, October 27 2011
Description: Generate random cov matrix in numpy.
"""

from numpy import *
from numpy.linalg import *
from numpy.random import *


def gencov(n):
    S = randn(n,n)
    S = dot(S.transpose(), S)
    s = sqrt(diag(S))
    t = diag(1.0/s)
    C = dot(dot(t,S),t)
    return C

if __name__ == '__main__':
    print gencov(2)
