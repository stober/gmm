#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: PLOT_GMM.PY
Date: Thursday, November  3 2011
Description: Code for plotting GMMs
"""

from plot_normal import draw2dnormal

def draw2dgmm(gmm, show = False, axes = None):
    
    for comp in gmm.comps:
        draw2dnormal(comp)

    
