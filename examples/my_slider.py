#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: MY_SLIDER.PY
Date: Tuesday, October 11 2011
Description: Testing slider with plots.
"""


import pylab as pl
from matplotlib.widgets import Slider


fig = pl.figure(1, figsize=(8,8))

rect_slide = [0.1, 0.85, 0.65 + 0.1, 0.05]
rect_2 = [0.1, 0.1, 0.65, 0.65]
rect_1 = [0.1 + 0.65 + 0.02, 0.1, 0.2, 0.65]


ax1 = pl.axes(rect_1)
ax2 = pl.axes(rect_2)
axs = pl.axes(rect_slide)

slider = Slider(axs, "TEST", -1.0, 1.0, valinit=0.5)

def update(val):
    print "test"

slider.on_changed(update)

pl.show()

