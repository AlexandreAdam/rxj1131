import pycs3 
from pycs3.sim import twk as twk 

def tweakml_colored_1(lcs, spline):
    return twk.tweakml(lcs, spline, beta=-2.0, sigma=0.6, fmin=1.0 / 500.0, fmax=0.2,
                       psplot=False)

def tweakml_colored_2(lcs, spline):
    return twk.tweakml(lcs, spline, beta=-1.1, sigma=0.9, fmin=1.0 / 500.0, fmax=0.2,
                       psplot=False)

def tweakml_colored_3(lcs, spline):
    return twk.tweakml(lcs, spline, beta=-0.7, sigma=3.2, fmin=1.0 / 500.0, fmax=0.2,
                       psplot=False)

def tweakml_colored_4(lcs, spline):
    return twk.tweakml(lcs, spline, beta=0, sigma=7.5, fmin=1.0 / 500.0, fmax=0.2,
                       psplot=False)

tweakml_list = [tweakml_colored_1,tweakml_colored_2,tweakml_colored_3,tweakml_colored_4,]