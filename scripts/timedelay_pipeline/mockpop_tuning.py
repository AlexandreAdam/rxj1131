import numpy as np
import os
import pycs3.gen.lc_func
import pycs3.gen.splml
import pycs3.gen.mrg
import pycs3.spl.topopt
import pycs3.regdiff.multiopt
import pycs3.regdiff.rslc
import pycs3.gen.util
import pycs3.sim.draw
import pycs3.sim.run
import pycs3.sim.plot
import pycs3.sim.twk
import pycs3.tdcomb.plot
import pycs3.tdcomb.comb
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
import json
loggerformat='%(message)s'
logging.basicConfig(format=loggerformat,level=logging.INFO)

'''
Script to generate small samples of mock lcs in order to tune
the power-law noise parameters
'''

# Spline optimization
def spl(lcs, **kwargs):
    # opt_fine: supposes that lcs are within 10 days of true delays
    spline = pycs3.spl.topopt.opt_rough(lcs, nit=5, knotstep=kwargs['kn_r'], verbose=False)
    spline = pycs3.spl.topopt.opt_fine(lcs, nit=5, knotstep=kwargs['kn_f'], verbose=False)
    return spline

# Gaussian process regdiff optimization
def regdiff(lcs, **kwargs):
    return pycs3.regdiff.multiopt.opt_ts(lcs, pd=kwargs['pd'], covkernel=kwargs['covkernel'], pow=kwargs['pow'],
                                        errscale=kwargs['errscale'], verbose=False, method='weights')


# Drawing mocks
kwargs_opt_spl = {'kn_r':40,'kn_f':20}

noise_params = {'shotnoise': 'magerrs',
				'betaA': -2.0,
				'betaB': -1.1,
				'betaC': -0.7,
				'betaD': 0,
				'sigmaA': 0.06,
				'sigmaB': 0.9,
				'sigmaC': 3.2,
				'sigmaD': 7.5}

simfolder="../../data/sim10"		# change folder for new parameters

if not os.path.isdir(simfolder):
    os.mkdir(simfolder)


# Deterministic component of generative model is intrinsic spline
# that was fit for 4 lcs of RXJ1131, along with the ML splines
lcs, spline = pycs3.gen.util.readpickle("../../data/sploptcurves.pkl")


# Mock light curve generation
# Parameters to tune: beta and sigma for each lc
def Atweakml(lcs, spline):
    return pycs3.sim.twk.tweakml(lcs, spline, beta=noise_params['betaA'], sigma=noise_params['sigmaA'], fmin=1/500, fmax=None, psplot=False)

def Btweakml(lcs, spline):
    return pycs3.sim.twk.tweakml(lcs, spline, beta=noise_params['betaB'], sigma=noise_params['sigmaB'], fmin=1/500, fmax=None, psplot=False)

def Ctweakml(lcs, spline):
    return pycs3.sim.twk.tweakml(lcs, spline, beta=noise_params['betaC'], sigma=noise_params['sigmaC'], fmin=1/500, fmax=None, psplot=False)

def Dtweakml(lcs, spline):
    return pycs3.sim.twk.tweakml(lcs, spline, beta=noise_params['betaD'], sigma=noise_params['sigmaD'], fmin=1/500, fmax=None, psplot=False)



# Eyeballing the tuning
pycs3.sim.draw.saveresiduals(lcs, spline)
mocklcs = pycs3.sim.draw.draw(lcs, spline, shotnoise=noise_params['shotnoise'], tweakml=[Atweakml, Btweakml, Ctweakml, Dtweakml], keeptweakedml=False)

for l in mocklcs:
	l.plotcolour = 'black'

rls_lcs = pycs3.gen.stat.subtract(lcs, spline)
rls_mock = pycs3.gen.stat.subtract(mocklcs, spline)
# pycs3.sim.twk.tweakml(lcs, spline, beta=-1, sigma=0.9, fmin=1/500)


# pycs3.gen.lc_func.display(mocklcs, [spline], collapseref=True)
pycs3.gen.stat.plotresiduals([rls_lcs, rls_mock])
# pycs3.gen.lc_func.display(lcs, [spline], collapseref=True)




# # Thorough analysis of z_run and standard deviation (create new simfolder for this)

# # Save noise_params dict
# with open(simfolder+'/noise_params.json', 'w') as f:
# 	json.dump(noise_params, f)

# pycs3.sim.draw.saveresiduals(lcs, spline)

# # Generate 5 pkl files containing 5 lc lists each
# pycs3.sim.draw.multidraw(lcs, spline=spline, shotnoise=noise_params['shotnoise'],  n=5, npkl=10, simset='mock',
#                         truetsr=20, tweakml=[Atweakml, Btweakml, Ctweakml, Dtweakml], destpath=simfolder)

# # Optimize mock curves
# success_dic = pycs3.sim.run.multirun('mock', lcs, spl, kwargs_opt_spl, optset='spl', tsrand=10, keepopt=True, destpath=simfolder)

# # Check residual statistics (this can be run individually from simfolder)
# stats = pycs3.gen.stat.anaoptdrawn(lcs, spline, simset='mock', optset='spl', showplot=True, nplots=1, 
# 									directory=simfolder)
