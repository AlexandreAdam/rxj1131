"""
=============================================
MCMC Time-delay Inference

Description: Infer time-delay distance
	and time-delay measurement from
	inferred lens parameters

=============================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee as mc
import pickle
import astropy.units as u
import astropy.constants as csts
from scipy.integrate import quad
from scipy.stats import pearson3
import corner
from pathos.multiprocessing import ProcessingPool as Pool

plt.style.use('science')

c = csts.c.to(u.Mpc / u.day).value
arcsec2rad = np.pi / 180 / 3600
fac = arcsec2rad ** 2 / c

# ===========================================
# Definition of known distributions
# ===========================================
Ddt_priorbounds = [0, 1e4]  # (Mpc)

potential_moments_file = './potential_moments_apr28_22_57.txt'
TD_moments_file = './TD_moments.txt'


# filereader for moment results
def filereader(file):
    data = pd.read_csv(file, sep=' ', index_col=0)
    return data


# We work with pairs BA, BC, BD
phi_moments = filereader(potential_moments_file)
TD_moments = filereader(TD_moments_file)
TD_moments = TD_moments.loc[['BA', 'BC', 'BD']]

# print(phi_moments)
# print()
# print(TD_moments)
# print()

# Convert to array every moment of both ditributions
moments = {
    'dphi': phi_moments['mean'].to_numpy(),
    'dphi_std': phi_moments['std'].to_numpy(),
    'dphi_skew': phi_moments['skew'].to_numpy(),
    'dt': TD_moments['mean'].to_numpy(),
    'dt_std': TD_moments['std'].to_numpy()
}


# ===========================================
# Log-likelihoods
# ===========================================
def TD_loglikelihood(delta_t, dt, dt_std):
    """
	ll of time delays

	:param delta_t: time delay estimate
	:param dt: measured time delay
	:param dt_std: std dev on measured time delay
	"""

    return -1 / 2 * ((delta_t - dt) ** 2) / (dt_std) ** 2


def phi_logprior(delta_phi, dphi, dphi_std, dphi_skew):
    """
	ll of Fermat potential  difference

	:param delta_phi: walker for potential difference
	:param dphi: previously inferred potential difference
	:param dphi_std: std dev on inferred potential difference
	:param dphi_skew: skewness of inferred potential difference
	"""

    return pearson3.logpdf(delta_phi, loc=dphi, scale=dphi_std, skew=dphi_skew)


def log_prob(walker):
    """
	function joining ll, to sample with emcee.
	Called for a single walker, of shape (ndims)

	:param walkers: MCMC walkers, array of shape (nwalkers, ndim)
					dims are (D_dt, dphi_BA, dphi_BC, dphi_BD)
	"""
    D_dt = walker[0]
    delta_phi = walker[1:]

    # Evaluate ll of N walker's three dphi estimates
    ll_dphi = np.sum(phi_logprior(delta_phi, moments['dphi'], moments['dphi_std'], moments['dphi_skew']))
    # ll_dphi = sum([phi_logprior(dphi[i], mean, std, skew) for i in range(3)])	# (nwalkers)

    # Compute time delay estimates from the Refsdal relation, for the three dphi of a walker
    dts = D_dt * fac * delta_phi    # (3,)

    # Evaluate ll of time delay esitmates
    ll_TD = np.sum(TD_loglikelihood(dts, moments['dt'], moments['dt_std']))

    return ll_TD + ll_dphi


# ===========================================
# Sampling
# ===========================================
def main(nwalkers, nburn, nsteps, nworkers):
    """
	Principal function to run the complete MCMC, with multiprocessing

	:param nwalkers:
	:param nburn:
	"""
    # First sample from Ddt prior for dim 0 of walkers:
    walkers_dim0 = np.random.uniform(low=0, high=1e4, size=(nwalkers, 1))

    # Then sample from dphi prior for dims 1:3 of walkers
    walkers_dim1_3 = pearson3.rvs(loc=moments['dphi'],
                                  scale=moments['dphi_std'],
                                  skew=moments['dphi_skew'],
                                  size=(nwalkers, 3))
    # initial guess
    p0 = np.hstack((walkers_dim0, walkers_dim1_3))

    # with Pool(nworkers) as pool:
    sampler = mc.EnsembleSampler(nwalkers, 4, log_prob)#, pool=pool)

    # Burn-in
    state = sampler.run_mcmc(p0, nburn)
    sampler.reset()

    sampler.run_mcmc(state, nsteps, progress=True)

    return sampler

# def results(sampler):



# def plotresults(sampler):


if __name__ == '__main__':
    sampler = main(nwalkers=100, nburn=100, nsteps=1000, nworkers=8)

    samples = sampler.get_chain(flat=True)
    print(np.quantile(samples[:,0], [0.16, 0.5, 0.84]))

    plt.figure(figsize=(8,6))
    plt.hist(samples[:,0], 100, color='k', histtype='step')
    plt.xlabel(r'$D_{\Delta t}$ (Mpc)')
    plt.gca().set_yticks([])
    plt.tight_layout()
    plt.show()