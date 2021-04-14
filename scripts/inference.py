"""
=============================================
Title: Nested Sampling Inference

Author(s): Alexandre Adam

Last modified: March 11, 2021

Description: Infere the lens parameter
    and time delay distance from quasar
    four images and time delay measurment with
    Dynamic Nested Sampling algorithm.
=============================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
import pickle
from dynesty import DynamicNestedSampler
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from pathos.multiprocessing import ProcessingPool as Pool # better Pool
from astropy.constants import c
from datetime import datetime
from scipy.integrate import quad
import matplotlib.pylab as pylab
from pprint import pprint
params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (10, 10),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
plt.style.use("science") # requires SciencePlots package

PI = np.pi
SPEED_OF_LIGHT = c.to(u.Mpc / u.day).value
ARCSEC2RAD = PI / 180 / 3600
FACTOR = ARCSEC2RAD**2 / SPEED_OF_LIGHT
JOINT_PRIOR = [
    [0, 1e4], # Ddt (Mpc)
    [0, 5], # r_ein (arcsec)
    [0, 0.3], # e
    [0, 1], # gamma_ext (external shear)
    [-PI, PI], # phi_ext (external shear angle)
    [0, PI], # phi
    [-1, 1], # x0 (arcsec)
    [-1, 1] # y0 (arcsec)
]

LABELS = [
    r"$D_{\Delta t}$",
    r"$\theta_E$",
    r"$e$",
    r"$\gamma_{\text{ext}}$",
    r"$\phi_{\text{ext}}$",
    r"$\phi$",
    r"$x_{0}$",
    r"$y_{0}$"
]


class QuadPseudoNIELensModel:
    """
    # We assume the coordinate center is roughly in the middle of the lens #
    # This is a pseud-elliptical model, it is only valid for small ellipticity (0, 0.3)#

    params: r_ein, e, theta_c, phi, x0, y0

    r_ein: Einstein radius (in arcseconds)
    e: ellipticity
    theta_c: Core radius (in arcseconds)
    phi: Orientation of the ellitpical profil (0 <= phi <= pi)
    (x0, y0): Position relative to coordinate center of the
        image plane grid (in arcseconds).

    x_image: horizontal coordinate of the 4 images (arcsec)
    y_image: vertical coordinate of the 4 images (arcsec)

    plate_scale: Plate scale of the image, used as uncertainty for
        likelihood
    """
    def __init__(self, x_image, y_image, plate_scale, kappa_ext=0, theta_c=0,
            prior=None):
        self.theta1 = x_image
        self.theta2 = y_image
        self.plate_scale = plate_scale
        self.theta_c = theta_c
        if prior is None:
            self.prior = [
                [0, 5], # r_ein
                [0, 0.3], # e
                [-1, 1], # gamma_1(external shear)
                [-1, 1], # gamma_2
                [0, PI], # phi
                [-1.5, 1.5], # x0
                [-1.5, 1.5] # y0
            ]
        else:
            self.prior = prior

    def external_shear_potential(self, gamma_ext, phi_ext):
        rho = np.hypot(self.theta1, self.theta2)
        varphi = np.arctan2(self.theta2, self.theta1)
        return 0.5 * gamma_ext * rho**2 * np.cos(2 * (varphi - phi_ext))

    def external_shear_deflection(self, gamma_ext, phi_ext):
        # see Meneghetti Lecture Scripts equation 3.83 (constant shear equation)
        alpha1 = gamma_ext * (self.theta1 * np.cos(phi_ext) + self.theta2 * np.sin(phi_ext))
        alpha2 = gamma_ext * (-self.theta1 * np.sin(phi_ext) + self.theta2 * np.cos(phi_ext))
        return alpha1, alpha2

    def rotated_and_shifted_coords(self, phi, x0, y0):
        """
        Important to shift then rotate, we move to the point of view of the
        lens before rotating the lens (rotation and translation are not commutative).
        """
        theta1 = self.theta1.copy() - x0
        theta2 = self.theta2.copy() - y0
        rho = np.hypot(theta1, theta2)
        varphi = np.arctan2(theta2, theta1) - phi
        theta1 = rho * np.cos(varphi)
        theta2 = rho * np.sin(varphi)
        return theta1, theta2

    def potential(self, theta1, theta2, r_ein, e): # arcsec^2
        return r_ein * np.sqrt(theta1**2/(1-e) + (1-e)*theta2**2 + self.theta_c**2)

    def deflection_angles(self, theta1, theta2, r_ein, e): # arcsec
        psi = np.sqrt(theta1**2/(1-e) + (1-e)*theta2**2 + self.theta_c**2)
        alpha1 = r_ein * (theta1 / psi) / (1 - e)
        alpha2 = r_ein * (1 - e) * (theta2 / psi)
        return alpha1, alpha2

    def backward_pass(self, params):
        r_ein, e, gamma_ext, phi_ext, phi, x0, y0 = params
        theta1, theta2 = self.rotated_and_shifted_coords(phi, x0, y0)
        alpha1, alpha2 = self.deflection_angles(theta1, theta2, r_ein, e)
        alpha1_sh, alpha2_sh = self.external_shear_deflection(gamma_ext, phi_ext)
        beta1 = theta1 - alpha1 - alpha1_sh
        beta2 = theta2 - alpha2 - alpha2_sh
        return beta1, beta2

    def joint_likelihood_forward(self, params):
        r_ein, e, gamma_ext, phi_ext, phi, x0, y0 = params
        theta1, theta2 = self.rotated_and_shifted_coords(phi, x0, y0)
        alpha = np.column_stack(self.deflection_angles(theta1, theta2, r_ein, e))
        alpha += np.column_stack(self.external_shear_deflection(gamma_ext, phi_ext))
        psi   = self.potential(theta1, theta2, r_ein, e)
        psi   += self.external_shear_potential(gamma_ext, phi_ext)
        return alpha, psi

    def time_delay(self, Ddt_params, i, j):
        Ddt = Ddt_params[0]
        params = Ddt_params[1:]
        alpha, psi = self.joint_likelihood_forward(params)
        norm = lambda x: np.dot(x, x)
        return Ddt * FACTOR * (norm(alpha[i])/2 - norm(alpha[j])/2 - psi[i] + psi[j])

    def loglikelihood(self, params):
        beta1, beta2 = self.backward_pass(params)
        dx_loglike = -0.5 * sum([(beta1[i] - beta1[j])**2 for i in range(3) for j in range(i + 1, 4)])
        dy_loglike = -0.5 * sum([(beta2[i] - beta2[j])**2 for i in range(3) for j in range(i + 1, 4)])
        return (dx_loglike + dy_loglike) / self.plate_scale**2

    def prior_transfom(self, x_uniform):
        return [(self.prior[i][1] - self.prior[i][0]) * x_uniform[i] + self.prior[i][0] for i in range(6)]

    def inference(self, nlive, nworkers=1):
        """
        Use Dynamic Nested Sampling to infer the posterior
        """
        with Pool(nworkers) as pool:
            sampler = DynamicNestedSampler(self.loglikelihood, self.prior_transfom, len(self.prior),
                pool=pool, nlive=nlive)
            sampler.run_nested()
            res = sampler.results
        return res


def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with as many quantiles as needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.atleast_1d(values)
    quantiles = np.atleast_1d(quantiles)
    if sample_weight is None:
         return np.percentile(values, list(100 * q))
    sample_weight = np.atleast_1d(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    cdf = np.cumsum(sample_weight)[:-1]  # compute CDF
    cdf /= cdf[-1]  # normalize CDF
    cdf = np.append(0, cdf)  # ensure proper span
    quantile_values = np.interp(quantiles, cdf, values)
    return quantile_values

def joint_prior_transform(params):
    return [(JOINT_PRIOR[i][1] - JOINT_PRIOR[i][0]) * params[i] + JOINT_PRIOR[i][0] for i in range(len(JOINT_PRIOR))]

def joint_loglikelihood_func(quad_model, delta_t, sigma_t):
    norm = lambda x: np.dot(x, x)
    A, B, C, D = list(range(4)) # to make life easier
    def joint_loglikelihood(params):
        Ddt = params[0]
        x = params[1:]
        alpha, psi = quad_model.joint_likelihood_forward(x)
        dts = [FACTOR * Ddt * (norm(alpha[i])/2 - norm(alpha[j])/2 - psi[i] + psi[j])
                for i,j in [(D, A), (A, B), (B, C)]]
        ll = quad_model.loglikelihood(x)
        ll += -0.5 * sum([(delta_t[i] - dts[i])**2/sigma_t[i]**2 for i in range(3)])
        return ll/15 # 15 is the temperature
    return joint_loglikelihood

def main(args=None):
    if args.results == "none":
        ndim = len(JOINT_PRIOR)
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_positions = pd.read_csv(args.image_positions)
        pprint(image_positions)
        # A=0, B=1, C=2, D=3
        x_image = image_positions["theta_x"].to_numpy()
        y_image = image_positions["theta_y"].to_numpy()
        quad_model = QuadPseudoNIELensModel(x_image, y_image, args.plate_scale)
        time_delays = pd.read_csv(args.time_delays)
        pprint(time_delays)

        # Expected: DA=0, DB=1, DC=2 (index)
        sigma_t = time_delays["sigma"].to_numpy() # units of days
        delta_t = time_delays["delta_t"].to_numpy()
        joint_loglikelihood = joint_loglikelihood_func(quad_model, delta_t, sigma_t)

        with Pool(args.nworkers) as pool:
            sampler = DynamicNestedSampler(joint_loglikelihood, joint_prior_transform,
                ndim, pool=pool, nlive=args.nlive, queue_size=args.nworkers)
            sampler.run_nested()
            res = sampler.results

        # save results
        with open(f"../results/joint_result_{date}.p", "wb") as f:
            pickle.dump(res, f)
    else: # we just want to plot the result from an older run
        with open(args.results, "rb") as f:
            res = pickle.load(f)
        ndim = res.samples.shape[1]

    if args.plot_results:
        # trace plot
        fig, axs = dyplot.traceplot(res, show_titles=True,
                    trace_cmap='plasma', connect=True,
                    connect_highlight=range(5),
                    labels=LABELS,
                          )
        fig.tight_layout(pad=2.0)
        fig.savefig("../figures/joint_inference_trace_plot.png", bbox_inches="tight")

        # corner points plot
        fig, axes = plt.subplots(ndim-1, ndim-1, figsize=(15, 15))
        axes.reshape([ndim-1, ndim-1])
        fg, ax = dyplot.cornerpoints(res, cmap='plasma', kde=False, fig=(fig, axes),
                    labels=LABELS
                    )
        fg.savefig("../figures/joint_inference_cornerpoints.png", bbox_inches="tight")

        # corner plot
        fig, axes = plt.subplots(ndim, ndim, figsize=(15, 15))
        axes.reshape([ndim, ndim])
        fg, ax = dyplot.cornerplot(res, fig=(fig, axes), color="b",
                    labels=LABELS,
                    show_titles=True)
        fg.savefig("../figures/joint_inference_corner_plot.png", bbox_inches="tight")

        #### marginalized posterior #####
        Ddt = res.samples[:, 0] # histogram of Ddt
        weights = np.exp(res['logwt'] - res['logz'][-1]) # posterior probability (Bayes theorem)

        # eliminate outliers with low and high + estimate confidance interval
        low, fifth, median, ninety_fifth, high = weighted_quantile(Ddt, [0.0001, 0.05, 0.5, 0.95, 0.9999], weights)
        error_plus = ninety_fifth - median # error estimate with 5th percentile
        error_minus = median - fifth       # error estimate with 95th percentile
        good = (Ddt > low) & (Ddt < high) # remove outliers
        Ddt = Ddt[good]
        weights = weights[good]

        plt.figure(figsize=(8, 8))
        plt.hist(Ddt, bins=100, weights=weights);
        plt.title(r"$D_{\Delta t}$=%.2f$^{+%.2f}_{-%.2f}$" % (median, error_plus, error_minus))
        plt.xlabel(r"$D_{\Delta t}$");
        plt.savefig("../figures/marginalized_posterior_Ddt.png")

        # assume a flat LambdCDM model (with negligible radiation)

        # We need to model kappa_ext for this step
        def integrand(z):
            return 1/np.sqrt(args.omega_m * (1 + z)**3 + args.omega_l)
        Dd  = quad(integrand, 0, args.z_lens)[0]/(1 + args.z_lens)
        Ds  = quad(integrand, 0, args.z_source)[0]/(1 + args.z_source)
        Dds = quad(integrand, args.z_lens, args.z_source)[0]/(1 + args.z_source)
        factor = (1 + args.z_lens) * Ds * Dd / Dds
        H0  = (c * factor / Ddt / u.Mpc).to(u.km / u.s / u.Mpc).value

        plt.figure(figsize=(8, 8))
        fifth, median, ninety_fifth = weighted_quantile(H0, [0.05, 0.5, 0.95], weights)
        error_plus = ninety_fifth - median
        error_minus = median - fifth
        plt.hist(H0, bins=100, weights=weights)
        plt.title(r"$H_0$=%.2f$^{+%.2f}_{-%.2f}$" % (median, error_plus, error_minus))
        plt.xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
        plt.savefig("../figures/marginalized_posterior_H0.png")

if __name__ == '__main__':
    # assumes script is ran from script folder!
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--time_delays", default="../data/time_delays.csv", type=str)
    parser.add_argument("--image_positions", default="../data/image_positions.csv", type=str)
    parser.add_argument("--results", default="none", type=str, help="Path to a result pickled file (skips the sampling)")
    parser.add_argument("--nworkers", default=1, type=int)
    parser.add_argument("--nlive", default=5000, type=int,
                                help="Number of live points in Nested Sampling")
    parser.add_argument("--plate_scale", default=0.05, type=float,
                                help="Plate scale of camera") # Actually half of it TODO formalize this
    parser.add_argument("--plot_results", action="store_true")
    parser.add_argument("--z_lens", default=0.295, type=float, help="Redshift of lens, default for RXJ1131-1231")
    parser.add_argument("--z_source", default=0.654, type=float, help="Redshift of source, default for RXJ1131-1231")
    parser.add_argument("--omega_m", default=0.29, type=float, help="Matter density parameter")
    parser.add_argument("--omega_l", default=0.71, type=float, help="Dark energy density parameter")
    args = parser.parse_args()
    main(args)
