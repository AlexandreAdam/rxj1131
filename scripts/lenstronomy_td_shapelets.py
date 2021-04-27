import numpy as np
import time
import astropy.io.fits as fits

import astropy.units as u
from astropy.wcs import WCS
from lenstronomy.Workflow.fitting_sequence import FittingSequence

from lenstronomy.Data.imaging_data import ImageData
import matplotlib.pylab as pylab
import pandas as pd

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 10),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

SNR = 5.098 # known from original fits file
sky_zero_point = -21.1 # mag
photflam = 6.9548454E-20 # inverse sensitivity ergs/cm2/Ang/electron
galfit_data = fits.open("../data/blocks1.fits")
im = galfit_data[3].data
wcs= WCS(galfit_data[1].header)
psf = fits.open("../data/rxj1131_psf1.fits")[0].data
psf2 = fits.open("../data/tiny_psf00_psf.fits")[0].data
im[im < 0] = 0
image_principales = np.loadtxt("../data/images_positions.txt")[:4] ## B, A, C, D, G, S
f_vega = 8.60e-10


# coordinate system in arcseconds, centered
N, M = im.shape
y = np.arange(N) #- N//2. + 0.5 * ((N + 1) % 2)
x = np.arange(M) #- M//2. + 0.5 * ((M + 1) % 2)
x, y = np.meshgrid(x, y)
lens_sky = wcs.pixel_to_world(x.ravel(), y.ravel())
x_center = (lens_sky.ra[M//2]).to(u.arcsec)
y_center = (lens_sky.dec[M*N//2]).to(u.arcsec)

x_image = image_principales[:, 0][[1, 0, 2, 3]] # A, B, C, D
y_image = image_principales[:, 1][[1, 0, 2, 3]]
positions = wcs.pixel_to_world(x_image, y_image)
x_image = (positions.ra - lens_sky.ra.max()).to(u.arcsec).value
y_image = (positions.dec - lens_sky.dec.min()).to(u.arcsec).value
image_position = pd.DataFrame(np.column_stack([x_image, y_image]), columns=["theta_x", "theta_y"])

# time relative to first image (A) in order AB, Ac, and AD
delta_t = pd.DataFrame(np.array([[0.7, -0.4, 91.72], [1.4, 2, 1.5]]).T, columns=["delta_t", "sigma"])
delta_t.to_csv("../data/time_delays.csv")

# data specifics for the lens image
sigma_bkg = .05  #  background noise per pixel (Gaussian)
exp_time = 1.  #  exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
delta_pix_x = np.abs(lens_sky.ra[0] - lens_sky.ra[1]).to(u.arcsec).value
delta_pix_y = np.abs(lens_sky.dec[0] - lens_sky.dec[M]).to(u.arcsec).value
kwargs_data = {"image_data": im, "exposure_time": exp_time,
               "background_rms": sigma_bkg,
               "transform_pix2angle":np.array([[-delta_pix_x, 0], [0, delta_pix_y]]),
               "ra_at_xy_0": 0,
               "dec_at_xy_0": 0
               }
data_class = ImageData(**kwargs_data)

x = lens_sky.ra.to(u.arcsec) - lens_sky.ra.max()
y = lens_sky.dec.to(u.arcsec) - lens_sky.dec.min()
x_center = (x[M//2]).to(u.arcsec).value
y_center = (y[M*N//2]).to(u.arcsec).value
grid = np.column_stack([x, y])
ee = 0.611 # encircled energy, source https://www.stsci.edu/hst/instrumentation/acs/data-analysis/aperture-corrections
# collect 4 pixels closest to the image position (correspond to 0.1 arcsec since pixelscale is 0.05 arcsec)
f = []
A = []
for i in range(4):
    pos = image_position.to_numpy()[i]
    diff = np.square(pos - grid.value)
    dist = np.einsum("ij -> i", diff)
    indexes = np.argpartition(dist, kth=4)[:4]
    A.append(np.sum(im.ravel()[indexes]))
    f.append(photflam * np.sum(im.ravel()[indexes])/ee) # see e.g. https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints
mag = np.array([-2.5 * np.log10(_f/f_vega) for _f in f])


kwargs_ps = [{'ra_image': x_image, 'dec_image': y_image}]
#                            'point_amp': np.abs(mag)*1000}]
point_source_list = ['LENSED_POSITION']

kwargs_psf = {'psf_type': "PIXEL", "kernel_point_source":psf, "pixel_size": delta_pix_x}

fixed_lens = [{}, {'ra_0': x_center, 'dec_0': y_center}]
fixed_source = [{"n_max": 7}]
fixed_ps = [{}]
# initial guess of non-linear parameters, we chose different starting parameters than the truth #
kwargs_lens_init = [{'theta_E': 1.2, 'e1': 0, 'e2': 0, 'gamma': 2., 'center_x': x_center, 'center_y': y_center},
    {'gamma1': 0, 'gamma2': 0}]
# kwargs_source_init = [{'R_sersic': 0.03, 'n_sersic': 4., 'e1': 0, 'e2': 0, 'center_x': x_center, 'center_y': y_center}]
kwargs_source_init = [{'amp': 1, 'beta': 0.2, 'center_x': x_center, 'center_y': y_center}]
kwargs_ps_init = [{'ra_image': x_image, 'dec_image': y_image}]

# initial spread in parameter estimation #
kwargs_lens_sigma = [{'theta_E': 0.3, 'e1': 0.2, 'e2': 0.2, 'gamma': .2, 'center_x': 0.1, 'center_y': 0.1},
    {'gamma1': 0.1, 'gamma2': 0.1}]
# kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': .5, 'center_x': .1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2}]
kwargs_source_sigma = [{'amp': 0.1, 'beta': 0.05, 'center_x': 0.2, 'center_y': 0.2}]
kwargs_ps_sigma = [{'ra_image': [0.02] * 4, 'dec_image': [0.02] * 4}]

# hard bound lower limit in parameter space #
kwargs_lower_lens = [{'theta_E': 0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'center_x': x.min().value, 'center_y': y.min().value},
    {'gamma1': -0.5, 'gamma2': -0.5}]
# kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': .5, 'e1': -0.5, 'e2': -0.5, 'center_x': x.min().value, 'center_y': y.min().value}]
kwargs_lower_source = [{'amp': 0, 'beta': 0.05, 'center_x': x.min().value, 'center_y': y.min().value}]
kwargs_lower_ps = [{'ra_image': -10 * np.ones_like(x_image), 'dec_image': -10 * np.ones_like(y_image)}]

# hard bound upper limit in parameter space #
kwargs_upper_lens = [{'theta_E': 10, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'center_x': x.max().value, 'center_y': y.max().value},
    {'gamma1': 0.5, 'gamma2': 0.5}]
# kwargs_upper_source = [{'R_sersic': 10, 'n_sersic': 5., 'e1': 0.5, 'e2': 0.5, 'center_x': x.max().value, 'center_y': y.max().value}]
kwargs_upper_source = [{'amp': 10, 'beta': 5, 'center_x': x.max().value, 'center_y': y.max().value}]
kwargs_upper_ps = [{'ra_image': 10 * np.ones_like(x_image), 'dec_image': 10 * np.ones_like(y_image)}]


lens_params = [kwargs_lens_init, kwargs_lens_sigma, fixed_lens, kwargs_lower_lens, kwargs_upper_lens]
source_params = [kwargs_source_init, kwargs_source_sigma, fixed_source, kwargs_lower_source, kwargs_upper_source]
ps_params = [kwargs_ps_init, kwargs_ps_sigma, fixed_ps, kwargs_lower_ps, kwargs_upper_ps]

fixed_special = {}
kwargs_special_init = {}
kwargs_special_sigma = {}
kwargs_lower_special = {}
kwargs_upper_special = {}

source_size_arcsec = 0.001
# If you want to keep the source size fixed during the fitting, outcomment the line below.
# fixed_special['source_size'] = source_size_arcsec
kwargs_special_init['source_size'] = source_size_arcsec
kwargs_special_sigma['source_size'] = source_size_arcsec
kwargs_lower_special['source_size'] = 0.0001
kwargs_upper_special['source_size'] = 1

kwargs_special_init['D_dt']= 5000
kwargs_special_sigma['D_dt'] = 10000
kwargs_lower_special['D_dt'] = 0
kwargs_upper_special['D_dt'] =  10000
special_params = [kwargs_special_init, kwargs_special_sigma, fixed_special, kwargs_lower_special, kwargs_upper_special]


# source_model_list = ['SERSIC_ELLIPSE']
source_model_list = ["SHAPELETS"]
lens_model_list = ['SPEP', 'SHEAR']

kwargs_params = {'lens_model': lens_params,
                'source_model': source_params,
#                 'lens_light_model': lens_light_params,
                'point_source_model': ps_params,
                'special': special_params}

kwargs_model = {'lens_model_list': lens_model_list,
#                  'lens_light_model_list': lens_light_model_list,
                'source_light_model_list': source_model_list,
                'point_source_model_list': point_source_list
                 }

plate_scale = (delta_pix_x + delta_pix_y)/2


# numerical options and fitting sequences


num_source_model = len(source_model_list)

kwargs_constraints = {'joint_source_with_point_source': [[0, 0, ["center_x", "center_y"]]],
                      'num_point_source_list': [4],
                      'solver_type': 'PROFILE_SHEAR',  # 'PROFILE', 'PROFILE_SHEAR', 'ELLIPSE', 'CENTER'
                      'Ddt_sampling': True,
                              }

kwargs_likelihood = {'check_bounds': True,
                     'force_no_add_image': False,
                     'source_marg': True,
                     'check_matched_source_position': True,
                     'source_position_tolerance': plate_scale/10,
                     'time_delay_likelihood': True,
                     'image_position_likelihood': True, # evaluate point source likelihood given the measured image positions
                     'image_position_uncertainty': plate_scale,
                    }
kwargs_numerics = {'supersampling_factor': 1}
image_band = [kwargs_data, kwargs_psf, kwargs_numerics]
multi_band_list = [image_band]
kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'single-band',
                    'time_delays_measured': delta_t["delta_t"].to_numpy(),
                    'time_delays_uncertainties': delta_t["sigma"].to_numpy(),
                    'ra_image_list': [x_image], 'dec_image_list': [y_image]
                    }

fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, mpi=False)

fitting_kwargs_list = [['PSO', {'sigma_scale': .3, 'n_particles': 10, 'n_iterations': 20, "threadCount": 5}],
        ['MCMC', {'n_burn': 10, 'n_run': 20, 'walkerRatio': 10, 'sigma_scale': .2, "threadCount":5}]
]

start_time = time.time()
chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
kwargs_result = fitting_seq.best_fit()
end_time = time.time()
print(end_time - start_time, 'total time needed for computation')
print('============ CONGRATULATION, YOUR JOB WAS SUCCESSFUL ================ ')



