"""
This script will find the generative noise model parameters that create mock lightcurves matching the data properties in term of gaussian and correlated noise
You can also provide directly the correct parameters in the config file. In this case I will just generate the python files to proceed to the step 3b and 3c
and skip the optimisation
"""
import matplotlib
matplotlib.use('Agg')
import os
import pycs3.sim.draw
import pycs3.gen.util
import pycs3.gen.splml
import pycs3.sim.twk as twk
import pycs3.spl.topopt
import pycs3.pipe.optimiser
import pycs3.pipe.pipe_utils as ut
import sys
import argparse as ap
import importlib
import logging
loggerformat='PID %(process)06d | %(asctime)s | %(levelname)s: %(name)s(%(funcName)s): %(message)s'
logging.basicConfig(format=loggerformat,level=logging.INFO)


def run_DIC(lcs, spline, fit_vector, kn, ml, optim_directory, config_file, stream, tolerance=0.75):
    config = importlib.import_module(config_file)
    pycs3.sim.draw.saveresiduals(lcs, spline)
    print("I'll try to recover these parameters :", fit_vector)
    dic_opt = pycs3.pipe.optimiser.DicOptimiser(lcs, fit_vector, spline, config.attachml, ml, knotstep=kn,
                                                savedirectory=optim_directory,
                                                recompute_spline=True, max_core=config.max_core,
                                                n_curve_stat=config.n_curve_stat,
                                                shotnoise=config.shotnoise_type, tweakml_type=config.tweakml_type,
                                                tweakml_name=config.tweakml_name, display=config.display, verbose=False,
                                                correction_PS_residuals=True, max_iter=config.max_iter, tolerance=tolerance,
                                                theta_init=None)

    chain = dic_opt.optimise()
    dic_opt.analyse_plot_results()
    chi2, B_best = dic_opt.get_best_param()
    A = dic_opt.A_correction
    dic_opt.reset_report()
    dic_opt.report()

    if dic_opt.success:
        print("I succeeded finding a parameter falling in the %2.2f sigma from the original lightcurve." % tolerance)

    else:
        print("I didn't find a parameter that falls in the %2.2f sigma from the original lightcurve." % tolerance)
        print("I then choose the best one... but be carefull ! ")

    for k in range(len(lcs)):
        def tweakml_PS_NUMBER(lcs, spline):
            return twk.tweakml_PS(lcs, spline, B_PARAM, f_min=1 / 300.0, psplot=False, verbose=False,
                                  interpolation='linear', A_correction=A_PARAM)

        ut.write_func_append(tweakml_PS_NUMBER, stream,
                             B_PARAM=str(B_best[k][0]), NUMBER=str(k + 1), A_PARAM=str(A[k]))


def main(lensname, dataname, work_dir='./'):
    sys.path.append(work_dir + "config/")
    config_file = "config_" + lensname + "_" + dataname
    config = importlib.import_module(config_file)
    tweakml_plot_dir = config.figure_directory + 'tweakml_plots/'
    optim_directory = tweakml_plot_dir + 'twk_optim_%s_%s/' % (config.optimiser, config.tweakml_name)

    if not os.path.isdir(tweakml_plot_dir):
        os.mkdir(tweakml_plot_dir)

    if config.mltype == "splml":
        if config.forcen:
            ml_param = config.nmlspl
            string_ML = "nmlspl"
        else:
            ml_param = config.mlknotsteps
            string_ML = "knml"
    elif config.mltype == "polyml":
        ml_param = config.degree
        string_ML = "deg"
    else:
        raise RuntimeError('I dont know your microlensing type. Choose "polyml" or "spml".')

    for i, kn in enumerate(config.knotstep):
        for j, ml in enumerate(ml_param):
            f = open(config.lens_directory + config.combkw[i, j] + '/tweakml_' + config.tweakml_name + '.py', 'w+')
            f.write('import pycs3 \n')
            f.write('from pycs3.sim import twk as twk \n')
            lcs, spline = pycs3.gen.util.readpickle(config.lens_directory + '%s/initopt_%s_ks%i_%s%i.pkl' % (
            config.combkw[i, j], dataname, kn, string_ML, ml))
            fit_vector = pycs3.pipe.optimiser.get_fit_vector(lcs, spline)  # we get the target parameter now
            if not os.path.isdir(optim_directory):
                os.mkdir(optim_directory)

            # We need spline microlensing for tweaking the curve, if it is not the case we change it here to a flat spline that can be tweaked.
            # the resulting mock light curve will have no ML anyway, we will attach it the ML defined in your config file before optimisation.
            polyml = False
            for k, l in enumerate(lcs):
                if l.ml == None:
                    print(
                        'I dont have ml, I have to introduce minimal extrinsic variation to generate the mocks. Otherwise I have nothing to modulate.')
                    pycs3.gen.splml.addtolc(l, n=2)
                elif l.ml.mltype == 'poly':
                    polyml = True
                    print(
                        'I have polyml and it can not be tweaked. I will replace it with a flat spline just for the mock light curve generation.')
                    l.rmml()

            if polyml:
                spline = pycs3.spl.topopt.opt_fine(lcs, nit=5, knotstep=kn,
                                                   verbose=False, bokeps=kn / 3.0,
                                                   stabext=100)  # we replace the spline optimised with poly ml by one without ml
                for l in lcs:
                    pycs3.gen.splml.addtolc(l, n=2)
                pycs3.gen.util.writepickle((lcs, spline),
                                           config.lens_directory + '%s/initopt_%s_ks%i_%s%i_generative_polyml.pkl' % (
                                           config.combkw[i, j], dataname, kn, string_ML, ml))

            # Starting to write tweakml function depending on tweak_ml_type :
            if config.tweakml_type == 'colored_noise':
                if config.shotnoise_type == None:
                    print('WARNING : you are using no shotnoise with the colored noise ! That will probably not work.')

                if config.find_tweak_ml_param == True:
                    raise NotImplementedError(
                        "I am not supporting automatic optimisation for colored_noise yet. You should provide your generative noise model parameter yourself or use PS_from_residuals.")
                else:
                    print("Colored noise : I will add the beta and sigma that you gave in input.")
                    for k in range(len(lcs)):
                        def tweakml_colored_NUMBER(lcs, spline):
                            return twk.tweakml(lcs, spline, beta=BETA, sigma=SIGMA, fmin=1.0 / 500.0, fmax=0.2,
                                               psplot=False)

                        ut.write_func_append(tweakml_colored_NUMBER, f,
                                             BETA=str(config.colored_noise_param[k][0]),
                                             SIGMA=str(config.colored_noise_param[k][1]), NUMBER=str(k + 1))

                list_string = 'tweakml_list = ['
                for k in range(len(lcs)):
                    list_string += 'tweakml_colored_' + str(k + 1) + ','
                list_string += ']'
                f.write('\n')
                f.write(list_string)


            elif config.tweakml_type == 'PS_from_residuals':
                if config.shotnoise_type != None:
                    print('If you use PS_from_residuals, the shotnoise should be set to None. I will do it for you !')
                    config.shotnoise_type = None

                if config.find_tweak_ml_param == True:
                    if config.optimiser == 'DIC':
                        run_DIC(lcs, spline, fit_vector, kn, ml, optim_directory, config_file, f)
                    else:
                        raise RuntimeError('I do not recognise your optimiser, please use DIC with PS_from_residuals')

                else:
                    print("Noise from Power Spectrum of the data : I use PS_param that you gave in input.")
                    for k in range(len(lcs)):
                        def tweakml_PS_NUMBER(lcs, spline):
                            return twk.tweakml_PS(lcs, spline, B_PARAM, f_min=1 / 300.0, psplot=False, verbose=False,
                                                  interpolation='linear')

                        ut.write_func_append(tweakml_PS_NUMBER, f,
                                             B_PARAM=str(config.PS_param_B[k]), NUMBER=str(k + 1))

                list_string = 'tweakml_list = ['
                for k in range(len(lcs)):
                    list_string += 'tweakml_PS_' + str(k + 1) + ','
                list_string += ']'
                f.write('\n')
                f.write(list_string)

            else:
                raise RuntimeError("I don't know your tweak_ml_type, please use colored_noise or PS_form_residuals.")
            f.close()
            # rename the file :
            files = [file for file in os.listdir(optim_directory)
                     if os.path.isfile(os.path.join(optim_directory, file)) and (string_ML not in file)]

            for file in files:
                prefix, extension = file.split('.')
                os.rename(os.path.join(optim_directory, file),
                          os.path.join(optim_directory, prefix + "_kn%i_%s%i." % (kn, string_ML, ml) + extension))


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Find the noise parameter to reproduce the data.",
                               formatter_class=ap.RawTextHelpFormatter)
    help_lensname = "name of the lens to process"
    help_dataname = "name of the data set to process (Euler, SMARTS, ... )"
    help_work_dir = "name of the working directory"
    parser.add_argument(dest='lensname', type=str,
                        metavar='lens_name', action='store',
                        help=help_lensname)
    parser.add_argument(dest='dataname', type=str,
                        metavar='dataname', action='store',
                        help=help_dataname)
    parser.add_argument('--dir', dest='work_dir', type=str,
                        metavar='', action='store', default='./',
                        help=help_work_dir)
    args = parser.parse_args()
    main(args.lensname, args.dataname, work_dir=args.work_dir)
