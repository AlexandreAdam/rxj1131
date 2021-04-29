"""
This script fit spline and regression difference to the data. This original fit will be used to create the generative noise model.
You can tune the spline and regdiff parameters from the config file.
"""
import argparse as ap
import importlib
import logging
import os
import sys

import numpy as np

import pycs3.gen.lc_func
import pycs3.gen.mrg
import pycs3.gen.stat
import pycs3.gen.util
import pycs3.pipe.pipe_utils as ut
import pycs3.regdiff.rslc

loggerformat='%(levelname)s: %(message)s'
logging.basicConfig(format=loggerformat,level=logging.INFO)


def main(lensname, dataname, work_dir='./'):
    sys.path.append(work_dir + "config/")
    config = importlib.import_module("config_" + lensname + "_" + dataname)

    figure_directory = config.figure_directory + "spline_and_residuals_plots/"
    if not os.path.isdir(figure_directory):
        os.mkdir(figure_directory)

    for i, lc in enumerate(config.lcs_label):
        print("I will apply a initial shift of : %2.4f days for %s" % (
            config.timeshifts[i], config.lcs_label[i]))

    # Do the optimisation with the splines
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
        raise RuntimeError("I don't know your microlensing type. Choose 'polyml' or 'spml'.")
    chi2 = np.zeros((len(config.knotstep), len(ml_param)))
    dof = np.zeros((len(config.knotstep), len(ml_param)))

    for i, kn in enumerate(config.knotstep):
        for j, ml in enumerate(ml_param):
            print(("ML param", j, ml))
            lcs = pycs3.gen.util.readpickle(config.data)
            if config.magshift is None :
                magsft = [-np.median(lc.getmags()) for lc in lcs]
            else :
                magsft = config.magshift
            pycs3.gen.lc_func.applyshifts(lcs, config.timeshifts, magsft) #remove median and set the time shift to the initial guess
            if ml != 0:
                config.attachml(lcs, ml)  # add microlensing

            spline = config.spl1(lcs, kn=kn)
            pycs3.gen.mrg.colourise(lcs)
            rls = pycs3.gen.stat.subtract(lcs, spline)
            chi2[i, j] = pycs3.gen.stat.compute_chi2(rls, kn, ml)
            dof[i, j] = pycs3.gen.stat.compute_dof_spline(rls, kn, ml)

            if config.display:
                pycs3.gen.lc_func.display(lcs, [spline], showlegend=True, showdelays=True, filename="screen")
                pycs3.gen.stat.plotresiduals([rls])
            else:
                pycs3.gen.lc_func.display(lcs, [spline], showlegend=True, showdelays=True,
                                          filename=figure_directory + "spline_fit_ks%i_%s%i.png" % (kn, string_ML, ml))
                pycs3.gen.stat.plotresiduals([rls], filename=figure_directory + "residual_fit_ks%i_%s%i.png" % (
                    kn, string_ML, ml))

            # and write data, again
            if not os.path.isdir(config.lens_directory + config.combkw[i, j]):
                os.mkdir(config.lens_directory + config.combkw[i, j])

            pycs3.gen.util.writepickle((lcs, spline), config.lens_directory + '%s/initopt_%s_ks%i_%s%i.pkl' % (
                config.combkw[i, j], dataname, kn, string_ML, ml))

    #--- REGDIFF ---#
    # DO the optimisation with regdiff as well, just to have an idea, this the first point of the grid !
    lcs = pycs3.gen.util.readpickle(config.data)
    pycs3.gen.mrg.colourise(lcs)
    pycs3.gen.lc_func.applyshifts(lcs, config.timeshifts, [-np.median(lc.getmags()) for lc in lcs])

    for ind, l in enumerate(lcs):
        l.shiftmag(ind * 0.1)

    if config.use_preselected_regdiff:
        kwargs_optimiser_simoptfct = ut.get_keyword_regdiff_from_file(config.preselection_file)
        regdiff_param_kw = ut.read_preselected_regdiffparamskw(config.preselection_file)
    else:
        kwargs_optimiser_simoptfct = ut.get_keyword_regdiff(config.pointdensity, config.covkernel, config.pow,
                                                            config.amp, config.scale, config.errscale)
        regdiff_param_kw = ut.generate_regdiffparamskw(config.pointdensity, config.covkernel, config.pow, config.amp,
                                                       config.scale, config.errscale)

    for i, k in enumerate(kwargs_optimiser_simoptfct):
        myrslcs = [pycs3.regdiff.rslc.factory(l, pd=k['pointdensity'], covkernel=k['covkernel'],
                                              pow=k['pow'], errscale=k['errscale']) for
                   l in lcs]

        if config.display:
            pycs3.gen.lc_func.display(lcs, myrslcs)
        pycs3.gen.lc_func.display(lcs, myrslcs, showdelays=True,
                                  filename=figure_directory + "regdiff_fit%s.png" % regdiff_param_kw[i])

        for ind, l in enumerate(lcs):
            l.shiftmag(-ind * 0.1)

        config.regdiff(lcs, **kwargs_optimiser_simoptfct[i])

        if config.display:
            pycs3.gen.lc_func.display(lcs, showlegend=False, showdelays=True)
        pycs3.gen.lc_func.display(lcs, showlegend=False, showdelays=True,
                                  filename=figure_directory + "regdiff_optimized_fit%s.png" % regdiff_param_kw[i])
        if not os.path.isdir(config.lens_directory + 'regdiff_fitting'):
            os.mkdir(config.lens_directory + 'regdiff_fitting')
        pycs3.gen.util.writepickle(lcs,
                                   config.lens_directory + 'regdiff_fitting/initopt_regdiff%s.pkl' % regdiff_param_kw[
                                       i])

    # Write the report :
    print("Report will be writen in " + config.lens_directory + 'report/report_fitting.txt')

    f = open(config.lens_directory + 'report/report_fitting.txt', 'w')
    f.write('Measured time shift after fitting the splines : \n')
    f.write('------------------------------------------------\n')

    for i, kn in enumerate(config.knotstep):
        f.write('knotstep : %i' % kn + '\n')
        f.write('\n')
        for j, ml in enumerate(ml_param):
            lcs, spline = pycs3.gen.util.readpickle(config.lens_directory + '%s/initopt_%s_ks%i_%s%i.pkl' % (
                config.combkw[i, j], dataname, kn, string_ML, ml), verbose=False)
            delay_pair, delay_name = ut.getdelays(lcs)
            f.write('Micro-lensing %s = %i' % (string_ML, ml) + "     Delays are " + str(delay_pair) + " for pairs " +
                    str(delay_name) + '. Chi2 Red : %2.5f ' % chi2[i, j] + ' DoF : %i \n' % dof[i, j])

        f.write('\n')

    f.write('------------------------------------------------\n')
    f.write('Measured time shift after fitting with regdiff : \n')
    f.write('\n')

    for i, k in enumerate(kwargs_optimiser_simoptfct):
        lcs = pycs3.gen.util.readpickle(
            config.lens_directory + 'regdiff_fitting/initopt_regdiff%s.pkl' % regdiff_param_kw[i], verbose=False)
        delay_pair, delay_name = ut.getdelays(lcs)
        f.write('Regdiff : ' + "     Delays are " + str(delay_pair) + " for pairs " + str(delay_name) + '\n')
        f.write('------------------------------------------------\n')

    starting_point = []
    for i in range(len(config.timeshifts)):
        for j in range(len(config.timeshifts)):
            if i >= j:
                continue
            else:
                starting_point.append(config.timeshifts[j] - config.timeshifts[i])

    f.write('Starting point used : ' + str(starting_point) + " for pairs " + str(delay_name) + '\n')
    f.close()


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Fit spline and regdiff on the data.",
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
