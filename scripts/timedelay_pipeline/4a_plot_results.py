"""
This script is going through your optimised mock lightcurves and measure the time delay estimates (i.e. the central value + error bars)
"""
import matplotlib.style
matplotlib.style.use('classic')
import matplotlib.pyplot as plt
import pycs3.sim.run
import pycs3.sim.plot
import pycs3.gen.util
import pycs3.tdcomb.plot
import pycs3.tdcomb.comb
import sys
import os
import importlib
import argparse as ap
import pickle as pkl
import logging
loggerformat='%(message)s'
logging.basicConfig(format=loggerformat,level=logging.INFO)


def main(lensname, dataname, work_dir='./'):
    sys.path.append(work_dir + "config/")
    config = importlib.import_module("config_" + lensname + "_" + dataname)

    regdiff_dir = os.path.join(config.lens_directory, "regdiff_outputs/")
    figure_directory = config.figure_directory + "final_results/"
    if not os.path.isdir(figure_directory):
        os.mkdir(figure_directory)
    if not os.path.isdir(regdiff_dir):
        os.mkdir(regdiff_dir)

    binclip = True  # be careful this could be dangerous, make sure you kick out only outlier otherwise errror bar will be underestimated. TODO : add warning if you exceed a certain percentage of rejected curves
    binclipr = 40.0  # rather conservative value
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

    for a, kn in enumerate(config.knotstep):
        for b, ml in enumerate(ml_param):
            for o, opt in enumerate(config.optset):

                # simulations
                toplot = []
                simres = [pycs3.sim.run.collect(
                    config.lens_directory + config.combkw[a, b] + '/sims_%s_opt_%s' % (config.simset_mock, opt),
                    'blue', dataname + "_" + config.combkw[a, b])]

                # Copies :
                if config.simoptfctkw == "regdiff":
                    kwargs = config.kwargs_optimiser_simoptfct[o]
                    dir_link = pkl.load(
                        open(os.path.join(config.lens_directory, 'regdiff_copies_link_%s.pkl' % kwargs['name']), 'rb'))

                    if a == 0 and b == 0:
                        regdiff_copie_dir = os.path.join(regdiff_dir, "copies/")
                        if not os.path.isdir(regdiff_copie_dir):
                            os.mkdir(regdiff_copie_dir)
                        copiesres = [pycs3.sim.run.collect(dir_link, 'blue',
                                                           dataname + "_regdiff_%s" % kwargs['name'])]
                        pycs3.sim.plot.hists(copiesres, r=50.0, nbins=100, dataout=True, usemedian=True,
                                             filename=figure_directory + 'delay_hist_%i-%i_sims_%s_opt_%s.png' % (
                                                 kn, ml, config.simset_copy, opt),
                                             outdir=regdiff_copie_dir)

                    regdiff_mocks_dir = os.path.join(regdiff_dir, "mocks_knt%i_mlknt%i/" % (kn, ml))
                    if not os.path.isdir(regdiff_mocks_dir):
                        os.mkdir(regdiff_mocks_dir)
                    pycs3.sim.plot.measvstrue(simres, r=2 * config.truetsr, nbins=1, plotpoints=True,
                                              ploterrorbars=True, sidebyside=True,
                                              errorrange=5., binclip=binclip, binclipr=binclipr, dataout=True,
                                              figsize=(12, 9),
                                              filename=figure_directory + 'deviation_hist_%i-%i_sims_%s_opt_%s.png' % (
                                                  kn, ml, config.simset_copy, opt),
                                              outdir=regdiff_mocks_dir)

                    cscontainer = pycs3.tdcomb.comb.CScontainer("Regdiff kn%s %s%s"%(kn, string_ML, ml), knots=str(kn), ml=str(ml),
                                                              result_file_delays=regdiff_copie_dir + 'sims_%s_opt_%s_delays.pkl' % (
                                                                  config.simset_copy, opt),
                                                              result_file_errorbars=regdiff_mocks_dir + 'sims_%s_opt_%s_errorbars.pkl' % (
                                                                  config.simset_mock, opt))

                elif config.simoptfctkw == "spl1":
                    copiesres = [pycs3.sim.run.collect(
                        config.lens_directory + config.combkw[a, b] + '/sims_%s_opt_%s' % (config.simset_copy, opt),
                        'blue',
                        dataname + "_" + config.combkw[a, b])]

                    pycs3.sim.plot.hists(copiesres, r=50.0, nbins=100, dataout=True, usemedian=True,
                                         filename=figure_directory + 'delay_hist_%i-%i_sims_%s_opt_%s.png' % (
                                             kn, ml, config.simset_copy, opt),
                                         outdir=config.lens_directory + config.combkw[a, b] + '/sims_%s_opt_%s/' % (
                                             config.simset_copy, opt))
                    pycs3.sim.plot.measvstrue(simres, r=2 * config.truetsr, nbins=1, plotpoints=True,
                                              ploterrorbars=True, sidebyside=True,
                                              errorrange=10., binclip=binclip, binclipr=binclipr, dataout=True,
                                              figsize=(12, 9),
                                              filename=figure_directory + 'deviation_hist_%i-%i_sims_%s_opt_%s.png' % (
                                                  kn, ml, config.simset_copy, opt),
                                              outdir=config.lens_directory + config.combkw[
                                                  a, b] + '/sims_%s_opt_%s/' % (
                                                         config.simset_copy, opt))

                    cscontainer = pycs3.tdcomb.comb.CScontainer("Spline kn%s %s%s"%(kn, string_ML,ml), knots=str(kn), ml=str(ml),
                                                              result_file_delays=os.path.join(
                                                                  config.lens_directory + config.combkw[
                                                                      a, b] + '/sims_%s_opt_%s/' % (
                                                                      config.simset_copy, opt) +
                                                                  'sims_%s_opt_%s_delays.pkl' % (
                                                                      config.simset_copy, opt)),
                                                              result_file_errorbars=config.lens_directory +
                                                                                    config.combkw[
                                                                                        a, b] + '/sims_%s_opt_%s/' % (
                                                                                    config.simset_copy,
                                                                                    opt) + 'sims_%s_opt_%s_errorbars.pkl' % (
                                                                                        config.simset_mock, opt))
                    print(cscontainer.result_file_delays)

                if config.display:
                    plt.show()

                toplot.append(pycs3.tdcomb.comb.getresults(cscontainer, useintrinsic=False))

                text = [(0.12, 0.9, r"$\mathrm{" + config.full_lensname + "}$", {"fontsize": 22})]

                pycs3.tdcomb.plot.delayplot(toplot, rplot=10.0, displaytext=True, text=text, showlegend=False,
                                          filename=figure_directory + "fig_delays_%i-%i_%s_%s.png" % (
                                              kn, ml, config.simset_mock, opt), autoobj=config.lcs_label)

                if config.display:
                    plt.show()


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Plot the final results.",
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
