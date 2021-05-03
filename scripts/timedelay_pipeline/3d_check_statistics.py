"""
This script simply check that the optimised mocks light curves have the same statistics than the real one in term of zruns and sigmas.
Plots are created in your figure directory.
"""
import matplotlib
matplotlib.use('Agg')
import pycs3.gen.stat
import pycs3.gen.util
import os
import sys
import glob
import importlib
import argparse as ap
import numpy as np
import logging
loggerformat='PID %(process)06d | %(asctime)s | %(levelname)s: %(name)s(%(funcName)s): %(message)s'
logging.basicConfig(format=loggerformat,level=logging.INFO)


def write_report_checkstat(f, lcs, stats, combkw, sset, ooset, tolerance=1.0):
    f.write('\n')
    f.write('-' * 30 + '\n')
    f.write('%s, simset %s, optimiseur %s : \n' % (combkw, sset, ooset))
    success = []
    for i, lc in enumerate(lcs):
        origin_zruns = stats[i][0]
        mean_mock_zruns = stats[i][1]
        std_mock_zruns = stats[i][2]
        origin_sig = stats[i][3]
        mean_mock_sig = stats[i][4]
        std_mock_sig = stats[i][5]
        f.write("++++++ %s ++++++ \n" % lc.object)
        f.write("zruns : %.2f (obs) vs %.2f +/- %.2f (sim) \n" % (origin_zruns, mean_mock_zruns, std_mock_zruns))
        f.write("sigma : %.4f (obs) vs %.4f +/- %.4f (sim) \n" % (origin_sig, mean_mock_sig, std_mock_sig))

        rel_error_zruns = np.abs(origin_zruns - mean_mock_zruns) / std_mock_zruns
        rel_error_sig = np.abs(origin_sig - mean_mock_sig) / std_mock_sig
        if rel_error_zruns < tolerance and rel_error_sig < tolerance:
            success.append(True)
        else:
            success.append(False)

    if all(success):
        f.write("Successfully matched zruns and sigmas within %2.2f sigmas \n" % tolerance)
    else:
        for i, suc in enumerate(success):
            f.write("WARNING : did not matched zruns and sigmas within %2.2f sigmas for curve %s\n" % (
                tolerance, lcs[i].object))


def main(lensname, dataname, work_dir='./'):
    sys.path.append(work_dir + "config/")
    config = importlib.import_module("config_" + lensname + "_" + dataname)

    check_stat_plot_dir = config.figure_directory + 'check_stat_plots/'
    report_file = os.path.join(config.report_directory, 'report_check_stats.txt')
    f = open(report_file, 'w')
    f.write('### REPORT STATISTICS ###')

    if not os.path.isdir(check_stat_plot_dir):
        os.mkdir(check_stat_plot_dir)

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
            simset_available = glob.glob(config.lens_directory + config.combkw[i, j] + '/sims_mocks_*')
            lcs, spline = pycs3.gen.util.readpickle(
                config.lens_directory + config.combkw[i, j] + '/initopt_%s_ks%i_%s%i.pkl' % (
                    dataname, kn, string_ML, ml))

            for a in simset_available:
                a = a.split('/')[-1]
                if "_opt_" in a:  # take only the optimised sub-folders
                    sset = a.split('_opt_')[0]
                    sset = sset[5:]
                    ooset = a.split('_opt_')[1]
                    if ooset[0:7] == 'regdiff':
                        continue  # it makes no sens to use this function for regdiff
                    else:
                        stats = pycs3.gen.stat.anaoptdrawn(lcs, spline, simset=sset, optset=ooset, showplot=False,
                                                           nplots=1,
                                                           directory=config.lens_directory + config.combkw[i, j] + '/',
                                                           plotpath= check_stat_plot_dir, id =config.combkw[i, j])
                        # move the figure to the correct directory :
                        # os.system("mv " + "fig_anaoptdrawn_%s_%s_resi_1.png " % (sset, ooset) + check_stat_plot_dir
                        #           + "%s_fig_anaoptdrawn_%s_%s_resi_1.png" % (config.combkw[i, j], sset, ooset))
                        # os.system("mv " + "fig_anaoptdrawn_%s_%s_resihists.png " % (sset, ooset) + check_stat_plot_dir
                        #           + "%s_fig_anaoptdrawn_%s_%s_resihists.png" % (config.combkw[i, j], sset, ooset))
                        write_report_checkstat(f, lcs, stats, config.combkw[i, j], sset, ooset)

    f.close()


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Check the noise statistics of the mock light curves.",
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
