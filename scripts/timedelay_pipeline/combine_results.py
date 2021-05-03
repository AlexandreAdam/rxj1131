"""
This script is to be called once copies and mocks have been optimized, runresults.pkl have been produced
"""
import argparse as ap
import importlib
import logging
import os
import pickle as pkl
import sys

import matplotlib.style

import pycs3.tdcomb.comb
import pycs3.tdcomb.plot
import pycs3.pipe.pipe_utils as ut


loggerformat='%(message)s'
logging.basicConfig(format=loggerformat,level=logging.INFO)

matplotlib.style.use('classic')
matplotlib.rc('font', family="Times New Roman")


def main(lensname, dataname, work_dir='./')
	sys.path.append(work_dir + "config/")
	config = importlib.import_module("config_" + lensname + "_" + dataname)

	figure_dir = config.figure_directory + "final_results/"
	if not os.path.isdir(figure_dir):
		os,mkdir(figure_dir)


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

	regdiffparamskw = ut.read_preselected_regdiffparamskw(config.config_directory + 'preset_regdiff.txt')[0]
	result_dir = config.lens_directory + config.combkw[0,0]

	regdiffresult_dir = result_dir + 'sims_%s_opt_regdiff%s_t%s/' % (config.simset_copy, regdiffparamskw, str(int(config.tsrand)))
	spl1result_dir = result_dir + 'sims_%s_opt_spl1t%s/' % (config.simset_copy, str(int(tsrand)))

	group_list = [pycs3.tdcomb.comb.getresults(
						pycs3.tdcomb.comb.CScontainer('Regression difference',
													result_file_delays=regdiffresult_dir + 'sims_%s_opt_regdiff%s_t%s_delays.pkl' % (config.simset_copy, regdiffparamskw, str(int(config.tsrand))),
													result_file_errorbars=regdiffresult_dir + 'sims_%s_opt_regdiff%s_t%s_errorbars.pkl' % (config.simset_mock, regdiffparamskw, str(int(config.tsrand))),
													colour='red')),
				  pycs3.tdcomb.comb.getresults(
						pycs3.tdcomb.comb.CScontainer('Free-knot splines',
													result_file_delays=spl1result_dir + 'sims_%s_opt_spl1t%s_delays.pkl' % (config.simset_copy, str(int(tsrand))),
													result_file_errorbars=spl1result_dir + 'sims_%s_opt_spl1t%s_errorbars.pkl' % (config.simset_mock, str(int(tsrand))),
													colour='blue'))]
	testmode = False
	nbins = 5000

	# linearize the distribution :
	binslist = [np.linspace(-100,100,nbins) for i in range(6)] #we provide the bins to compute the distribution, one array per delay.
	for g, group in enumerate(group_list):
	    group.binslist = binslist
	    group.linearize(testmode=testmode)

	# compute combined estimate :
	combined = pycs3.tdcomb.comb.combine_estimates(group_list, sigmathresh=0.0, testmode=testmode)
	combined.linearize(testmode=testmode)
	combined.name = 'combined (marginalization)'
	combined.plotcolor = 'black'

	pycs3.tdcomb.plot.delayplot(group_list + [combined], rplot=10, refgroup=combined, hidedetails=True,
								showbias=False, showran=False, showlegend=True, figsize=(15,10),
								horizontaldisplay=False, legendfromrefgroup=False, auto_radius=True,
								tick_step_auto=True, filename=figure_dir + 'combined_results_%s.png' % (config.combkw[0,0]))

	print("COMBINATION SUCCESFUL")
	

if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Plot TD of all pairs for both estimators",
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
