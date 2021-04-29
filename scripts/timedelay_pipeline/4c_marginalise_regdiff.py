"""
Combine the the regdiff optimiser Groups of time delay Estimates. I am selecting the most precise Groups among different generative noise model.
I am using the threshold defined in your config file to select which Groups to combine among your regdiff set of parameter estimator.
See Millon et al. (2020) for details.
"""
import argparse as ap
import copy
import importlib
import logging
import os
import pickle as pkl
import sys

import matplotlib.style
import numpy as np

import pycs3.pipe.pipe_utils as ut
import pycs3.tdcomb.comb
import pycs3.tdcomb.plot

loggerformat='%(message)s'
logging.basicConfig(format=loggerformat,level=logging.INFO)

matplotlib.style.use('classic')
matplotlib.rc('font', family="Times New Roman")


def main(lensname, dataname, work_dir='./'):
    sys.path.append(work_dir + "config/")
    config = importlib.import_module("config_" + lensname + "_" + dataname)
    marginalisation_plot_dir = config.figure_directory + 'marginalisation_plots/'
    regdiff_dir = os.path.join(config.lens_directory, "regdiff_outputs/")
    regdiff_copie_dir = os.path.join(regdiff_dir, "copies/")

    if not os.path.isdir(marginalisation_plot_dir):
        os.mkdir(marginalisation_plot_dir)

    indiv_marg_dir = marginalisation_plot_dir + config.name_marg_regdiff + '/'
    if not os.path.isdir(indiv_marg_dir):
        os.mkdir(indiv_marg_dir)

    marginalisation_dir = config.lens_directory + config.name_marg_regdiff + '/'
    if not os.path.isdir(marginalisation_dir):
        os.mkdir(marginalisation_dir)

    if config.testmode:
        nbins = 500
    else:
        nbins = 5000

    colors = ["royalblue", "crimson", "seagreen", "darkorchid", "darkorange", 'indianred', 'purple', 'brown', 'black',
              'violet', 'dodgerblue', 'palevioletred', 'olive',
              'brown', 'salmon', 'chocolate', 'indigo', 'steelblue', 'cyan', 'gold', 'lightcoral']

    f = open(marginalisation_dir + 'report_%s_sigma%2.1f.txt' % (config.name_marg_regdiff, config.sigmathresh), 'w')
    path_list = []
    name_list = []
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

    kw_list = ut.read_preselected_regdiffparamskw(config.preselection_file)
    kw_dic = ut.get_keyword_regdiff_from_file(config.preselection_file)
    for s, set in enumerate(kw_dic):
        if 'name' not in list(set.keys()):
            set['name'] = 'Set %i' % s

    for paramskw, dickw in zip(kw_list, kw_dic):
        for n, noise in enumerate(config.tweakml_name_marg_regdiff):

            count = 0
            color_id = 0

            group_list = []
            medians_list = []
            errors_up_list = []
            errors_down_list = []
            simset_mock_ava = ["mocks_n%it%i_%s" % (int(config.nsim * config.nsimpkls), config.truetsr, twk) for twk in
                               config.tweakml_name_marg_regdiff]
            opt = 'regdiff'

            if not config.use_preselected_regdiff:
                raise RuntimeError(
                    "Turn the use_preselected_regdiff to True and set your preselection_file before using this script.")

            for a, kn in enumerate(config.knotstep_marg_regdiff):
                for b, ml in enumerate(config.mlknotsteps_marg_regdiff):

                    regdiff_mocks_dir = os.path.join(regdiff_dir, "mocks_knt%i_mlknt%i/" % (kn, ml))

                    result_file_delay = regdiff_copie_dir + 'sims_%s_opt_regdiff%s' % (config.simset_copy, paramskw) \
                                        + 't%i_delays.pkl' % int(config.tsrand)
                    result_file_errorbars = regdiff_mocks_dir \
                                            + 'sims_%s_opt_regdiff%s' % (simset_mock_ava[n], paramskw) + \
                                            't%i_errorbars.pkl' % int(config.tsrand)

                    if not os.path.isfile(result_file_delay) or not os.path.isfile(result_file_errorbars):
                        print('Error I cannot find the files %s or %s. ' \
                              'Did you run the 3c and 4a?' % (result_file_delay, result_file_errorbars))
                        f.write('Error I cannot find the files %s or %s. \n' % (
                            result_file_delay, result_file_errorbars))
                        continue

                    group_list.append(pycs3.tdcomb.comb.getresults(
                        pycs3.tdcomb.comb.CScontainer(data=dataname, knots=kn, ml=ml,
                                                    name="knstp %i mlknstp %i" % (kn, ml),
                                                    drawopt=config.optfctkw, runopt=opt,
                                                    ncopy=config.ncopy * config.ncopypkls,
                                                    nmocks=config.nsim * config.nsimpkls, truetsr=config.truetsr,
                                                    colour=colors[color_id],
                                                    result_file_delays=result_file_delay,
                                                    result_file_errorbars=result_file_errorbars)))
                    medians_list.append(group_list[-1].medians)
                    errors_up_list.append(group_list[-1].errors_up)
                    errors_down_list.append(group_list[-1].errors_down)

                    if np.isnan(medians_list[-1]).any() or np.isnan(errors_up_list[-1]).any() or np.isnan(
                            errors_down_list[-1]).any():
                        print("There is some Nan value in %s, for noise %s, kn %i, %s%i" % (
                            dickw['name'], noise, kn, string_ML, ml))
                        print("I could erase this entry and continue the marginalisation. ")
                        ut.proquest(True)
                        medians_list = medians_list[:-1]
                        errors_down_list = errors_down_list[:-1]
                        errors_up_list = errors_down_list[:-1]
                        group_list = group_list[:-1]
                        continue
                    color_id += 1
                    count += 1
                    if color_id >= len(colors):
                        print("Warning : I don't have enough colors in my list, I'll restart from the beginning.")
                        color_id = 0  # reset the color form the beginning

                    f.write('Set %i, knotstep : %2.2f, %s : %2.2f \n' % (count, kn, string_ML, ml))
                    f.write('covkernel : %s, point density: %2.2f, pow : %2.2f, errscale:%2.2f \n'
                            % (dickw["covkernel"], dickw["pointdensity"],
                               dickw["pow"], dickw["errscale"]))
                    f.write('Tweak ml name : %s \n' % noise)
                    f.write('------------------------------------------------ \n')

            # build the bin list :
            medians_list = np.asarray(medians_list)
            errors_down_list = np.asarray(errors_down_list)
            errors_up_list = np.asarray(errors_up_list)
            binslist = []
            for i, lab in enumerate(config.delay_labels):
                bins = np.linspace(min(medians_list[:, i]) - 10 * min(errors_down_list[:, i]),
                                   max(medians_list[:, i]) + 10 * max(errors_up_list[:, i]), nbins)
                binslist.append(bins)

            color_id = 0
            for g, group in enumerate(group_list):
                group.binslist = binslist
                group.plotcolor = colors[color_id]
                group.linearize(testmode=config.testmode)
                group.objects = config.lcs_label
                color_id += 1
                if color_id >= len(colors):
                    print("Warning : I don't have enough colors in my list, I'll restart from the beginning.")
                    color_id = 0  # reset the color form the beginning

            combined = copy.deepcopy(
                pycs3.tdcomb.comb.combine_estimates(group_list, sigmathresh=1000.0, testmode=config.testmode))
            combined.linearize(testmode=config.testmode)
            combined.name = 'Most precise'

            print("%s : Taking the best of all spline parameters for regdiff parameters set %s" % (
                config.name_marg_regdiff, dickw['name']))
            combined.niceprint()

            # plot the results :

            text = [
                (0.85, 0.90, r"$\mathrm{" + config.full_lensname + "}$" + "\n" + r"$\mathrm{PyCS\ estimates}$",
                 {"fontsize": 26, "horizontalalignment": "center"})]
            radius = (np.max(errors_up_list) + np.max(errors_down_list)) / 2.0 * 3.5
            ncurve = len(config.lcs_label)

            if ncurve > 2:
                auto_radius = True
                figsize = (17, 13)
            else:
                auto_radius = False
                figsize = (15, 10)

            if config.display:
                pycs3.tdcomb.plot.delayplot(group_list + [combined], rplot=radius, refgroup=combined,
                                          text=text, hidedetails=True, showbias=False, showran=False,
                                          tick_step_auto=True, autoobj=config.lcs_label,
                                          showlegend=True, figsize=figsize, horizontaldisplay=False,
                                          legendfromrefgroup=False, auto_radius=auto_radius)

            pycs3.tdcomb.plot.delayplot(group_list + [combined], rplot=radius, refgroup=combined, text=text,
                                      hidedetails=True, tick_step_auto=True, autoobj=config.lcs_label,
                                      showbias=False, showran=False, showlegend=True, figsize=figsize,
                                      horizontaldisplay=False, auto_radius=auto_radius,
                                      legendfromrefgroup=False,
                                      filename=indiv_marg_dir + config.name_marg_regdiff + "_%s_%s.png" % (
                                          dickw['name'], noise))

            pkl.dump(group_list, open(
                marginalisation_dir + config.name_marg_regdiff + "_%s_%s" % (dickw['name'], noise) + '_groups.pkl',
                'wb'))
            pkl.dump(combined, open(
                marginalisation_dir + config.name_marg_regdiff + "_%s_%s" % (dickw['name'], noise) + '_combined.pkl',
                'wb'))
            path_list.append(
                marginalisation_dir + config.name_marg_regdiff + "_%s_%s" % (dickw['name'], noise) + '_combined.pkl')
            name_list.append('%s, Noise : %s ' % (dickw['name'], noise))

    # ------  MAKE THE FINAL REGDIFF ESTIMATE ------#
    final_groups, final_combined = pycs3.tdcomb.comb.group_estimate(path_list, name_list=name_list, colors=colors,
                                                                  sigma_thresh=config.sigmathresh,
                                                                  new_name_marg=config.name_marg_regdiff,
                                                                  testmode=config.testmode,
                                                                  object_name=config.lcs_label)
    radius_f = (final_combined.errors_down[0] + final_combined.errors_up[0]) / 2.0 * 2.5
    text = [
        (0.85, 0.90, r"$\mathrm{" + config.full_lensname + "}$" + "\n" + r"$\mathrm{PyCS\ estimates}$",
         {"fontsize": 26, "horizontalalignment": "center"})]

    if config.display:
        pycs3.tdcomb.plot.delayplot(final_groups + [final_combined], rplot=radius_f, refgroup=final_combined,
                                  tick_step_auto=True,
                                  text=text, hidedetails=True, showbias=False, showran=False, auto_radius=auto_radius,
                                  autoobj=config.lcs_label,
                                  showlegend=True, figsize=(15, 10), horizontaldisplay=False, legendfromrefgroup=False)

    pycs3.tdcomb.plot.delayplot(final_groups + [final_combined], rplot=radius_f, refgroup=final_combined, text=text,
                              hidedetails=True,
                              showbias=False, showran=False, showlegend=True, figsize=(15, 10), horizontaldisplay=False,
                              autoobj=config.lcs_label,
                              legendfromrefgroup=False, auto_radius=auto_radius, tick_step_auto=True,
                              filename=indiv_marg_dir + config.name_marg_regdiff + "_sigma_%2.2f.png" % (
                                  config.sigmathresh))

    pkl.dump(final_groups,
             open(
                 marginalisation_dir + config.name_marg_regdiff + "_sigma_%2.2f" % config.sigmathresh + '_groups.pkl',
                 'wb'))
    pkl.dump(final_combined, open(
        marginalisation_dir + config.name_marg_regdiff + "_sigma_%2.2f" % config.sigmathresh + '_combined.pkl', 'wb'))


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Marginalise over the regdiff optimiser parameters.",
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
