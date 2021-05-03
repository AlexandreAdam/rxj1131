"""
LAst level of combination, combining across dataset. You can choose which dataset to use in a specific config file :
Comination/lensname/config_combination_lensname.py
"""
import argparse as ap
import importlib
import logging
import os
import pickle as pkl
import sys
from shutil import copyfile

import matplotlib.style

import pycs3.pipe.pipe_utils as ut
import pycs3.tdcomb.comb
import pycs3.tdcomb.plot

loggerformat='%(levelname)s: %(message)s'
logging.basicConfig(format=loggerformat,level=logging.INFO)

matplotlib.style.use('classic')
matplotlib.rc('font', serif='Times New Roman')


def main(lensname, work_dir='./'):
    combi_dir = os.path.join(os.path.join(work_dir, 'Combination'), lensname)
    simu_dir = os.path.join(work_dir, 'Simulation')
    plot_dir = os.path.join(combi_dir, 'plots')
    if not os.path.exists(combi_dir):
        ut.mkdir_recursive(combi_dir)
    if not os.path.exists(plot_dir):
        ut.mkdir_recursive(plot_dir)

    config_file = work_dir + "Combination/" + lensname + "/config_combination_" + lensname + '.py'

    if os.path.isfile(config_file):
        print("Combination config file already exists.")
        print("Using :", config_file)
    else:
        print("Combination config file do not exist yet. I will create it.")
        copyfile("config_combination_default.py", config_file)
        print("Please edit : ", config_file)
        print("and restart this script...")
        exit()

    sys.path.append(combi_dir)
    config = importlib.import_module("config_combination_" + lensname)

    print("Working on :", lensname)
    print("Combining the following data sets :", config.data_sets)

    # ACCESSING THE GROUP FILE #

    lens_directory = [os.path.join(simu_dir, lensname + '_' + d) for d in config.data_sets]
    lens_directory_extra = [os.path.join(simu_dir, lensname + '_' + d) for d in config.extra_data_sets]
    path_list = [ldir + '/' + marg + '/' + marg + '_sigma_%2.2f' % sig + '_combined.pkl' for marg, sig, ldir in
                 zip(config.marg_to_combine, config.sigma_to_combine, lens_directory)]
    path_list_spline = [ldir + '/' + marg + '/' + marg + '_sigma_%2.2f' % sig + '_combined.pkl'
                        for marg, sig, ldir in
                        zip(config.marg_to_combine_spline, config.sigma_to_combine_spline, lens_directory)]
    path_list_regdiff = [ldir + '/' + marg + '/' + marg + '_sigma_%2.2f' % sig + '_combined.pkl'
                         for marg, sig, ldir in
                         zip(config.marg_to_combine_regdiff, config.sigma_to_combine_regdiff, lens_directory)]

    path_list_extra = [ldir + '/' + marg + '/' + marg + '_sigma_%2.2f' % sig + '_combined.pkl'
                       for marg, sig, ldir in
                       zip(config.extra_marg_to_combine, config.extra_sigma_to_combine, lens_directory_extra)]
    path_list_extra_regdiff = [ldir + '/' + marg + '/' + marg + '_sigma_%2.2f' % sig + '_combined.pkl'
                               for marg, sig, ldir in
                               zip(config.extra_marg_to_combine_regdiff, config.extra_sigma_to_combine_regdiff,
                                   lens_directory_extra)]
    path_list_extra_spline = [ldir + '/' + marg + '/' + marg + '_sigma_%2.2f' % sig + '_combined.pkl'
                              for marg, sig, ldir in
                              zip(config.extra_marg_to_combine_spline, config.extra_sigma_to_combine_spline,
                                  lens_directory_extra)]
    name_list = [d for d in config.data_sets]

    colors = ["royalblue", "crimson", "darkorchid", "darkorange", 'indianred', 'purple', 'brown', 'black',
              'violet', 'dodgerblue', 'palevioletred', 'olive',
              'brown', 'salmon', "seagreen", 'chocolate', 'indigo', 'steelblue', 'cyan', 'gold', 'lightcoral']

    groups, sum = pycs3.tdcomb.comb.group_estimate(path_list, name_list=name_list, colors=colors,
                                                 sigma_thresh=config.sigma_thresh, new_name_marg="Sum",
                                                 testmode=config.testmode)
    sum.name = "PyCS-Sum"
    sum.plotcolor = "black"

    mult = pycs3.tdcomb.comb.mult_estimates(groups)
    mult.name = "PyCS-Mult"
    mult.plotcolor = "gray"

    groups_extra = []
    for i, p in enumerate(path_list_extra):
        with open(p, 'r') as q:
            g = pkl.load(q)
            g.name = config.extra_data_sets[i] + "$^*$"
            g.plotcolor = 'green'
            groups_extra.append(g)
    groups_extra_spline = []
    for i, p in enumerate(path_list_extra_spline):
        with open(p, 'r') as q:
            g = pkl.load(q)
            g.name = "Spline " + config.extra_data_sets[i] + "$^*$"
            g.plotcolor = 'silver'
            groups_extra_spline.append(g)
    groups_extra_regdiff = []
    for i, p in enumerate(path_list_extra_regdiff):
        with open(p, 'r') as q:
            g = pkl.load(q)
            g.name = "Regdiff " + config.extra_data_sets[i] + "$^*$"
            g.plotcolor = 'darkgrey'
            groups_extra_regdiff.append(g)

    radius = (sum.errors_down[0] + sum.errors_up[0]) / 2.0 * 3.0
    ncurve = len(config.lcs_label)
    toplot = groups + groups_extra + [sum] + [mult]

    if ncurve > 2:
        auto_radius = True
        xlabelfontsize = 25
        figsize = (15, 10)
        bottom = 0.08
        txtstep = 0.03
        legendy_offset = 0.14
    else:
        auto_radius = False
        xlabelfontsize = 32
        figsize = (12, 9)
        bottom = 0.15
        for g in toplot:
            g.labelfontsize = 24
            g.legendfontsize = 20
        legendy_offset = 0.15
        txtstep = 0.035

    text = [
        (0.85, 0.88, r"$\mathrm{" + config.full_lensname + "}$" + "\n" + r"$\mathrm{PyCS\ estimates}$",
         {"fontsize": 26, "horizontalalignment": "center"})]

    if config.display:
        pycs3.tdcomb.plot.delayplot(toplot, rplot=radius, refgroup=mult, text=text,
                                  autoobj=config.lcs_label, bottom=bottom, legendy_offset=legendy_offset,
                                  hidedetails=True, showbias=False, showran=False, showlegend=True, tick_step_auto=True,
                                  figsize=figsize, horizontaldisplay=False, legendfromrefgroup=False, txtstep=txtstep,
                                  auto_radius=auto_radius, xlabelfontsize=xlabelfontsize, update_group_style=False)

    pycs3.tdcomb.plot.delayplot(toplot, rplot=radius, refgroup=mult, text=text,
                              autoobj=config.lcs_label,
                              hidedetails=True, showbias=False, showran=False, showlegend=True, figsize=figsize,
                              auto_radius=auto_radius, tick_step_auto=True, bottom=bottom,
                              horizontaldisplay=False, legendfromrefgroup=False, legendy_offset=legendy_offset,
                              filename=plot_dir + "/" + lensname + "_combined_estimate_" + config.combi_name + ".png",
                              xlabelfontsize=xlabelfontsize, update_group_style=False, txtstep=txtstep)

    pkl.dump(groups, open(os.path.join(combi_dir, config.combi_name + '_groups.pkl'), 'wb'))
    pkl.dump([sum, mult], open(os.path.join(combi_dir, "sum-mult_" + config.combi_name + '.pkl'), 'wb'))

    legendx = 0.82
    legendy_offset = 0.17

    # Plot with the spline only
    groups_spline, sum_spline = pycs3.tdcomb.comb.group_estimate(path_list_spline, name_list=name_list, colors=colors,
                                                               sigma_thresh=config.sigma_thresh, new_name_marg="Sum",
                                                               testmode=config.testmode)
    sum_spline.name = "Sum"
    sum_spline.plotcolor = "black"
    text = [
        (0.82, 0.85,
         r"$\mathrm{" + config.full_lensname + "}$" + "\n" + r"$\mathrm{PyCS}$" + '\n' + r"$\mathrm{Free-knot\ Spline}$",
         {"fontsize": 26, "horizontalalignment": "center"})]
    if config.display:
        pycs3.tdcomb.plot.delayplot(groups_spline + groups_extra_spline + [sum_spline], rplot=radius, refgroup=sum_spline,
                                  text=text,
                                  hidedetails=True, showbias=False, showran=False, showlegend=True, tick_step_auto=True,
                                  figsize=(15, 10), horizontaldisplay=False, legendfromrefgroup=False,
                                  auto_radius=auto_radius,
                                  legendx=legendx, legendy_offset=legendy_offset, autoobj=config.lcs_label, )

    pycs3.tdcomb.plot.delayplot(groups_spline + groups_extra_spline + [sum_spline], rplot=radius, refgroup=sum_spline,
                              text=text,
                              hidedetails=True, showbias=False, showran=False, showlegend=True, figsize=(15, 10),
                              auto_radius=auto_radius, tick_step_auto=True,
                              horizontaldisplay=False, legendfromrefgroup=False,
                              filename=plot_dir + "/" + lensname + "_combined_estimate_spline_" + config.combi_name + ".png",
                              legendx=legendx, legendy_offset=legendy_offset)

    # Plot with regdiff only
    groups_regdiff, sum_regdiff = pycs3.tdcomb.comb.group_estimate(path_list_regdiff, name_list=name_list, colors=colors,
                                                                 sigma_thresh=config.sigma_thresh, new_name_marg="Sum",
                                                                 testmode=config.testmode)
    sum_regdiff.name = "Sum"
    sum_regdiff.plotcolor = "black"
    text = [
        (0.82, 0.85,
         r"$\mathrm{" + config.full_lensname + "}$" + "\n" + r"$\mathrm{PyCS}$" + '\n' + r"$\mathrm{Regression\ Difference}$",
         {"fontsize": 26, "horizontalalignment": "center"})]
    if config.display:
        pycs3.tdcomb.plot.delayplot(groups_regdiff + groups_extra_regdiff + [sum_regdiff], rplot=radius,
                                  refgroup=sum_regdiff, text=text,
                                  hidedetails=True, showbias=False, showran=False, showlegend=True, tick_step_auto=True,
                                  figsize=(15, 10), horizontaldisplay=False, legendfromrefgroup=False,
                                  auto_radius=auto_radius,
                                  legendx=legendx, legendy_offset=legendy_offset, autoobj=config.lcs_label, )

    pycs3.tdcomb.plot.delayplot(groups_regdiff + groups_extra_regdiff + [sum_regdiff], rplot=radius, refgroup=sum_regdiff,
                              text=text,
                              hidedetails=True, showbias=False, showran=False, showlegend=True, figsize=(15, 10),
                              auto_radius=auto_radius, tick_step_auto=True,
                              horizontaldisplay=False, legendfromrefgroup=False,
                              filename=plot_dir + "/" + lensname + "_combined_estimate_regdiff_" + config.combi_name + ".png",
                              legendx=legendx, legendy_offset=legendy_offset)

    # Plot with regdiff and spline together
    name_list_all = []
    for i, p in enumerate(path_list_regdiff):
        name_list_all.append("Regdiff %s" % config.data_sets[i])
    for i, p in enumerate(path_list_spline):
        name_list_all.append("Spline %s" % config.data_sets[i])

    groups_all, sum_all = pycs3.tdcomb.comb.group_estimate(path_list_regdiff + path_list_spline, name_list_all,
                                                         config.delay_labels,
                                                         colors, config.sigma_thresh, "Sum", testmode=config.testmode)
    sum_all.name = "Sum"
    sum_all.plotcolor = "black"
    text = [
        (0.85, 0.90, r"$\mathrm{" + config.full_lensname + "}$" + "\n" + r"$\mathrm{PyCS}$",
         {"fontsize": 24, "horizontalalignment": "center"})]
    if config.display:
        pycs3.tdcomb.plot.delayplot(groups_all + groups_extra_spline + groups_extra_regdiff + [sum_all], rplot=radius,
                                  refgroup=sum_all, text=text,
                                  hidedetails=True, showbias=False, showran=False, showlegend=True, tick_step_auto=True,
                                  figsize=(15, 10), horizontaldisplay=False, legendfromrefgroup=False,
                                  auto_radius=auto_radius,
                                  legendx=0.85, legendy_offset=0.12, autoobj=config.lcs_label, )

    pycs3.tdcomb.plot.delayplot(groups_all + groups_extra_spline + groups_extra_regdiff + [sum_all], rplot=radius,
                              refgroup=sum_all, text=text,
                              hidedetails=True, showbias=False, showran=False, showlegend=True, figsize=(15, 10),
                              auto_radius=auto_radius, tick_step_auto=True,
                              horizontaldisplay=False, legendfromrefgroup=False,
                              filename=plot_dir + "/" + lensname + "_combined_estimate_regdiff-spline" + config.combi_name + ".png",
                              legendx=0.85, legendy_offset=0.12)


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Combine the data sets.",
                               formatter_class=ap.RawTextHelpFormatter)
    help_lensname = "name of the lens to process"
    help_work_dir = "name of the working directory. default : ./"

    parser.add_argument(dest='lensname', type=str,
                        metavar='lens_name', action='store',
                        help=help_lensname)
    parser.add_argument('--dir', dest='work_dir', type=str,
                        metavar='', action='store', default='./',
                        help=help_work_dir)

    args = parser.parse_args()

    main(args.lensname, work_dir=args.work_dir)
