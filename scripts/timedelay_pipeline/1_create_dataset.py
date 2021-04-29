"""
Setup the config file and organise the lens folder.
If you have more than 4 images, you might need to modify the config file and rerun this script.
"""
import argparse as ap
import logging
import os
import sys
from shutil import copyfile

import numpy as np

import pycs3.gen.lc_func
import pycs3.gen.mrg
import pycs3.gen.util

loggerformat='%(levelname)s: %(message)s'
logging.basicConfig(format=loggerformat,level=logging.INFO)


def main(lensname, dataname, work_dir='./'):
    data_directory = work_dir + "data/"
    pickle_directory = work_dir + "pkl/"
    config_directory = work_dir + "config/"
    simu_directory = work_dir + "Simulation/"
    lens_directory = work_dir + "Simulation/" + lensname + "_" + dataname + "/"
    figure_directory = work_dir + "Simulation/" + lensname + "_" + dataname + "/figure/"
    report_directory = work_dir + "Simulation/" + lensname + "_" + dataname + "/report/"

    if not os.path.exists(data_directory):
        print("I will create the data directory for you ! ")
        os.mkdir(data_directory)
    if not os.path.exists(pickle_directory):
        print("I will create the pickle directory for you ! ")
        os.mkdir(pickle_directory)
    if not os.path.exists(simu_directory):
        print("I will create the simulation directory for you ! ")
        os.mkdir(simu_directory)
    if not os.path.exists(config_directory):
        print("I will create the config directory for you ! ")
        os.mkdir(config_directory)
    if not os.path.exists(lens_directory):
        print("I will create the lens directory for you ! ")
        os.mkdir(lens_directory)

    rdbfile = data_directory + lensname + "_" + dataname + '.rdb'
    d = np.loadtxt(rdbfile, skiprows=2)
    n_curve = (len(d[0, :]) - 1) / 2

    if not os.path.isfile(config_directory + "config_" + lensname + "_" + dataname + ".py"):
        print("I will create the lens config file for you ! ")
        if n_curve == 2 or n_curve == 3 or n_curve == 4:
            if n_curve == 2:
                copyfile("config_default_double.py", config_directory + "config_" + lensname + "_" + dataname + ".py")
            if n_curve == 3:
                copyfile("config_default_triple.py", config_directory + "config_" + lensname + "_" + dataname + ".py")
            elif n_curve == 4:
                copyfile("config_default_quads.py", config_directory + "config_" + lensname + "_" + dataname + ".py")

            cfile = open(os.path.join(config_directory, "config_" + lensname + "_" + dataname + ".py"), 'a')
            cfile.write("#Automatically generated paths : \n")
            cfile.write("work_dir='%s'\n" % work_dir)
            cfile.write("data_directory='%s'\n" % data_directory)
            cfile.write("pickle_directory='%s'\n" % pickle_directory)
            cfile.write("simu_directory='%s'\n" % simu_directory)
            cfile.write("config_directory='%s'\n" % config_directory)
            cfile.write("lens_directory='%s'\n" % lens_directory)
            cfile.write("figure_directory='%s'\n" % figure_directory)
            cfile.write("report_directory='%s'\n" % report_directory)
            cfile.write("data = pickle_directory + '%s_%s.pkl' \n" % (lensname, dataname))
            cfile.close()

            print("Default config file created ! You might want to change the default parameters. ")
            #utils.proquest(True)

        else:
            print(
                " Warning : do you have a quad, a triple or a double ? Make sure you update lcs_label in the config file ! I'll copy the double template for this time !")
            copyfile("config_default_double.py", config_directory + "config_" + lensname + "_" + dataname + ".py")
            cfile = open(os.path.join(config_directory, "config_" + lensname + "_" + dataname + ".py"), 'a')
            cfile.write("#Automaticcaly generated paths : \n")
            cfile.write("work_dir='%s'\n" % work_dir)
            cfile.write("data_directory='%s'\n" % data_directory)
            cfile.write("pickle_directory='%s'\n" % pickle_directory)
            cfile.write("simu_directory='%s'\n" % simu_directory)
            cfile.write("config_directory='%s'\n" % config_directory)
            cfile.write("lens_directory='%s'\n" % lens_directory)
            cfile.write("figure_directory='%s'\n" % figure_directory)
            cfile.write("report_directory='%s'\n" % report_directory)
            cfile.write("data = pickle_directory + '%s_%s.pkl' \n" % (lensname, dataname))
            cfile.close()
            print("Please change the default parameters according to your object and rerun this script.")
            sys.exit()

    if not os.path.exists(figure_directory):
        os.mkdir(figure_directory)
    if not os.path.exists(report_directory):
        os.mkdir(report_directory)

    sys.path.append(config_directory)
    import importlib
    config = importlib.import_module("config_" + lensname + "_" + dataname)

    # import the data
    rdbfile = data_directory + lensname + "_" + dataname + '.rdb'

    lcs = []
    for i, a in enumerate(config.lcs_label):
        lcs.append(pycs3.gen.lc_func.rdbimport(rdbfile, a, 'mag_' + a, 'magerr_' + a, dataname))

    if config.display:
        pycs3.gen.mrg.colourise(lcs)
        pycs3.gen.lc_func.display(lcs, showdates=True)

    pycs3.gen.util.writepickle(lcs, pickle_directory + lensname + "_" + dataname + '.pkl')

    if not os.path.exists(lens_directory + 'figure/'):
        os.mkdir(lens_directory + 'figure/')

    print("I have created a new config file for this object ! You probably want to edit it in %s" % (
                config_directory + "config_" + lensname + "_" + dataname + ".py"))


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Create the data set and organise the file system for a new measurement",
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
