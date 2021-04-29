"""
Optimise the copy and mock data. WARNING : this may take loooooooong. You probably want to launch that on several cores.
I'm not re-running on already optimized lcs ! It should be safe to launch this script many times, it will run on different batch of lightcurves.
"""

import argparse as ap
import copy
import importlib
import logging
import os
import pickle as pkl
import sys
import time

import numpy as np
import pycs3.gen.lc_func
import pycs3.gen.util
import pycs3.sim.run
from multiprocess import Pool, cpu_count

loggerformat='PID %(process)06d | %(asctime)s | %(levelname)s: %(name)s(%(funcName)s): %(message)s'
logging.basicConfig(format=loggerformat,level=logging.WARNING)

def exec_worker_copie_aux(args):
    return exec_worker_copie(*args)


def exec_worker_copie(i, simset_copy, lcs, simoptfct, kwargs_optim, optset, tsrand, destpath):
    print("worker %i starting..." % i)
    time.sleep(i)
    sucess_dic = pycs3.sim.run.multirun(simset_copy, lcs, simoptfct, kwargs_optim=kwargs_optim,
                                       optset=optset, tsrand=tsrand, destpath=destpath)
    return sucess_dic


def exec_worker_mocks_aux(args):
    return exec_worker_mocks(*args)


def exec_worker_mocks(i, simset_mock, lcs, simoptfct, kwargs_optim, optset, tsrand, destpath):
    print("worker %i starting..." % i)
    time.sleep(i)
    sucess_dic = pycs3.sim.run.multirun(simset_mock, lcs, simoptfct, kwargs_optim=kwargs_optim,
                                       optset=optset, tsrand=tsrand, keepopt=True, destpath=destpath)
    return sucess_dic


def write_report_optimisation(f, success_dic):
    if success_dic == None:
        f.write('This set was already optimised.\n')
    else:
        for i, dic in enumerate(success_dic):
            f.write('------------- \n')
            if dic == None:
                continue
            if dic['success']:
                f.write('None of the optimisations have failed for pickle %i. \n' % i)
            else:
                f.write('The optimisation of the following curves have failed in pickle %i : \n' % i)
                for id in dic['failed_id']:
                    f.write("   Curve %i :" % id + str(dic['error_list'][0]) + ' \n')
                f.write('\n')


def main(lensname, dataname, work_dir='./'):
    main_path = os.getcwd()
    sys.path.append(work_dir + "config/")
    config = importlib.import_module("config_" + lensname + "_" + dataname)
    base_lcs = pycs3.gen.util.readpickle(config.data)
    f = open(os.path.join(config.report_directory, 'report_optimisation_%s.txt' % config.simoptfctkw), 'w')

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
            lcs = copy.deepcopy(base_lcs)
            destpath = os.path.join(main_path, config.lens_directory + config.combkw[a, b] + '/')
            print(destpath)
            ##### We start by shifting our curves "by eye", to get close to the result and help the optimisers to do a good job
            if config.magshift is None :
                magsft = [-np.median(lc.getmags()) for lc in lcs]
            else :
                magsft = config.magshift
            pycs3.gen.lc_func.applyshifts(lcs, config.timeshifts,magsft)  # be careful, this remove ml as well.

            # We also give them a microlensing model (here, similar to Courbin 2011)
            config.attachml(lcs, ml)  # this is because they were saved as raw lcs, wihtout lcs.

            if config.max_core == None:
                nworkers = cpu_count()
            else:
                nworkers = config.max_core

            for c, opts in enumerate(config.optset):
                if config.simoptfctkw == "spl1":
                    kwargs = {'kn': kn, 'name': 'spl1'}
                elif config.simoptfctkw == "regdiff":
                    kwargs = config.kwargs_optimiser_simoptfct[c]
                else:
                    print("Error : simoptfctkw must be spl1 or regdiff")

                if config.run_on_copies:
                    print("I will run the optimiser on the copies with the parameters :", kwargs)
                    p = Pool(nworkers)
                    if config.simoptfctkw == "spl1":
                        job_args = [
                            (j, config.simset_copy, lcs, config.simoptfct, kwargs, opts, config.tsrand, destpath) for j
                            in
                            range(nworkers)]
                        success_list_copies = p.map(exec_worker_copie_aux, job_args)
                        p.close()
                        p.join()

                    elif config.simoptfctkw == "regdiff":
                        if a == 0 and b == 0:  # for copies, run on only 1 (knstp,mlknstp) as it the same for others
                            job_args = (
                            0, config.simset_copy, lcs, config.simoptfct, kwargs, opts, config.tsrand, destpath)
                            success_list_copies = exec_worker_copie_aux(job_args)
                            success_list_copies = [
                                success_list_copies]  # we hace to turn it into a list to match spl format
                            dir_link = os.path.join(destpath, "sims_%s_opt_%s" % (config.simset_copy, opts))
                            print("Dir link :", dir_link)
                            pkl.dump(dir_link, open(
                                os.path.join(config.lens_directory, 'regdiff_copies_link_%s.pkl' % kwargs['name']),
                                'wb'))

                    f.write('COPIES, kn%i, %s%i, optimiseur %s : \n' % (kn, string_ML, ml, kwargs['name']))
                    write_report_optimisation(f, success_list_copies)
                    f.write('################### \n')

                if config.run_on_sims:
                    print("I will run the optimiser on the simulated lcs with the parameters :", kwargs)
                    p = Pool(nworkers)
                    job_args = [(j, config.simset_mock, lcs, config.simoptfct, kwargs, opts, config.tsrand, destpath) for j in range(nworkers)]
                    """
                    Serial version of this code :
                        job_args = (0, config.simset_mock, lcs, config.simoptfct, kwargs, opts, config.tsrand, destpath)
                        success_list_simu = exec_worker_mocks_aux(job_args)  # if regdiff uses another level of parallelism.
                        success_list_simu = [success_list_simu]# p.map(exec_worker_copie_aux, job_args)
                    """
                    success_list_simu = p.map(exec_worker_mocks_aux, job_args)
                    p.close()
                    p.join()
                    f.write('SIMULATIONS, kn%i, %s%i, optimiseur %s : \n' % (kn, string_ML, ml, kwargs['name']))
                    write_report_optimisation(f, success_list_simu)
                    f.write('################### \n')

    print("OPTIMISATION DONE : report written in %s" % (
        os.path.join(config.report_directory, 'report_optimisation_%s.txt' % config.simoptfctkw)))
    f.close()


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Shift the mock curves and the copies.",
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
