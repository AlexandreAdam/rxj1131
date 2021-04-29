"""
This scrip will create copy of the data and mock light curves, according to your generative noise model.
I am using multithreading to do that.
"""
import os
import pycs3.gen.util
import pycs3.sim.draw
import sys
import glob
import argparse as ap
import multiprocess
import importlib
import logging
loggerformat='PID %(process)06d | %(asctime)s | %(levelname)s: %(name)s(%(funcName)s): %(message)s'
logging.basicConfig(format=loggerformat,level=logging.WARNING)


def draw_mock_para(i, j, kn, ml, string_ML, lensname, dataname, work_dir):
    current_dir = os.getcwd()
    sys.path.append(work_dir + "config/")
    config = importlib.import_module("config_" + lensname + "_" + dataname)

    print("I am drawing curves for ks%i, knml%i" % (kn, ml))
    os.chdir(config.lens_directory + config.combkw[i, j])
    lcs, spline = pycs3.gen.util.readpickle('initopt_%s_ks%i_%s%i.pkl' % (dataname, kn, string_ML, ml))

    pycs3.sim.draw.saveresiduals(lcs, spline)

    if config.run_on_copies:
        files_copy = glob.glob("sims_" + config.simset_copy + '/*.pkl')
        pycs3.sim.draw.multidraw(lcs, onlycopy=True, n=config.ncopy, npkl=config.ncopypkls,
                                 simset=config.simset_copy)

    if config.run_on_sims:
        # add splml so that mytweakml will be applied by multidraw
        polyml = False
        for l in lcs:
            if l.ml is None:
                print('Adding flat ML')
                pycs3.gen.splml.addtolc(l, n=2)
            elif l.ml.mltype == 'poly':
                polyml = True
                print('Poly ML : Using the saved generative curve instead')

        if polyml:
            lcs, spline = pycs3.gen.util.readpickle(
                'initopt_%s_ks%i_%s%i_generative_polyml.pkl' % (dataname, kn, string_ML, ml))
            pycs3.sim.draw.saveresiduals(lcs, spline)

        # import the module with the parameter of the noise :
        print('I will use the parameter from : %s' % ('tweakml_' + config.tweakml_name + '.py'))
        exec(compile(open('tweakml_' + config.tweakml_name + '.py', "rb").read(),
                     'tweakml_' + config.tweakml_name + '.py', 'exec'), globals())

        files_mock = glob.glob("sims_" + config.simset_mock + '/*.pkl')
        pycs3.sim.draw.multidraw(lcs, spline, onlycopy=False, n=config.nsim, npkl=config.nsimpkls,
                                 simset=config.simset_mock, tweakml=tweakml_list,
                                 shotnoise=config.shotnoise_type, trace=False,
                                 truetsr=config.truetsr, shotnoisefrac=1.0, scaletweakresi=False)
    os.chdir(current_dir)


def draw_mock_para_aux(arguments):
    return draw_mock_para(*arguments)


def main(lensname, dataname, work_dir='./'):
    import importlib
    sys.path.append(work_dir + "config/")
    config = importlib.import_module("config_" + lensname + "_" + dataname)

    if config.max_core is None:
        processes = multiprocess.cpu_count()
    else:
        processes = config.max_core

    p = multiprocess.Pool(processes=processes)
    print("Running on %i cores. " % processes)
    job_args = []

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
        raise RuntimeError("I dont know your microlensing type. Choose 'polyml' or 'spml''.")

    for i, kn in enumerate(config.knotstep):
        for j, ml in enumerate(ml_param):
            for simset in [config.simset_mock, config.simset_copy]:
                file = glob.glob(os.path.join(config.lens_directory + config.combkw[i, j], "sims_" + simset + '/*.pkl'))
                if len(file) != 0 and config.askquestions == True:
                    while True:
                        answer = int(input(
                            "You already have files in the folder %s. Do you want to add more (1) or replace the existing file (2) ? (1/2)" % simset))
                        if answer != 1 or answer != 2:
                            break
                        else:
                            print("I did not understand your answer.")

                    if answer == 1:
                        print("OK, deleting everything ! ")
                        for f in file:
                            os.remove(f)
                    elif answer == 2:
                        print("OK, I'll add more mocks !")
                elif len(file) != 0:
                    print(
                        "You already have files in the folder %s. You did not turn your ask question flag. By default, I will replace your simulation !" % simset)
                    print("Warning : I am not deleting the optimised curves, you might want to delete them manually.")
                    for f in file:
                        os.remove(f)
                    print("OK, deleted previous simulations ! ")

            job_args.append((i, j, kn, ml, string_ML, lensname, dataname, work_dir))
    if processes > 1:
        p.map(draw_mock_para_aux, job_args)
    else:
        for args in job_args:
            draw_mock_para(*args)
    print("Done.")


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Prepare the copies of the light curves and draw some mock curves.",
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
