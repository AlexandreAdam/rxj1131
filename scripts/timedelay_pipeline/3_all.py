"""
Script to run steps 3a,3b,3c and 3d, all at once.
"""

import sys

try:
    exec(compile(open('3a_generate_tweakml.py', "rb").read(), '3a_generate_tweakml.py', 'exec'))
except:
    print("Error in script 3a.")
    sys.exit()

try:
    exec(compile(open('3b_draw_copy_mocks.py', "rb").read(), '3b_draw_copy_mocks.py', 'exec'))
except:
    print("Error in script 3b.")
    sys.exit()

try:
    exec(compile(open('3c_optimise_copy_mocks.py', "rb").read(), '3c_optimise_copy_mocks.py', 'exec'))
except:
    print("Error in script 3c.")
    sys.exit()

try:
    exec(compile(open('3d_check_statistics.py', "rb").read(), '3d_check_statistics.py', 'exec'))
except:
    print("Error in script 3d.")
    sys.exit()
