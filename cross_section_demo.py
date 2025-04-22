import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import sample_bounds, parse_conditions_given, run_cfd, calculate_N_clean
import shutil
from mesh_generation.coil_cross_section import create_mesh
import time
from datetime import datetime
import numpy as np

# define the extent of the parameterisation
n_circ = 6
n_cross_section = 6
coils = 2
length = np.pi * 2 * 0.010391 * coils

# define the bounds on fidelities
z_bounds = {}
z_bounds["fid_axial"] = [15.55, 40.45]
z_bounds["fid_radial"] = [1.55, 4.45]

# and on cross-section variables...
x_bounds = {}
for i in range(n_circ):
    for j in range(n_cross_section):
        x_bounds["r_" + str(i) + "_" + str(j)] = [0.001, 0.004]


def eval_cfd(data: dict):
    # parse into a list of lists (each for one cross-section)
    x_list = []
    for i in range(n_circ):
        x_add = []
        for j in range(n_cross_section):
            x_add.append(data["r_" + str(i) + "_" + str(j)])

        x_list.append(np.array(x_add))

    # no pulsed-flow! (frequency = 0)
    a = 0
    f = 0
    re = 50
    start = time.time()

    # for pathname
    ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    z = [data["fid_axial"], data["fid_radial"]]

    # create mesh
    create_mesh(x_list, z, ID)
    parse_conditions_given(ID, a, f, re)

    # when run locally, should probably fail at this step!
    times, values = run_cfd(ID)
    N, penalty = calculate_N_clean(values, times, ID)
    shutil.rmtree(ID)
    end = time.time()
    return {"obj": N - penalty, "cost": end - start}


s = sample_bounds(z_bounds | x_bounds, 1, random=True)[0]
# create dictionary with keys and values
s_dict = {}
i = 0
for k, v in (z_bounds | x_bounds).items():
    s_dict[k] = s[i]
    i += 1

print(s_dict)  # dictionary containing both z and x values

eval_cfd(s_dict)
