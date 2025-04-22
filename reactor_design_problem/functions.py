import sys
import os
# import everything from utils which is the folder above 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from mesh_generation.coil_cross_section import create_mesh as create_mesh_cs
from flask import Flask, jsonify, request
import time
from datetime import datetime
import shutil

app = Flask(__name__)

@app.route("/cross_section", methods=["POST"])
def cross_section():
    """
    This endpoint is used to evaluate the CFD simulation for the reactor design problem.
    The API takes in a JSON object with the following format:
    {
        "x": List,
        "z": List, # optional
        "keep_files": Boolean,
        "cpus": Integer # optional
    }

    where x is a list of variables that define the cross section of the reactor,
    z is a list of fidelities, and cpus is the number of CPUs for parallel simulation.
    Depending on whether fidelities are given or not,
    the endpoint will perform a single fidelity or multi-fidelity optimisation.
    The endpoint returns a JSON object with the following format:
    {
        "obj": float,
        "cost": float # if z is present
    }

    Variable information (x):
    Specifically x contains 6 groups of 6 interpolating points
    that define the cross section of the reactor at 6 locations along the axial direction.
    The cross section is defined by a Polar Gaussian process
    that is interpolated between the 6 points.
    The bounds have been normalised between 0 and 1.

    Dimensionality: 36
    Bounds: [0,1]
    Type: Continuous

    Fidelity information (z):
    The fidelities define by the axial and radial resolution of the mesh
    (along the length and cross section respectively).
    The bounds have been normalised between 0 and 1.
    Officially these are discrete however we treat them as continuous.

    Dimensionality: 2
    Bounds: [0,1]
    Type: Continuous
    """

    # obtain data from request
    data = request.get_json()

    # check variables are in correct format
    if len(data["x"]) != 36:
        raise ValueError("There must be 36 decision variables.")

    # check if all values are between 0 and 1
    if not all(0 <= value <= 1 for value in data["x"]):
        raise ValueError("Variables out of bounds.")

    # variable bounds for un-normalisation
    x_bounds = [0.002, 0.004]
    # un-normalise variables
    x = list(data["x"])
    for i in range(len(x)):
        x[i] = x[i] * (x_bounds[1] - x_bounds[0]) + x_bounds[0]

    # check fidelities exist
    try:
        z = list(data["z"])
        z_flag = True
        if not all(0 <= value <= 1 for value in data["z"]):
            raise ValueError("Fidelities out of bounds.")
        if len(z) != 2:
            raise ValueError("There must be exactly 2 fidelities.")
    except KeyError:
        # if not then perform single fidelity optimisation
        # note: the single fidelity IS NOT the highest
        # this is to make the problem a little bit less time consuming
        # at the cost of a small amount of accuracy.
        z = [0.75, 0.75]
        z_flag = False

    # fidelity bounds for un-normalisation
    z_bounds = {}
    z_bounds["fid_axial"] = [15.55, 40.45]
    z_bounds["fid_radial"] = [1.55, 4.45]
    # un-normalise fidelity bounds and round to nearest integer
    z[1] = np.rint(
        (
            z[1] * (z_bounds["fid_radial"][1] - z_bounds["fid_radial"][0])
            + z_bounds["fid_radial"][0]
        )
    )
    z[0] = np.rint(
        (
            z[0] * (z_bounds["fid_axial"][1] - z_bounds["fid_axial"][0])
            + z_bounds["fid_axial"][0]
        )
    )

    # steady-flow conditions, note these can be changed if needed.
    a = 0
    f = 0
    re = 50

    # start timing for simulation cost
    start = time.time()
    ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") 
    ID = 'test'

    parallel = False
    print("Single CPU utilised")

    # mesh simulation
    print("Creating mesh...")
    create_mesh_cs(x, z, ID)
    # replace with operating conditions (here fixed)
    parse_conditions_given(ID, a, f, re)

    # run CFD and obtain tracer concentration at exit
    print("Running simulation...")
    times, values = run_cfd(ID, parallel)
    # calculate a penalised tanks-in-series
    N, penalty = calculate_N_clean(values, times, ID)
    if not data["keep_files"]:
        shutil.rmtree(ID)
    end = time.time()

    if z_flag == True:
        return jsonify({"obj": N - penalty, "cost": end - start})
    else:
        return jsonify({"obj": N - penalty})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
