#!/usr/bin/python3
#PBS -v PYTHONPATH
#PBS -l nodes=bead59
#PBS -o out/$PBS_JOBNAME.out
#PBS -j oe

import dataclasses
import os
import sys

# if running as batch job need to explicitly change to the correct directory
if 'PBS_O_WORKDIR' in os.environ:
    working_dir = os.environ["PBS_O_WORKDIR"]
    os.chdir(working_dir)
    sys.path.append(".")
    sys.path.append("../rumdpy-dev/")

"""
Test if NVU RT is equivalent to NVT or NVE

Lennard-Jones potential
"""
import os
import sys
from typing import Any, Dict, Literal, Union, List
import numpy as np
import numpy.typing as npt
from argparse import ArgumentParser
from numba import cuda
import matplotlib.pyplot as plt
from tools import *
import NVU_RT_paper
import rumdpy as rp

# This module is used to add some simulations to the known simulations list
_ = NVU_RT_paper 


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("sims", help="Simulation to run", nargs="+")
    parser.add_argument("-a", "--all", help="Eq to -teu", action="store_true")
    parser.add_argument("-t", "--other", help="Run Other (NVT or NVE)", action="store_true")
    parser.add_argument("-p", "--plot-figures", help="Plot Figures", action="store_true")
    parser.add_argument("-u", "--nvu_rt", help="Run NVU_RT", action="store_true")
    parser.add_argument("-e", "--nvu_rt_eq", help="Run NVU_RT eq stage", action="store_true")
    parser.add_argument("-s", "--show_figures", help="Show Figures", action="store_true")
    parser.add_argument("-o", "--output_pdf", help="Output PDF", action="store_true")
    parser.add_argument("-d", "--device", help="Select NVIDIA device", type=int, default=None)
    parser.add_argument("-n", "--names", help="Saves the simulation with a different name", nargs="*", default=None)

    if 'PBS_O_WORKDIR' in os.environ:
        name = os.environ.get("PBS_JOBNAME", None)
        flags = os.environ.get("flags", "")
        if name is None:
            print("ERROR: you must pass a name")
            exit(1)
        args_str = f"{name.replace('|', ' ')} {flags}"
        raw_args = args_str.split()
    else:
        raw_args = None
    args = parser.parse_args(raw_args)

    if args.device is not None:
        print("Using device:", args.device)
        cuda.select_device(args.device)
    else:
        for i in range(3):
            try:
                cuda.select_device(i)
                print(f"Using device {i}")
                break
            except Exception:
                pass
        else:
            cuda.select_device(0)

    for sim in args.sims:
        if sim == "list":
            print("\n".join(f"- `{x}`" for x in KNOWN_SIMULATIONS.keys()))
            exit(0)

    params = []
    for sim in args.sims:
        p = KNOWN_SIMULATIONS.get(sim, None)
        if p is None:
            print(f"ERROR: No simulation named `{sim}`", file=sys.stderr)
            exit(1)
        params.append(p)

    if args.names is not None:
        if len(args.names) != len(args.sims):
            print(f"ERROR: Got {len(args.sims)} simulations but there where only {len(args.names)} passed.", file=sys.stderr)
            exit(1)
        for i, name in enumerate(args.names):
            if name in KNOWN_SIMULATIONS:
                print(f"ERROR: Can not run simulation {args.sims[i]} with name `{name}` because simulation already exists", file=sys.stderr)
                exit(1)
            params[i] = dataclasses.replace(params[i], name=name)

    for p in params:
        p.init()
        print(p.info())

        if args.other or args.all:
            run_NVTorNVE(p)
        if args.nvu_rt or args.all:
            run_NVU_RT(p, args.nvu_rt_eq or args.all)
        if args.plot_figures:
            plot_nvu_vs_figures(p)

        if args.output_pdf:
            save_current_figures_to_pdf(p.output_figs)

    if args.show_figures:
        plt.show()

asd_parameters: Dict[str, Union[npt.NDArray[np.float32], float]] = { 
    "eps": np.array([[1.000, 0.894],
                     [0.894, 0.788]], dtype=np.float32), 
    "sig": np.array([[1.000, 0.342],
                     [0.342, 0.117]], dtype=np.float32), 
    "bonds": np.array([[0.584, 3000.], ]),
    "b_mass": 0.195,
}
asd_parameters["cut"] = 2.5 * asd_parameters["sig"]

kob_andersen_parameters: Dict[str, Union[npt.NDArray[np.float32], float]] = { 
    "eps": np.array([[1.00, 1.50],
                     [1.50, 0.50]], dtype=np.float32), 
    "sig": np.array([[1.00, 0.80],
                     [0.80, 0.88]], dtype=np.float32), 
}
kob_andersen_parameters["cut"] = 2.5 * kob_andersen_parameters["sig"]


SimulationVsNVE(
    name="NVE_no-inertia_big",
    description=
"""LJ
See what changes if instead of reflecting velocities we use as new valocites direction the direction of the force""",
    root_folder="output/NVU_RT_VS/",
    rho=0.844,
    temperature=1,
    cells=[16, 16, 16],
    dt=0.005,
    steps=2**20,
    steps_per_timeblock=2**15,
    scalar_output=64,

    pair_potential_name="LJ",
    pair_potential_params={ "eps": 1, "sig": 1, "cut": 2.5 },

    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-6,
    nvu_params_eps=5e-7,
    nvu_params_max_steps=1000,
    nvu_params_max_initial_step_corrections=8,
    nvu_params_initial_step=0.001,
    nvu_params_initial_step_if_high=.0001,
    nvu_params_step=0.0001,
    nvu_params_mode="no-inertia",
)

SimulationVsNVE(
    name="NVE_no-inertia_long",
    description=
"""LJ
See what changes if instead of reflecting velocities we use as new valocites direction the direction of the force""",
    root_folder="output/NVU_RT_VS/",
    rho=0.844,
    temperature=1,
    cells=[8, 8, 8],
    dt=0.005,
    steps=2**20,
    steps_per_timeblock=2**15,
    scalar_output=64,

    pair_potential_name="LJ",
    pair_potential_params={ "eps": 1, "sig": 1, "cut": 2.5 },

    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-6,
    nvu_params_eps=5e-7,
    nvu_params_max_steps=1000,
    nvu_params_max_initial_step_corrections=8,
    nvu_params_initial_step=0.001,
    nvu_params_initial_step_if_high=.0001,
    nvu_params_step=0.0001,
    nvu_params_mode="no-inertia",
)

SimulationVsNVE(
    name="NVE_no-inertia",
    description=
"""LJ
See what changes if instead of reflecting velocities we use as new valocites direction the direction of the force""",
    root_folder="output/NVU_RT_VS/",
    rho=0.844,
    temperature=1,
    cells=[8, 8, 8],
    dt=0.005,
    steps=2**15,
    steps_per_timeblock=2**10,
    scalar_output=16,

    pair_potential_name="LJ",
    pair_potential_params={ "eps": 1, "sig": 1, "cut": 2.5 },

    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-6,
    nvu_params_eps=5e-7,
    nvu_params_max_steps=1000,
    nvu_params_max_initial_step_corrections=8,
    nvu_params_initial_step=0.001,
    nvu_params_initial_step_if_high=.0001,
    nvu_params_step=0.0001,
    nvu_params_mode="no-inertia",
)

SimulationVsNVE(
    name="NVE_KobAndersen_paper_IV",
    description=
"""Kob Andersen
Replicate results of paper `NVU dynamics. II. Comparing to four other dynamics`.
Figure 7. (In reality just trying to have some intermediate temperatures to compare)
""",
    root_folder="output/NVU_RT_VS/",
    rho=1.2,
    temperature=0.50,
    cells=[8, 8, 8],
    dt=0.005,
    steps=2**23,
    steps_per_timeblock=2**15,
    scalar_output=64,

    pair_potential_name="Kob-Andersen",
    pair_potential_params=kob_andersen_parameters,

    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-6,
    nvu_params_eps=5e-7,
    nvu_params_max_steps=1000,
    nvu_params_max_initial_step_corrections=8,
    nvu_params_initial_step=0.001,
    nvu_params_initial_step_if_high=.0001,
    nvu_params_step=0.0001,
)


SimulationVsNVE(
    name="NVE_KobAndersen_paper_III",
    description=
"""Kob Andersen
Replicate results of paper `NVU dynamics. II. Comparing to four other dynamics`.
Figure 7.
""",
    root_folder="output/NVU_RT_VS/",
    rho=1.2,
    temperature=0.44,
    cells=[8, 8, 8],
    dt=0.005,
    steps=2**23,
    steps_per_timeblock=2**15,
    scalar_output=64,

    pair_potential_name="Kob-Andersen",
    pair_potential_params=kob_andersen_parameters,

    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-6,
    nvu_params_eps=5e-7,
    nvu_params_max_steps=1000,
    nvu_params_max_initial_step_corrections=8,
    nvu_params_initial_step=0.001,
    nvu_params_initial_step_if_high=.0001,
    nvu_params_step=0.0001,
)

SimulationVsNVE(
    name="NVE_KobAndersen_paper_II",
    description=
"""Kob Andersen
Replicate results of paper `NVU dynamics. II. Comparing to four other dynamics`.
Figure 2b.
""",
    root_folder="output/NVU_RT_VS/",
    rho=1.2,
    temperature=0.405,
    cells=[8, 8, 8],
    dt=0.005,
    steps=2**23,
    steps_per_timeblock=2**15,
    scalar_output=64,

    pair_potential_name="Kob-Andersen",
    pair_potential_params=kob_andersen_parameters,

    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-6,
    nvu_params_eps=5e-7,
    nvu_params_max_steps=1000,
    nvu_params_max_initial_step_corrections=8,
    nvu_params_initial_step=0.001,
    nvu_params_initial_step_if_high=.0001,
    nvu_params_step=0.0001,
)


SimulationVsNVE(
    name="NVE_KobAndersen_paper_I",
    description=
"""Kob Andersen
Replicate results of paper `NVU dynamics. II. Comparing to four other dynamics`.
Figure 2a.
""",
    root_folder="output/NVU_RT_VS/",
    rho=1.2,
    temperature=2,
    cells=[8, 8, 8],
    dt=0.005,
    steps=2**23,
    steps_per_timeblock=2**15,
    scalar_output=64,

    pair_potential_name="Kob-Andersen",
    pair_potential_params=kob_andersen_parameters,

    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-6,
    nvu_params_eps=5e-7,
    nvu_params_max_steps=1000,
    nvu_params_max_initial_step_corrections=8,
    nvu_params_initial_step=0.001,
    nvu_params_initial_step_if_high=.0001,
    nvu_params_step=0.0001,
)


SimulationVsNVE(
    name="NVE_KobAndersen_long",
    description=
"""Kob Andersen
Comparision with NVE""",
    root_folder="output/NVU_RT_VS/",
    rho=0.844,
    temperature=1,
    cells=[8, 8, 8],
    dt=0.005,
    steps=2**23,
    steps_per_timeblock=2**15,
    scalar_output=64,

    pair_potential_name="Kob-Andersen",
    pair_potential_params=kob_andersen_parameters,

    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-7,
    nvu_params_eps=5e-8,
    nvu_params_max_steps=1000,
    nvu_params_max_initial_step_corrections=10,
    nvu_params_initial_step=0.01,
    nvu_params_initial_step_if_high=.0001,
    nvu_params_step=0.0001,
)

SimulationVsNVE(
    name="NVE_KobAndersen_long25",
    description=
"""Kob Andersen
Comparision with NVE. 2**25 timesteps """,
    root_folder="output/NVU_RT_VS/",
    rho=0.844,
    temperature=1,
    cells=[8, 8, 8],
    dt=0.005,
    steps=2**25,
    steps_per_timeblock=2**20,
    scalar_output=2**8,

    pair_potential_name="Kob-Andersen",
    pair_potential_params=kob_andersen_parameters,

    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-7,
    nvu_params_eps=5e-8,
    nvu_params_max_steps=1000,
    nvu_params_max_initial_step_corrections=10,
    nvu_params_initial_step=0.01,
    nvu_params_initial_step_if_high=.0001,
    nvu_params_step=0.0001,
)


SimulationVsNVE(
    name="NVE_KobAndersen_short",
    description=
"""Kob Andersen
Comparision with NVE""",
    root_folder="output/NVU_RT_VS/",
    rho=0.844,
    temperature=1,
    cells=[8, 8, 8],
    dt=0.005,
    steps=2**15,
    steps_per_timeblock=2**10,
    scalar_output=16,

    pair_potential_name="Kob-Andersen",
    pair_potential_params=kob_andersen_parameters,

    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-6,
    nvu_params_eps=5e-7,
    nvu_params_max_steps=1000,
    nvu_params_max_initial_step_corrections=8,
    nvu_params_initial_step=0.001,
    nvu_params_initial_step_if_high=.0001,
    nvu_params_step=0.0001,
)



SimulationVsNVE(
    name="NVE_long",
    description=
"""Lennard-Jones. 
Comparision with NVE""",
    root_folder="output/NVU_RT_VS/",
    rho=0.844,
    temperature=1,
    cells=[8, 8, 8],
    dt=0.005,
    steps=2**23,
    steps_per_timeblock=2**18,
    scalar_output=64,

    pair_potential_name="LJ",
    pair_potential_params={ "eps": 1, "sig": 1, "cut": 2.5 },

    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-7,
    nvu_params_eps=5e-8,
    nvu_params_max_steps=1000,
    nvu_params_max_initial_step_corrections=10,
    nvu_params_initial_step=0.01,
    nvu_params_initial_step_if_high=.0001,
    nvu_params_step=0.0001,
)

SimulationVsNVE(
    name="NVE_bigger_longer",
    description=
"""Lennard-Jones. 
Comparision with NVE""",
    root_folder="output/NVU_RT_VS/",
    rho=0.844,
    temperature=1,
    dt=0.005,
    steps=2**23,
    steps_per_timeblock=2**18,
    scalar_output=2**9,
    cells=[16, 16, 16],

    pair_potential_name="LJ",
    pair_potential_params={ "eps": 1, "sig": 1, "cut": 2.5 },

    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-6,
    nvu_params_eps=5e-7,
    nvu_params_max_steps=1000,
    nvu_params_max_initial_step_corrections=8,
    nvu_params_initial_step=0.001,
    nvu_params_initial_step_if_high=.0001,
    nvu_params_step=0.0001,
)

SimulationVsNVE(
    name="NVE_bigger",
    description=
"""Lennard-Jones. 
Comparision with NVE""",
    root_folder="output/NVU_RT_VS/",
    rho=0.844,
    temperature=1,
    dt=0.005,
    steps=2**16,
    steps_per_timeblock=2**11,
    scalar_output=16,
    cells=[16, 16, 16],

    pair_potential_name="LJ",
    pair_potential_params={ "eps": 1, "sig": 1, "cut": 2.5 },

    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-6,
    nvu_params_eps=5e-7,
    nvu_params_max_steps=1000,
    nvu_params_max_initial_step_corrections=8,
    nvu_params_initial_step=0.001,
    nvu_params_initial_step_if_high=.0001,
    nvu_params_step=0.0001,
)

SimulationVsNVE(
    name="NVE_short",
    description=
"""Lennard-Jones. 
Comparision with NVE""",
    root_folder="output/NVU_RT_VS/",
    rho=0.844,
    temperature=1,
    cells=[8, 8, 8],
    dt=0.005,
    steps=2**17,
    steps_per_timeblock=2**12,
    scalar_output=16,

    pair_potential_name="LJ",
    pair_potential_params={ "eps": 1, "sig": 1, "cut": 2.5 },

    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-6,
    nvu_params_eps=5e-7,
    nvu_params_max_steps=1000,
    nvu_params_max_initial_step_corrections=8,
    nvu_params_initial_step=0.001,
    nvu_params_initial_step_if_high=.0001,
    nvu_params_step=0.0001,
    nvu_params_save_path_u=True,
)

SimulationVsNVT(
    name="NVT_long",
    description=
"""Lennard-Jones. 
Find out if there is any correlation between the delta time variation
and the curvature of Omega or the configuration temperature.""",
    root_folder="output/NVU_RT_VS/",
    rho=0.844,
    temperature=1,
    cells=[8, 8, 8],
    tau=0.2,
    dt=0.005,
    steps=2**23,
    steps_per_timeblock=2**18,
    scalar_output=64,

    pair_potential_name="LJ",
    pair_potential_params={ "eps": 1, "sig": 1, "cut": 2.5 },

    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-7,
    nvu_params_eps=5e-8,
    nvu_params_max_steps=1000,
    nvu_params_max_initial_step_corrections=10,
    nvu_params_initial_step=0.01,
    nvu_params_initial_step_if_high=.0001,
    nvu_params_step=0.0001,
)

SimulationVsNVT(
    name="NVT_short",
    description=
"""Lennard-Jones. 
Find out if there is any correlation between the delta time variation
and the curvature of Omega or the configuration temperature.""",
    root_folder="output/NVU_RT_VS/",
    rho=0.844,
    temperature=1,
    cells=[8, 8, 8],
    tau=0.2,
    dt=0.005,
    steps=2**15,
    steps_per_timeblock=2**10,
    scalar_output=16,

    pair_potential_name="LJ",
    pair_potential_params={ "eps": 1, "sig": 1, "cut": 2.5 },

    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-6,
    nvu_params_eps=5e-7,
    nvu_params_max_steps=1000,
    nvu_params_max_initial_step_corrections=8,
    nvu_params_initial_step=0.001,
    nvu_params_initial_step_if_high=.0001,
    nvu_params_step=0.0001,
    nvu_params_save_path_u=True,
)

SimulationVsNVE(
    name="NVE_parabola",
    description=
"""Lennard-Jones. 
Find out if there is any correlation between the delta time variation
and the curvature of Omega or the configuration temperature.""",
    root_folder="output/NVU_RT_VS/",
    rho=0.844,
    temperature=1,
    cells=[8, 8, 8],
    dt=0.005,
    # steps=4,
    # steps_per_timeblock=2,
    # scalar_output=0,
    steps=2**15,
    steps_per_timeblock=2**10,
    scalar_output=16,

    pair_potential_name="LJ",
    pair_potential_params={ "eps": 1, "sig": 1, "cut": 2.5 },

    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-6,
    nvu_params_eps=5e-7,
    nvu_params_max_steps=100,
    nvu_params_max_initial_step_corrections=20,
    # I dont really know how to tune this parameter
    nvu_params_initial_step=1,
    nvu_params_initial_step_if_high=1e-3,
    nvu_params_step=0.0001,
    nvu_params_save_path_u=True,
    nvu_params_raytracing_method="parabola"
)

SimulationVsNVT(
    name="NVT_parabola",
    description=
"""Lennard-Jones. 
Find out if there is any correlation between the delta time variation
and the curvature of Omega or the configuration temperature.""",
    root_folder="output/NVU_RT_VS/",
    rho=0.844,
    temperature=1,
    cells=[8, 8, 8],
    tau=0.2,
    dt=0.005,
    # steps=4,
    # steps_per_timeblock=2,
    # scalar_output=0,
    steps=2**15,
    steps_per_timeblock=2**10,
    scalar_output=16,

    pair_potential_name="LJ",
    pair_potential_params={ "eps": 1, "sig": 1, "cut": 2.5 },

    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-6,
    nvu_params_eps=5e-7,
    nvu_params_max_steps=100,
    nvu_params_max_initial_step_corrections=20,
    # I dont really know how to tune this parameter
    nvu_params_initial_step=10,
    nvu_params_initial_step_if_high=1e-3,
    nvu_params_step=1,
    nvu_params_save_path_u=True,
    nvu_params_raytracing_method="parabola"
)


NVT_DEBUG = SimulationVsNVT(
    name="NVT_debug",
    description=
"""Lennard-Jones. 
Find out if there is any correlation between the delta time variation
and the curvature of Omega or the configuration temperature.""",
    root_folder="output/NVU_RT_VS/",
    rho=0.844,
    temperature=1,
    cells=[8, 8, 8],
    tau=0.2,
    dt=0.005,
    # steps=4,
    # steps_per_timeblock=2,
    # scalar_output=0,
    steps=2**15,
    steps_per_timeblock=2**10,
    scalar_output=16,

    pair_potential_name="LJ",
    pair_potential_params={ "eps": 1, "sig": 1, "cut": 2.5 },

    nvu_params_max_abs_val=2,
    # 10^-3 is the safest option
    nvu_params_threshold=1e-3,
    nvu_params_eps=5e-5,
    nvu_params_max_steps=10,
    nvu_params_max_initial_step_corrections=20,
    # I dont really know how to tune this parameter
    nvu_params_initial_step=0.5/0.844**(1/3),
    nvu_params_initial_step_if_high=1e-3,
    nvu_params_step=1,
    # nvu_params_save_path_u=True,
    nvu_params_raytracing_method="parabola",
)

dataclasses.replace(
    NVT_DEBUG,
    name="NVT_debug_low_rho",
    rho=0.45,
    nvu_params_initial_step=0.5/0.45**(1/3),
)

TB = SimulationVsNVT(
    name="NVT_benchmark_parabola",
    description="""BENCHMARK""",
    root_folder="output/benchmark/",
    rho=0.844,
    temperature=1,
    cells=[8, 8, 8],
    tau=0.2,
    dt=0.005,
    # steps=4,
    # steps_per_timeblock=2,
    # scalar_output=0,
    steps=2**23,
    steps_per_timeblock=2**18,
    scalar_output=2**11,

    pair_potential_name="LJ",
    pair_potential_params={ "eps": 1, "sig": 1, "cut": 2.5 },

    nvu_params_max_abs_val=2,
    # 10^-3 is the safest option
    nvu_params_threshold=1e-3,
    nvu_params_eps=5e-5,
    nvu_params_max_steps=10,
    nvu_params_max_initial_step_corrections=20,
    # I dont really know how to tune this parameter
    nvu_params_initial_step=0.5/0.844**(1/3),
    nvu_params_initial_step_if_high=1e-3,
    nvu_params_step=1,
    # nvu_params_save_path_u=True,
    nvu_params_raytracing_method="parabola",
)
dataclasses.replace(
    TB, name="NVT_benchmark_newton", 
    nvu_params_raytracing_method="parabola-newton",
)


EB = SimulationVsNVE(
    name="NVE_benchmark_parabola",
    description="""BENCHMARK""",
    root_folder="output/benchmark/",
    rho=0.844,
    temperature=1,
    cells=[8, 8, 8],
    dt=0.005,
    # steps=4,
    # steps_per_timeblock=2,
    # scalar_output=0,
    steps=2**23,
    steps_per_timeblock=2**18,
    scalar_output=2**11,

    pair_potential_name="LJ",
    pair_potential_params={ "eps": 1, "sig": 1, "cut": 2.5 },

    nvu_params_max_abs_val=2,
    # 10^-3 is the safest option
    nvu_params_threshold=1e-3,
    nvu_params_eps=5e-5,
    nvu_params_max_steps=10,
    nvu_params_max_initial_step_corrections=20,
    # I dont really know how to tune this parameter
    nvu_params_initial_step=0.5/0.844**(1/3),
    nvu_params_initial_step_if_high=1e-3,
    nvu_params_step=1,
    # nvu_params_save_path_u=True,
    nvu_params_raytracing_method="parabola",
)
dataclasses.replace(
    EB, name="NVE_benchmark_newton", 
    nvu_params_raytracing_method="parabola-newton",
)

SimulationVsNVT(
    name="NVT_ASD",
    description="""ASD""",
    root_folder="output/NVU_RT_VS/",
    rho=1.863,  # For ASD this is atomic density
    temperature=0.465,
    temperature_eq=rp.make_function_ramp(
        value0=10.000, x0=0.005 * 2**23 * (1 / 8),
        value1=0.465, x1=0.005 * 2**23 * (1 / 4)),
    cells=[8, 8, 8],
    tau=0.2,
    dt=0.005,
    # steps=4,
    # steps_per_timeblock=2,
    # scalar_output=0,
    steps=2**23,
    steps_per_timeblock=2**18,
    scalar_output=2**11,

    pair_potential_name="ASD",
    pair_potential_params=asd_parameters,

    nvu_params_max_abs_val=2,
    # 10^-3 is the safest option
    nvu_params_threshold=1e-3,
    nvu_params_eps=5e-5,
    nvu_params_max_steps=10,
    nvu_params_max_initial_step_corrections=20,
    # I dont really know how to tune this parameter
    nvu_params_initial_step=0.5/0.844**(1/3),
    nvu_params_initial_step_if_high=1e-3,
    nvu_params_step=1,
    # nvu_params_save_path_u=True,
    nvu_params_raytracing_method="parabola",
)

if __name__ == "__main__":
    main()

