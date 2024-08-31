#!/usr/bin/python3
#!/usr/bin/python3
#PBS -v PYTHONPATH
#PBS -l nodes=bead59
#PBS -o out/$PBS_JOBNAME.out
#PBS -j oe

from argparse import ArgumentParser
import os
import sys

if 'PBS_O_WORKDIR' in os.environ:
    working_dir = os.environ["PBS_O_WORKDIR"]
    os.chdir(working_dir)
    sys.path.append(".")
    sys.path.append("../rumdpy-dev/")


import dataclasses
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import rumdpy as rp
from numba import cuda
from tools import KNOWN_SIMULATIONS, SimulationParameters, SimulationVsNVT, load_conf_from_npz


for i in range(3):
    try:
        print("Trying device", i)
        cuda.select_device(i)
        break
    except Exception:
        pass


def poly(params: SimulationParameters, output_path: str):
    delta_test = np.linspace(0.0,0.99, 50)
    ntypes: int = params.pair_potential_params["ntypes"]  # type: ignore
    delta: float
    if params.pair_potential_name == "LJ-eps_poly":
        delta = params.pair_potential_params["d_eps"]  # type: ignore
    elif params.pair_potential_name == "LJ-sig_poly":
        delta = params.pair_potential_params["d_sig"]  # type: ignore
    else:
        assert False, "Unreachable"
    delta_test = np.sort(np.concatenate([delta_test, [delta]]))

    output = rp.tools.load_output(params.nvu_output)
    configuration, _target_u = load_conf_from_npz(params.nvu_eq_conf_output)

    configuration.copy_to_device()

    pair_func = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
    sig0, eps0, _cut = params.get_pair_potential_params()
    evaluators = []
    for i in range(delta_test.shape[0]):
        delta_test_i = delta_test[i]
        if params.pair_potential_name == "LJ-eps_poly":
            eps_pp = np.diag(eps0)
            if delta == 0:
                eps_pp_new = (np.arange(ntypes)/(ntypes-1)*2 - 1)*delta_test_i + 1
            else:
                eps_pp_new = (eps_pp - 1) / delta * delta_test_i + 1
            sig = np.ones((ntypes, ntypes))*params.pair_potential_params["sig"]
            eps = np.sqrt(np.outer(eps_pp_new, eps_pp_new))
        elif params.pair_potential_name == "LJ-sig_poly":
            sig_pp = np.diag(sig0)
            if delta == 0:
                sig_pp_new = (np.arange(ntypes)/(ntypes-1)*2 - 1)*delta_test_i + 1
            else:
                sig_pp_new = (sig_pp - 1) / delta * delta_test_i + 1
            sig = np.add.outer(sig_pp_new, sig_pp_new) / 2
            eps = np.ones((ntypes, ntypes))*params.pair_potential_params["eps"]
        else:
            assert False, "Unreachable"
        cut = np.array(sig)*2.5
        pair_pot = rp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)
        ev = rp.Evaluater(configuration, pair_pot)
        evaluators.append(ev)

    fsq_all =  [[] for _ in range(len(delta_test))]
    u_all = [[] for _ in range(len(delta_test))]
    Tconf_all = [[] for _ in range(len(delta_test))]
    positions = output['block'][:,:,0,:,:]

    for block_now in range(positions.shape[0]):
        for j in range(4):
            pos = positions[block_now, -(j+1), :, :]
            configuration['r'] = pos
            
            for i in range(len(delta_test)):
                evaluators[i].evaluate()
                f = evaluators[i].configuration['f']
                lap = evaluators[i].configuration['lap']
                Tconf = np.sum(f**2) / np.sum(lap)

                Tconf_all[i].append(Tconf)
                u_all[i].append(np.sum(evaluators[i].configuration['u']))
                fsq_all[i].append(np.sum(f**2))


    u_all = np.array(u_all)
    fsq_all = np.array(fsq_all)
    Tconf_all = np.array(Tconf_all)
    np.savez(output_path, u=u_all, fsq=fsq_all, tconf=Tconf_all, delta_test=delta_test)


def plot_poly(params: SimulationParameters, output_path: str) -> plt.Figure:
    fig = plt.figure(figsize=(12, 16))
    gs = gridspec.GridSpec(6, 1, hspace=.5)

    delta: float
    if params.pair_potential_name == "LJ-eps_poly":
        delta = params.pair_potential_params["d_eps"]  # type: ignore
        kind = "epsilon"
    elif params.pair_potential_name == "LJ-sig_poly":
        delta = params.pair_potential_params["d_sig"]  # type: ignore
        kind = "sigma"
    else:
        assert False, "Unreachable"
    threshold: float = params.nvu_params_threshold

    data = np.load(output_path)
    u = data["u"]
    s = u.std(axis=1)[:, np.newaxis]
    if "delta_eps_test" in data:
        data["delta_test"] = data["delta_eps_test"]
        del data["delta_eps_test"]
        np.savez(output_path, **data)
    delta_test = data["delta_test"]

    ax = fig.add_subplot(gs[0])
    ax.axvline(delta, color="black", linestyle='--', linewidth=1, alpha=.5)
    for i in range(u.shape[1]):
        u_i = u[:, i]
        ax.plot(delta_test, u_i, alpha=.5)
    ax.set_title(rf"$U(\delta)$")
    ax.set_xlabel(rf"$\delta_{{\{kind}}}$")

    ax = fig.add_subplot(gs[1])
    ax.axvline(delta, color="black", linestyle='--', linewidth=1, alpha=.5)
    du = (u - u.mean(axis=1)[:, np.newaxis]) / np.abs(u.mean(axis=1)[:, np.newaxis])
    for i in range(u.shape[1]):
        du_i = du[:, i]
        ax.plot(delta_test, du_i, alpha=.5)
    ax.set_title(rf"threshold = ${threshold:.0e}$; $\frac{{U(\delta) - \overline{{U}}(\delta)}}{{|\overline{{U}}(\delta)|}}$")
    ax.set_xlabel(rf"$\delta_{{\{kind}}}$")

    ax = fig.add_subplot(gs[2])
    ax.axvline(delta, color="black", linestyle='--', linewidth=1, alpha=.5)
    du = (u - u.mean(axis=1)[:, np.newaxis]) / s
    for i in range(u.shape[1]):
        du_i = du[:, i]
        ax.plot(delta_test, du_i, alpha=.5)
    ax.set_title(rf"$\frac{{U(\delta) - \overline{{U}}(\delta)}}{{\sigma_{{U}}(\delta)}}$")
    ax.set_xlabel(rf"$\delta_{{\{kind}}}$")

    ax = fig.add_subplot(gs[3])
    ax.axvline(delta, color="black", linestyle='--', linewidth=1, alpha=.5)
    for i in range(u.shape[1]):
        du_i = du[:, i]
        if i > 10:
            continue
        ax.plot(delta_test, du_i, linewidth=1, color="black", alpha=.2)
    for i in range(u.shape[1]):
        du_i = du[:, i]
        if i > 10:
            continue
        ax.scatter(delta_test, du_i, 2, alpha=.9)
    ax.set_title(rf"$\frac{{U(\delta) - \overline{{U}}(\delta)}}{{\sigma_{{U}}(\delta)}}$")
    ax.set_xlabel(rf"$\delta_{{\{kind}}}$")

    ax = fig.add_subplot(gs[3])
    ax.axvline(delta, color="black", linestyle='--', linewidth=1, alpha=.5)
    for i in range(u.shape[1]):
        du_i = du[:, i]
        if i > 20:
            continue
        ax.plot(delta_test, du_i, linewidth=1, color="black", alpha=.1)
    for i in range(u.shape[1]):
        du_i = du[:, i]
        if i > 20:
            continue
        ax.scatter(delta_test, du_i, 3, alpha=.9)
    ax.set_title(rf"$\frac{{U(\delta) - \overline{{U}}(\delta)}}{{\sigma_{{U}}(\delta)}}$")
    ax.set_xlabel(rf"$\delta_{{\{kind}}}$")
    ax.set_xlim(delta-.05, delta + .05)

    ax = fig.add_subplot(gs[4])
    ax.axvline(delta, color="black", linestyle='--', linewidth=1, alpha=.5)
    for i in range(u.shape[1]):
        du_i = du[:, i]
        if i > 20:
            continue
        ax.plot(delta_test, du_i, linewidth=1, color="black", alpha=.1)
    for i in range(u.shape[1]):
        du_i = du[:, i]
        if i > 20:
            continue
        ax.scatter(delta_test, du_i, 3, alpha=.9)
    ax.set_title(rf"$U$")
    ax.set_xlabel(rf"$\delta_{{\{kind}}}$")
    ax.set_xlim(delta-.05, delta + .05)
    return fig


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("-r", "--run_new", action="store_true")
    parser.add_argument("-s", "--show", action="store_true")
    parser.add_argument("-o", "--save_figure", action="store_true")
    if 'PBS_O_WORKDIR' in os.environ:
        flags = os.environ.get("flags", "")
        raw_args = flags.split()
    else:
        raw_args = None
    args = parser.parse_args(raw_args)

    for name in args.names:
        params = KNOWN_SIMULATIONS.get(name, None)
        if params is None:
            print(f"ERROR: simulation with name {name} does not exist")
            exit(1)
        if params.pair_potential_name not in ("LJ-eps_poly", "LJ-sig_poly"):
            print(f"ERROR: Expected simulation with name {name} to use potential `LJ-sig_poly` but got `{params.pair_potential_name}`")
            exit(1)

    for name in args.names:
        params = KNOWN_SIMULATIONS[name]
        kind = "eps" if params.pair_potential_name == "LJ-eps_poly" else "sig"
        folder = f"isomorph_check/{kind}"
        if not os.path.exists(folder):
            os.makedirs(folder)

        output_path = os.path.join(folder, f"data_{params.name}.npz")
        if not os.path.exists(output_path) or args.run_new:
            poly(params, output_path)

        fig = plot_poly(params, output_path)
        if args.save_figure:
            fig.savefig(os.path.join(folder, f"{params.name}.png"))

    if args.show:
        plt.show()


POLY_EPS_0_5_TEST = SimulationVsNVT(
    name="Polydisperse_eps_0_5_test",
    description=
"""Lennard-Jones. 
Polydisperse_eps system""",
    root_folder="output/NVU_RT_VS/",
    rho=0.844,
    temperature=1,
    cells=[8, 8, 8],
    tau=0.2,
    dt=0.005,
    # steps=4,
    # steps_per_timeblock=2,
    # scalar_output=0,
    steps=2**21,
    steps_per_timeblock=2**16,
    scalar_output=2**9,

    pair_potential_name="LJ-eps_poly",
    pair_potential_params={ "d_eps": 0, "sig": 1, "ntypes": 256 },

    nvu_params_max_abs_val=2,
    # 10^-3 is the safest option
    nvu_params_threshold=1e-5,
    nvu_params_eps=5e-6,
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
    POLY_EPS_0_5_TEST,
    name="Polydisperse_sig_8_5",
    pair_potential_name="LJ-sig_poly",
    nvu_params_threshold=1e-5,
    pair_potential_params={ "d_sig": 0.8, "eps": 1, "ntypes": 256 },
)
dataclasses.replace(
    POLY_EPS_0_5_TEST,
    name="Polydisperse_sig_8_3",
    pair_potential_name="LJ-sig_poly",
    nvu_params_threshold=1e-3,
    pair_potential_params={ "d_sig": 0.8, "eps": 1, "ntypes": 256 },
)
dataclasses.replace(
    POLY_EPS_0_5_TEST,
    name="Polydisperse_sig_8_2",
    pair_potential_name="LJ-sig_poly",
    nvu_params_threshold=1e-2,
    pair_potential_params={ "d_sig": 0.8, "eps": 1, "ntypes": 256 },
)
dataclasses.replace(
    POLY_EPS_0_5_TEST,
    name="Polydisperse_sig_0_5",
    pair_potential_name="LJ-sig_poly",
    nvu_params_threshold=1e-5,
    pair_potential_params={ "d_sig": 0.0, "eps": 1, "ntypes": 256 },
)
dataclasses.replace(
    POLY_EPS_0_5_TEST,
    name="Polydisperse_sig_0_3",
    pair_potential_name="LJ-sig_poly",
    nvu_params_threshold=1e-3,
    pair_potential_params={ "d_sig": 0.0, "eps": 1, "ntypes": 256 },
)
dataclasses.replace(
    POLY_EPS_0_5_TEST,
    name="Polydisperse_sig_0_2",
    pair_potential_name="LJ-sig_poly",
    nvu_params_threshold=1e-2,
    pair_potential_params={ "d_sig": 0.0, "eps": 1, "ntypes": 256 },
)
dataclasses.replace(
    POLY_EPS_0_5_TEST,
    name="Polydisperse_eps_8_6",
    pair_potential_name="LJ-eps_poly",
    nvu_params_threshold=1e-6,
    nvu_params_eps=5e-7,
    pair_potential_params={ "d_eps": 0.8, "sig": 1, "ntypes": 256 },
)
dataclasses.replace(
    POLY_EPS_0_5_TEST,
    name="Polydisperse_eps_8_5",
    pair_potential_name="LJ-eps_poly",
    nvu_params_threshold=1e-5,
    pair_potential_params={ "d_eps": 0.8, "sig": 1, "ntypes": 256 },
)
dataclasses.replace(
    POLY_EPS_0_5_TEST,
    name="Polydisperse_eps_8_3",
    pair_potential_name="LJ-eps_poly",
    nvu_params_threshold=1e-3,
    pair_potential_params={ "d_eps": 0.8, "sig": 1, "ntypes": 256 },
)
dataclasses.replace(
    POLY_EPS_0_5_TEST,
    name="Polydisperse_eps_8_2",
    pair_potential_name="LJ-eps_poly",
    nvu_params_threshold=1e-2,
    pair_potential_params={ "d_eps": 0.8, "sig": 1, "ntypes": 256 },
)
dataclasses.replace(
    POLY_EPS_0_5_TEST,
    name="Polydisperse_eps_0_6",
    pair_potential_name="LJ-eps_poly",
    nvu_params_threshold=1e-6,
    nvu_params_eps=5e-7,
    pair_potential_params={ "d_eps": 0.0, "sig": 1, "ntypes": 256 },
)
dataclasses.replace(
    POLY_EPS_0_5_TEST,
    name="Polydisperse_eps_0_5",
    pair_potential_name="LJ-eps_poly",
    nvu_params_threshold=1e-5,
    pair_potential_params={ "d_eps": 0.0, "sig": 1, "ntypes": 256 },
)
dataclasses.replace(
    POLY_EPS_0_5_TEST,
    name="Polydisperse_eps_0_3",
    pair_potential_name="LJ-eps_poly",
    nvu_params_threshold=1e-3,
    pair_potential_params={ "d_eps": 0.0, "sig": 1, "ntypes": 256 },
)
dataclasses.replace(
    POLY_EPS_0_5_TEST,
    name="Polydisperse_eps_0_2",
    pair_potential_name="LJ-eps_poly",
    nvu_params_threshold=1e-2,
    pair_potential_params={ "d_eps": 0.0, "sig": 1, "ntypes": 256 },
)


if __name__ == "__main__":
    main()

