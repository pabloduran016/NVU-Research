#!/usr/bin/python3
#PBS -v PYTHONPATH
#PBS -l nodes=bead53
#PBS -o out/$PBS_JOBNAME.out
#PBS -j oe

import os
import sys

# if running as batch job need to explicitly change to the correct directory
if 'PBS_O_WORKDIR' in os.environ:
    working_dir = os.environ["PBS_O_WORKDIR"]
    os.chdir(working_dir)
    sys.path.append(".")
    sys.path.append("../rumdpy-dev/")

import traceback
import dataclasses
from matplotlib.gridspec import GridSpec
from numba import cuda
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import rumdpy as rp
import os
from dataclasses import dataclass
from typing import Any, Dict, Set, Tuple, Optional, Union
from tools import KNOWN_SIMULATIONS, SimulationParameters, SimulationVsNVE, SimulationVsNVT, \
    load_conf_from_npz, plot_nvu_vs_figures, run_NVTorNVE, run_NVU_RT, \
    save_current_figures_to_pdf
import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float32]

@dataclass
class Output:
    target_u: float
    other_prod_output: Dict[str, Any]
    eq_conf0: rp.Configuration
    eq_output: Dict[str, Any]
    prod_conf0: rp.Configuration
    prod_output: Dict[str, Any]
    prod_rdf: Dict[str, Any]
    other_prod_rdf: Dict[str, Any]


def get_output(params: SimulationParameters, run_new: bool) -> Output:
    params.init()

    if type(params) == SimulationVsNVT:
        kind = "NVT"
        other_output_path = params.nvt_output
        eq_conf0_path = params.nvt_conf_output
    elif type(params) == SimulationVsNVE:
        kind = "NVE"
        other_output_path = params.nve_output
        eq_conf0_path = params.nve_conf_output
    else:
        raise ValueError("Expected vs NVE or vs NVT")

    if run_new or not os.path.exists(other_output_path) or not os.path.exists(eq_conf0_path):
        run_NVTorNVE(params)
    print(f"Loading {kind} production output from path `{other_output_path}`")
    other_prod_output = rp.tools.load_output(other_output_path)
    print(f"Loading NVU EQ initial configuration from path `{eq_conf0_path}`")
    eq_conf0, target_u = load_conf_from_npz(eq_conf0_path)

    if run_new or not os.path.exists(params.nvu_output):
        do_nvu_eq = True
        if not run_new and os.path.exists(params.nvu_eq_output) and os.path.exists(params.nvu_eq_conf_output):
            do_nvu_eq = False
        run_NVU_RT(params, do_nvu_eq)
    print(f"Loading NVU EQ output from path `{params.nvu_eq_output}`")
    nvu_eq_output = rp.tools.load_output(params.nvu_eq_output)
    print(f"Loading NVU PROD initial configuration from path `{params.nvu_eq_conf_output}`")
    prod_conf0, _target_u = load_conf_from_npz(params.nvu_eq_conf_output)
    print(f"Loading NVU PROD output from path `{params.nvu_output}`")
    nvu_prod_output = rp.tools.load_output(params.nvu_output)

    if run_new or not os.path.exists(params.output_figs):
        plot_nvu_vs_figures(params)
        save_current_figures_to_pdf(params.output_figs)
        plt.close("all")

    if run_new or not os.path.exists(params.other_prod_rdf):
        other_prod_rdf = get_rdf(other_prod_output)
        np.savez(params.other_prod_rdf, **other_prod_rdf)
    else:
        other_prod_rdf = np.load(params.other_prod_rdf)

    if run_new or not os.path.exists(params.nvu_prod_rdf):
        prod_rdf = get_rdf(nvu_prod_output)
        np.savez(params.nvu_prod_rdf, **prod_rdf)
    else:
        prod_rdf = np.load(params.nvu_prod_rdf)

    return Output(
        target_u=target_u,
        other_prod_output=other_prod_output,
        eq_conf0=eq_conf0,
        eq_output=nvu_eq_output,
        prod_conf0=prod_conf0,
        prod_output=nvu_prod_output,
        prod_rdf=prod_rdf,
        other_prod_rdf=other_prod_rdf,
    )


def get_delta_time(output: Dict[str, Any]) -> FloatArray:
    dt0, cos_v_f, = rp.extract_scalars(output, ["dt", "cos_v_f", ], 
        integrator_outputs=rp.integrators.NVU_RT.outputs)
    cos_v_f[cos_v_f > 1] = 1
    cos_v_f[cos_v_f < -1] = -1
    dt = dt0 * (np.pi / 2 - np.arccos(cos_v_f)) / cos_v_f
    return dt


def get_steps(output: Dict[str, Any]) -> FloatArray:
    nblocks, nscalar_per_block, _nscalars = output["scalars"].shape
    steps = np.arange(nblocks * nscalar_per_block) * output["steps_between_output"]
    return steps


def get_path_u(output: Dict[str, Any]) -> Tuple[FloatArray, FloatArray]:
    nblocks, npaths_per_block, npoints, n = output["path_u"].shape
    data = output["path_u"].reshape(nblocks * npaths_per_block, npoints, n)
    xs = data[:, :, 0].T
    ys = data[:, :, 1].T
    xs = xs/xs[[-1], :]
    return xs, ys


def get_msd(output: Dict[str, Any]) -> FloatArray:
    return rp.tools.calc_dynamics(output, first_block=0)["msd"][:, 0]


def get_rdf(output: Dict[str, Any]) -> Dict[str, FloatArray]:
    _, _, _, n, d = output["block"].shape
    positions = output["block"][:, :, 0, :, :]
    conf = rp.Configuration(D=d, N=n)
    conf['m'] = 1
    conf.ptype = output["ptype"]
    conf.simbox = rp.Simbox(D=d, lengths=output["attrs"]["simbox_initial"])
    cal_rdf = rp.CalculatorRadialDistribution(conf, num_bins=1000)
    for i in range(positions.shape[0]):
        pos = positions[i, -1, :, :]
        conf["r"] = pos
        conf.copy_to_device()
        cal_rdf.update()

    rdf = cal_rdf.read()
    return rdf


def method(run_new: Set[str]) -> None:
    ## DISTRIBUTION OF DELTA TIMES
    ##   - Plot \Delta t over steps
    output_n0 = get_output(LJ_N0, run_new=LJ_N0.name in run_new)
    n0 = output_n0.prod_output["block"].shape[3]
    output_n1 = get_output(LJ_N1, run_new=LJ_N1.name in run_new)
    n1 = output_n1.prod_output["block"].shape[3]
    output_n2 = get_output(LJ_N2, run_new=LJ_N2.name in run_new)
    n2 = output_n2.prod_output["block"].shape[3]
    n0_dt = get_delta_time(output_n0.prod_output)
    n0_steps = get_steps(output_n0.prod_output)
    n1_dt = get_delta_time(output_n1.prod_output)
    n1_steps = get_steps(output_n1.prod_output)
    n2_dt = get_delta_time(output_n2.prod_output)
    n2_steps = get_steps(output_n2.prod_output)

    n0_dt_raw, = rp.extract_scalars(output_n0.prod_output, ["dt"], integrator_outputs=rp.integrators.NVU_RT.outputs)
    n1_dt_raw, = rp.extract_scalars(output_n1.prod_output, ["dt"], integrator_outputs=rp.integrators.NVU_RT.outputs)
    n2_dt_raw, = rp.extract_scalars(output_n2.prod_output, ["dt"], integrator_outputs=rp.integrators.NVU_RT.outputs)

    fig = plt.figure(figsize=(10, 8))
    # fig.suptitle(r"Correction to $\Delta t$ to account for curvature fo the surface")
    gs0 = GridSpec(3, 1, hspace=.6)
    for (i, steps, dt, dt_raw, n) in zip(
        range(3), 
        (n0_steps, n1_steps, n2_steps), 
        (n0_dt, n1_dt, n2_dt),
        (n0_dt_raw, n1_dt_raw, n2_dt_raw),
        (n0, n1, n2)
    ):
        sub = fig.add_subfigure(gs0[i])
        sub.suptitle(rf"$N = {n}$")
        gs = GridSpec(1, 2, width_ratios=[1, 0.5], hspace=.6, wspace=0)

        ax0 = sub.add_subplot(gs[0])
        ax0.plot(steps, dt/dt_raw - 1, color="black", linewidth=.5, alpha=0.8)
        ax0.set_xlabel(r"$steps$")
        # ax0.set_ylabel(r"$\frac{\alpha}{\sin\alpha} - 1 = \frac{\Delta t}{\Delta t_{not\,corrected}} - 1$")
        ax0.grid(alpha=0.5)

        ax1 = sub.add_subplot(gs[1])
        ax1.hist(dt/dt_raw - 1, orientation="horizontal", density=True, color="black", alpha=0.8)
        ax1.set_xlabel(r"Probability")
        ax1.set_ylabel(r"$\frac{\alpha}{\sin\alpha} - 1 = \frac{\Delta t}{\Delta t_{not\,corrected}} - 1$")
        ax1.yaxis.set_label_position("right")
        ax1.yaxis.tick_right()
        ax1.grid(alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_RAW_DT_OVER_STEPS)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(n0_steps, n0_dt, linewidth=1, color="#BBB", alpha=0.8, label=rf"$N = {n0}$")
    ax.plot(n1_steps, n1_dt, linewidth=1, color="#555", alpha=0.8, label=rf"$N = {n1}$")
    ax.plot(n2_steps, n2_dt, linewidth=1, color="#000", alpha=0.8, label=rf"$N = {n2}$")
    ax.legend()
    ax.set_xlabel(r"$steps$")
    ax.set_ylabel(r"$\Delta t$")
    ax.grid(alpha=.5)
    fig.tight_layout()
    fig.savefig(FIG_DT_OVER_STEPS)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(n0_dt - n0_dt.mean(), bins=30, density=True, facecolor="#BBB", alpha=0.8, label=rf"$N = {n0}$")
    ax.hist(n1_dt - n1_dt.mean(), bins=30, density=True, facecolor="#555", alpha=0.8, label=rf"$N = {n1}$")
    ax.hist(n2_dt - n2_dt.mean(), bins=30, density=True, facecolor="#000", alpha=0.8, label=rf"$N = {n2}$")
    ax.set_xlabel(r"$\Delta t - \langle \Delta t\rangle$")
    ax.set_ylabel(r"Probability")
    ax.grid(alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DT_HIST)
    plt.close(fig)

    ## PARABOLA IN PATHS
    ##   - Plot Data + Fit
    ##   - Plot Error

    xs, ys = get_path_u(output_n1.prod_output)
    target_u = output_n0.target_u
    
    u = np.arange(xs.shape[0])
    p = np.polyfit(u, ys, 2)
    y_pred = p[0, :] * u[:, np.newaxis]**2 + p[1, :] * u[:, np.newaxis] + p[2, :]
    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 2, width_ratios=[1, .5], wspace=0)
    ax0 = fig.add_subplot(gs[0])
    error = (y_pred - ys) / ys
    ax0.plot(error.T.flatten()[::4], ".", color="black", alpha=0.2)
    ax0.set_xlabel("sample points")
    ax0.set_ylabel(r"$\frac{y_{pred} - y_i}{y_i}$", rotation=0, labelpad=20)
    ax0.grid(alpha=0.5)
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    ax1.hist(error.flatten(), orientation="horizontal", density=True, bins=30, color="black", alpha=0.8)
    ax1.set_xlabel(r"Probability")
    ax1.set_ylabel(r"$\frac{y_{pred} - y_i}{y_i}$", rotation=0, labelpad=20)
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()
    ax1.grid(alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_PARABOLAS_RELATIVE_ERROR)
    plt.close(fig)

    indices = np.argsort(ys.max(axis=0))[::-1][[0, 1, 2, 20]]
    print(indices)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot()
    for i, path_i in enumerate(indices):
        x_i = xs[:, path_i]
        y_i = ys[:, path_i]
        c = np.polyfit(x_i, y_i, 2)
        p = np.poly1d(c)
        xs_i = np.linspace(x_i[0], x_i[-1])
        line0, = ax.plot(xs_i, p(xs_i), '--', color="black", )
        line1, = ax.plot(x_i, y_i, ".", linewidth=0, alpha=.8,
                color="red", markeredgewidth=1.5,
                markersize=15, markeredgecolor="black")
        if i == 0:
            line0.set_label("Fit")
            line1.set_label("Data")
    ax.legend()
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$U$")
    ax.grid(alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_PARABOLAS)
    plt.close(fig)


def lennard_jones(run_new: Set[str]) -> None:
    output_a = get_output(LJ_A, run_new=LJ_A.name in run_new)
    output_b = get_output(LJ_B, run_new=LJ_B.name in run_new)
    output_c = get_output(LJ_C, run_new=LJ_C.name in run_new)

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 4, wspace=.5, hspace=.2)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[1, 1:3])
    for ax, output, params in (
        (ax_a, output_a, LJ_A),
        (ax_b, output_b, LJ_B),
        (ax_c, output_c, LJ_C)
    ):
        rdf = np.mean(output.prod_rdf["rdf"], axis=0)[::10]
        distances = output.prod_rdf["distances"][::10]
        other_rdf = np.mean(output.other_prod_rdf["rdf"], axis=0)
        ax.plot(output.other_prod_rdf["distances"], other_rdf, linewidth=1, color="black", label="NVT")
        ax.plot(distances, rdf, marker='.', linewidth=0, 
            markeredgewidth=1, markersize=9, markeredgecolor="black",
            color="red", alpha=.9, label="NVU_RT")
        ax.annotate(rf"temperature = ${params.temperature}$", (0.4, 0.62), xycoords="axes fraction")
        ax.annotate(rf"$\rho$ = ${params.rho}$", (0.4, 0.55), xycoords="axes fraction")
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_LJ_RDF)

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 4, wspace=.5, hspace=.2)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[1, 1:3])
    for ax, output, params in (
        (ax_a, output_a, LJ_A),
        (ax_b, output_b, LJ_B),
        (ax_c, output_c, LJ_C)
    ):
        msd = get_msd(output.prod_output)
        dt = get_delta_time(output.prod_output)
        time = np.mean(dt) * 2 ** np.arange(len(msd))

        other_msd = get_msd(output.other_prod_output)
        other_time = params.dt * 2 ** np.arange(len(other_msd))

        ax.loglog(other_time, other_msd, linewidth=1, color="black", label="NVT")
        ax.loglog(time, msd, marker='.', linewidth=0, 
                  markeredgewidth=1, markersize=9, markeredgecolor="black",
                  color="red", alpha=.9, label="NVU_RT")
        ax.annotate(rf"temperature = ${params.temperature}$", (0.5, 0.32), xycoords="axes fraction")
        ax.annotate(rf"$\rho$ = ${params.rho}$", (0.5, 0.25), xycoords="axes fraction")
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_LJ_MSD)


def kob_andersen(run_new: Set[str]) -> None:
    outputs = [
        get_output(params, run_new=params.name in run_new)
        for params in KA_PARAMS
    ]

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, wspace=.5, hspace=.2)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    for ax, output, params in (
        (ax_a, outputs[0], KA_PARAMS[0]),
        (ax_b, outputs[-1], KA_PARAMS[-1]),
    ):
        rdf = np.mean(output.prod_rdf["rdf"], axis=0)[::10]
        distances = output.prod_rdf["distances"][::10]
        other_rdf = np.mean(output.other_prod_rdf["rdf"], axis=0)
        ax.plot(output.other_prod_rdf["distances"], other_rdf, linewidth=1, color="black", label="NVT")
        ax.plot(distances, rdf, marker='.', linewidth=0, 
            markeredgewidth=1, markersize=9, markeredgecolor="black",
            color="red", alpha=.9, label="NVU_RT")
        ax.annotate(rf"temperature = ${params.temperature}$", (0.4, 0.62), xycoords="axes fraction")
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_KA_RDF)

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 3, wspace=.5, hspace=.2)
    ax = fig.add_subplot()
    for i, (output, params) in enumerate(zip(outputs, KA_PARAMS)):
        # ax = fig.add_subplot(gs[i])
        msd = get_msd(output.prod_output)
        dt = get_delta_time(output.prod_output)
        time = np.mean(dt) * 2 ** np.arange(len(msd))

        other_msd = get_msd(output.other_prod_output)
        other_time = params.dt * 2 ** np.arange(len(other_msd))

        line0, = ax.loglog(other_time, other_msd, linewidth=1, color="black")
        line1, = ax.loglog(time, msd, marker='.', linewidth=0, 
                           markeredgewidth=1, markersize=9, markeredgecolor="black",
                           color="red", alpha=.9)
        if i == 0:
            line0.set_label("NVT")
            line1.set_label("NVU_RT")
            ax.annotate(rf"temperature = ${params.temperature:.03f}$", (0.2, 0.62), xycoords="axes fraction")
        if i == len(outputs) - 1:
            ax.annotate(rf"temperature = ${params.temperature:.03f}$", (0.4, 0.32), xycoords="axes fraction")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_KA_MSD)


def no_inertia(run_new: Set[str]) -> None:
    output = get_output(NI, run_new=NI.name in run_new)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()
    rdf = np.mean(output.prod_rdf["rdf"], axis=0)[::10]
    distances = output.prod_rdf["distances"][::10]
    other_rdf = np.mean(output.other_prod_rdf["rdf"], axis=0)
    ax.plot(output.other_prod_rdf["distances"], other_rdf, linewidth=1, color="black", label="NVT")
    ax.plot(distances, rdf, marker='.', linewidth=0, 
        markeredgewidth=1, markersize=9, markeredgecolor="black",
        color="red", alpha=.9, label="NVU_RT (no inertia)")
    ax.annotate(rf"temperature = ${NI.temperature}$", (0.4, 0.62), xycoords="axes fraction")
    ax.annotate(rf"$\rho$ = ${NI.rho}$", (0.4, 0.55), xycoords="axes fraction")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_NI_RDF)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()
    msd = get_msd(output.prod_output)
    dt = get_delta_time(output.prod_output)
    time = np.mean(dt) * 2 ** np.arange(len(msd))

    other_msd = get_msd(output.other_prod_output)
    other_time = NI.dt * 2 ** np.arange(len(other_msd))

    ax.loglog(other_time, other_msd, linewidth=1, color="black", label="NVT")
    ax.loglog(time, msd, marker='.', linewidth=0, 
              markeredgewidth=1, markersize=9, markeredgecolor="black",
              color="red", alpha=.9, label="NVU_RT (no inertia)")
    ax.annotate(rf"temperature = ${NI.temperature}$", (0.5, 0.32), xycoords="axes fraction")
    ax.annotate(rf"$\rho$ = ${NI.rho}$", (0.5, 0.25), xycoords="axes fraction")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_NI_MSD)



def main(
    run_method: bool, 
    run_lj: bool, 
    run_ka: bool, 
    run_no_inertia: bool,
    run_new: Set[str],
) -> None:
    if run_method:
        try:
            method(run_new=run_new)
        except Exception:
            traceback.print_exc()
    if run_lj:
        try:
            lennard_jones(run_new=run_new)
        except Exception:
            traceback.print_exc()
    if run_ka:
        try:
            kob_andersen(run_new=run_new)
        except Exception:
            traceback.print_exc()
    if run_no_inertia:
        try:
            no_inertia(run_new=run_new)
        except Exception:
            traceback.print_exc()


DATA_ROOT_FOLDER = "paper-output"
FIG_ROOT_FOLDER = "paper-figs"
if not os.path.exists(FIG_ROOT_FOLDER):
    os.makedirs(FIG_ROOT_FOLDER)
FIG_DT_OVER_STEPS = os.path.join(FIG_ROOT_FOLDER, "dt_over_steps.svg")
FIG_RAW_DT_OVER_STEPS = os.path.join(FIG_ROOT_FOLDER, "dt_correction.svg")
FIG_DT_HIST = os.path.join(FIG_ROOT_FOLDER, "dt_hist.svg")
FIG_PARABOLAS = os.path.join(FIG_ROOT_FOLDER, "parabolas.svg")
FIG_PARABOLAS_RELATIVE_ERROR = os.path.join(FIG_ROOT_FOLDER, "parabolas_relative_error.svg")
FIG_LJ_RDF = os.path.join(FIG_ROOT_FOLDER, "lennard_jones_rdf.svg")
FIG_LJ_MSD = os.path.join(FIG_ROOT_FOLDER, "lennard_jones_msd.svg")
FIG_KA_RDF = os.path.join(FIG_ROOT_FOLDER, "kob_andersen_rdf.svg")
FIG_KA_MSD = os.path.join(FIG_ROOT_FOLDER, "kob_andersen_msd.svg")
FIG_NI_RDF = os.path.join(FIG_ROOT_FOLDER, "no_inertia_rdf.svg")
FIG_NI_MSD = os.path.join(FIG_ROOT_FOLDER, "no_inertia_msd.svg")


LJ_N0 = SimulationVsNVT(
    name="LJ_N0",
    description=
"""Single-component Lennard-Jones""",
    root_folder=DATA_ROOT_FOLDER,
    rho=0.85,
    steps=2**19,
    steps_per_timeblock=2**13,
    scalar_output=2**6,
    temperature=2.32,
    tau=0.2,
    dt=0.005,
    cells=[4, 4, 4],

    pair_potential_name="LJ",
    pair_potential_params={ "eps": 1, "sig": 1, "cut": 2.5},
    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-6,
    nvu_params_eps=1e-7,
    nvu_params_max_steps=20,
    nvu_params_max_initial_step_corrections=20,
    nvu_params_initial_step=1,
    nvu_params_initial_step_if_high=1,
    nvu_params_step=1,
    nvu_params_mode="reflection",
    nvu_params_save_path_u=True,
    nvu_params_root_method="parabola",
)

LJ_N1 = dataclasses.replace(
    LJ_N0,
    name="LJ_N1",
    cells=[8, 8, 8],
)

LJ_N2 = dataclasses.replace(
    LJ_N0,
    name="LJ_N2",
    cells=[8, 16, 16],
)


LJ_A = SimulationVsNVT(
    name="LJ_A",
    description=
"""Single-component Lennard-Jones.
Figure 5 a of paper I""",
    root_folder=DATA_ROOT_FOLDER,
    rho=0.85,
    temperature=2.32,
    steps=2**22,
    steps_per_timeblock=2**17,
    scalar_output=2**10,
    tau=0.2,
    dt=0.005,
    cells=[8, 8, 8],

    pair_potential_name="LJ",
    pair_potential_params={ "eps": 1, "sig": 1, "cut": 2.5},
    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-6,
    nvu_params_eps=1e-7,
    nvu_params_max_steps=20,
    nvu_params_max_initial_step_corrections=20,
    nvu_params_initial_step=1,
    nvu_params_initial_step_if_high=1,
    nvu_params_step=1,
    nvu_params_mode="reflection",
    nvu_params_save_path_u=True,
    nvu_params_root_method="parabola",
)

LJ_B = dataclasses.replace(
    LJ_A,
    name="LJ_B",
    rho=.427,
    temperature=1.10,
    nvu_params_initial_step=10,
)

LJ_C = dataclasses.replace(
    LJ_A,
    name="LJ_C",
    rho=.85,
    temperature=.28,
)

kob_andersen_parameters: Dict[str, Union[npt.NDArray[np.float32], float]] = { 
    "eps": np.array([[1.00, 1.50],
                     [1.50, 0.50]], dtype=np.float32), 
    "sig": np.array([[1.00, 0.80],
                     [0.80, 0.88]], dtype=np.float32), 
}
kob_andersen_parameters["cut"] = 2.5 * kob_andersen_parameters["sig"]

KA0 = SimulationVsNVT(
    name="KA0",
    description=
"""KABLJ
Figure 2 a and b of paper II""",
    root_folder=DATA_ROOT_FOLDER,
    rho=1.2,
    temperature=2,
    steps=2**25,
    steps_per_timeblock=2**20,
    scalar_output=2**13,
    tau=0.2,
    dt=0.005,
    cells=[8, 8, 8],

    pair_potential_name="Kob-Andersen",
    pair_potential_params=kob_andersen_parameters,
    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-6,
    nvu_params_eps=1e-7,
    nvu_params_max_steps=20,
    nvu_params_max_initial_step_corrections=20,
    nvu_params_initial_step=1,
    nvu_params_initial_step_if_high=1,
    nvu_params_step=1,
    nvu_params_mode="reflection",
    nvu_params_save_path_u=True,
    nvu_params_root_method="parabola",
)

KA_PARAMS = [KA0]
KA_TEMPERATURES = [2.0, 0.80, 0.60, 0.50, 0.44, 0.42, 0.405]
for i in range(1, len(KA_TEMPERATURES)):
    p = dataclasses.replace(
        KA0,
        name=f"KA{i}",
        temperature=KA_TEMPERATURES[i],
        temperature_eq=rp.make_function_ramp(
            value0=KA_TEMPERATURES[i-1],
            x0=KA0.steps * KA0.dt * (1 / 8),
            value1=KA_TEMPERATURES[i],
            x1=KA0.steps * KA0.dt * (1 / 4)),
        initial_conf=KA_PARAMS[i-1].nvt_conf_output,
    )
    KA_PARAMS.append(p)

# for i, k in enumerate(KA_PARAMS):
#     xs = np.arange(0, KA0.steps, KA0.steps_per_timeblock//20) * KA0.dt
#     print(k.name, k.temperature, k.initial_conf)
#     if i > 0:
#         plt.figure()
#         plt.plot(xs, np.vectorize(k.temperature)(xs))
#         plt.show()
# exit(0)

NI = SimulationVsNVT(
    name="NI",
    description=
"""Single-component Lennard-Jones.
Try to prove that time-reversibility is essential for the integrator""",
    root_folder=DATA_ROOT_FOLDER,
    rho=0.85,
    temperature=2.32,
    steps=2**15,
    steps_per_timeblock=2**10,
    scalar_output=2**4,
    tau=0.2,
    dt=0.005,
    cells=[8, 8, 8],

    pair_potential_name="LJ",
    pair_potential_params={ "eps": 1, "sig": 1, "cut": 2.5},
    nvu_params_mode="no-inertia",
    nvu_params_root_method="bisection",
    nvu_params_max_steps=1000,
    nvu_params_max_initial_step_corrections=20,
    nvu_params_initial_step=.1,
    nvu_params_initial_step_if_high=.1,
    nvu_params_step=.1,
    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-6,
    nvu_params_eps=1e-7,
)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r", "--run", help="Not used saved results", nargs='*', default=[])
    parser.add_argument("-m", "--method", help="Run Method", action="store_true")
    parser.add_argument("-l", "--lj", help="Run Lennard-Jones", action="store_true")
    parser.add_argument("-k", "--ka", help="Run Kob-Andersen", action="store_true")
    parser.add_argument("-n", "--no_inertia", help="Run No inertia", action="store_true")
    parser.add_argument("-d", "--device", help="Select NVIDIA device", type=int, default=None)
    if 'PBS_O_WORKDIR' in os.environ:
        flags = os.environ.get("flags", "")
        raw_args = flags.split()
    else:
        raw_args = None
    args = parser.parse_args(raw_args)

    if args.device is not None:
        print("Using device:", args.device)
        cuda.select_device(args.device)

    for name in args.run:
        if name not in KNOWN_SIMULATIONS:
            raise ValueError(f"Expected one of `{', '.join(KNOWN_SIMULATIONS.keys())}` for the run argument, but got `{name}`")

    main(
        run_method=args.method, 
        run_lj=args.lj, 
        run_ka=args.ka, 
        run_no_inertia=args.no_inertia,
        run_new=set(args.run),
    )

