#!/usr/bin/python3
#PBS -v PYTHONPATH
#PBS -l nodes=bead53
#PBS -o out/$PBS_JOBNAME.out
#PBS -j oe

import math
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
from tools import KNOWN_SIMULATIONS, SimulationParameters, SimulationVsNVE, SimulationVsNVT, calculate_rdf, \
    load_conf_from_npz
import numpy as np
import numpy.typing as npt


plt.style.use("./science.mplstyle")


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


def scientific_notation(v: Union[float, np.float32], n: int) -> str:
    s = f"{{:.0{n}e}}".format(v)
    s = s.replace("e", r"\cdot 10^{")
    s = s + "}"
    return s


def get_output(params: SimulationParameters, rdf_new: bool) -> Output:
    if params.name in USE_BACKUP:
        params = USE_BACKUP[params.name]
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

    print(f"Loading {kind} production output from path `{other_output_path}`")
    other_prod_output = rp.tools.load_output(other_output_path)
    print(f"Loading NVU EQ initial configuration from path `{eq_conf0_path}`")
    eq_conf0, target_u = load_conf_from_npz(eq_conf0_path)

    print(f"Loading NVU EQ output from path `{params.nvu_eq_output}`")
    nvu_eq_output = rp.tools.load_output(params.nvu_eq_output)
    print(f"Loading NVU PROD initial configuration from path `{params.nvu_eq_conf_output}`")
    prod_conf0, _target_u = load_conf_from_npz(params.nvu_eq_conf_output)
    print(f"Loading NVU PROD output from path `{params.nvu_output}`")
    nvu_prod_output = rp.tools.load_output(params.nvu_output)

    conf_per_block = math.floor(512 / nvu_prod_output["block"].shape[0])
    if rdf_new or not os.path.exists(params.other_prod_rdf):
        other_prod_rdf = calculate_rdf(other_prod_output, conf_per_block)
        np.savez(params.other_prod_rdf, **other_prod_rdf)
    else:
        other_prod_rdf = np.load(params.other_prod_rdf)

    if rdf_new or not os.path.exists(params.nvu_prod_rdf):
        prod_rdf = calculate_rdf(nvu_prod_output, conf_per_block)
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


def get_delta_time_from_msd(msd: FloatArray, temperature: float, mass: float = 1) -> FloatArray:
    kb = 1
    beta = np.sqrt(mass * msd[0] / (3 * kb * temperature))
    return beta


def get_delta_time(output: Dict[str, Any]) -> FloatArray:
    dt0, cos_v_f, = rp.extract_scalars(output, ["dt", "cos_v_f", ], 
        integrator_outputs=rp.integrators.NVU_RT.outputs)
    cos_v_f[cos_v_f > 1] = 1
    cos_v_f[cos_v_f < -1] = -1
    dt = dt0 * (np.pi / 2 - np.arccos(cos_v_f)) / cos_v_f
    return dt


def get_steps(output: Dict[str, Any]) -> FloatArray:
    nblocks, nscalar_per_block, _nscalars = output["scalars"].shape
    steps = np.arange(nblocks * nscalar_per_block) * output.attrs["steps_between_output"]
    return steps


def get_path_u(output: Dict[str, Any]) -> Tuple[FloatArray, FloatArray]:
    nblocks, npaths_per_block, npoints, n = output["path_u"].shape
    data = output["path_u"][:].reshape(nblocks * npaths_per_block, npoints, n)
    xs = data[:, :, 0].T
    ys = data[:, :, 1].T
    return xs, ys


def get_msd(output: Dict[str, Any]) -> FloatArray:
    return rp.tools.calc_dynamics(output, first_block=0)["msd"]


def method(rdf_new: Set[str], rdf_all: bool) -> None:
    ## DISTRIBUTION OF DELTA TIMES
    ##   - Plot \Delta t over steps
    output_n0 = get_output(LJ_N0, rdf_new=LJ_N0.name in rdf_new or rdf_all)
    n0 = output_n0.prod_output["block"].shape[3]
    output_n1 = get_output(LJ_N1, rdf_new=LJ_N1.name in rdf_new or rdf_all)
    n1 = output_n1.prod_output["block"].shape[3]
    output_n2 = get_output(LJ_N2, rdf_new=LJ_N2.name in rdf_new or rdf_all)
    n2 = output_n2.prod_output["block"].shape[3]
    # n0_dt = get_delta_time(output_n0.prod_output)
    # n0_steps = get_steps(output_n0.prod_output)
    # n1_dt = get_delta_time(output_n1.prod_output)
    # n1_steps = get_steps(output_n1.prod_output)
    # n2_dt = get_delta_time(output_n2.prod_output)
    # n2_steps = get_steps(output_n2.prod_output)
    #
    # n0_dt_raw, = rp.extract_scalars(output_n0.prod_output, ["dt"], integrator_outputs=rp.integrators.NVU_RT.outputs)
    # n1_dt_raw, = rp.extract_scalars(output_n1.prod_output, ["dt"], integrator_outputs=rp.integrators.NVU_RT.outputs)
    # n2_dt_raw, = rp.extract_scalars(output_n2.prod_output, ["dt"], integrator_outputs=rp.integrators.NVU_RT.outputs)
    #
    # fig = plt.figure(figsize=(10, 8))
    # # fig.suptitle(r"Correction to $\Delta t$ to account for curvature fo the surface")
    # gs = GridSpec(3, 2, width_ratios=[1, 0.5], hspace=.6, wspace=0)
    # fig.text(1.0, 0.5, r"$\frac{\alpha}{\sin\alpha} - 1 = \frac{\Delta t}{\Delta t_{not\,corrected}} - 1$", 
    #          va='center', rotation=90, fontsize="x-large")
    # for (i, steps, dt, dt_raw, n) in zip(
    #     range(3), 
    #     (n0_steps, n1_steps, n2_steps), 
    #     (n0_dt, n1_dt, n2_dt),
    #     (n0_dt_raw, n1_dt_raw, n2_dt_raw),
    #     (n0, n1, n2)
    # ):
    #     row = fig.add_subplot(gs[i, :])
    #     row.axis('off')
    #     row.set_title(rf"$N = {n}$")
    #
    #     ax0 = fig.add_subplot(gs[i, 0])
    #     ax0.plot(steps, dt/dt_raw - 1, color="black", linewidth=.5, alpha=0.8)
    #     ax0.set_xlabel(r"steps")
    #     # ax0.set_ylabel(r"$\frac{\alpha}{\sin\alpha} - 1 = \frac{\Delta t}{\Delta t_{not\,corrected}} - 1$")
    #     ax0.grid(alpha=0.5)
    #
    #     ax1 = fig.add_subplot(gs[i, 1])
    #     ax1.hist(dt/dt_raw - 1, orientation="horizontal", density=True, color="black", alpha=0.8)
    #     ax1.set_xlabel(r"Probability")
    #     ax1.yaxis.set_label_position("right")
    #     ax1.yaxis.tick_right()
    #     ax1.grid(alpha=0.5)
    #
    # fig.savefig(FIG_DT_CORRECTION)
    # plt.close(fig)
    #
    # fig = plt.figure(figsize=(8, 3))
    # ax = fig.add_subplot()
    # ax.plot(n0_steps, n0_dt, linewidth=1, color="#BBB", alpha=0.8, label=rf"$N = {n0}$")
    # ax.plot(n1_steps, n1_dt, linewidth=1, color="#555", alpha=0.8, label=rf"$N = {n1}$")
    # ax.plot(n2_steps, n2_dt, linewidth=1, color="#000", alpha=0.8, label=rf"$N = {n2}$")
    # ax.legend()
    # ax.set_xlabel(r"steps")
    # ax.set_ylabel(r"$\Delta t$")
    # ax.grid(alpha=.5)
    # fig.savefig(FIG_DT_OVER_STEPS)
    # plt.close(fig)
    #
    # fig = plt.figure(figsize=(8, 3))
    # ax = fig.add_subplot()
    # d_time_sq0 = n0_dt**2 - (n0_dt**2).mean()
    # d_time_sq1 = n1_dt**2 - (n1_dt**2).mean()
    # d_time_sq2 = n2_dt**2 - (n2_dt**2).mean()
    # s_0 = np.std(d_time_sq0)
    # s_1 = np.std(d_time_sq1)
    # s_2 = np.std(d_time_sq2)
    # 
    # ax.hist(d_time_sq0, bins=30, density=True, facecolor="#eb3434", alpha=0.8, label=rf"$N = {n0}$; $\sigma = {scientific_notation(s_0, 2)}$")
    # ax.hist(d_time_sq1, bins=30, density=True, facecolor="#34b1eb", alpha=0.8, label=rf"$N = {n1}$; $\sigma = {scientific_notation(s_1, 2)}$")
    # ax.hist(d_time_sq2, bins=30, density=True, facecolor="black", alpha=0.6, label=rf"$N = {n2}$; $\sigma = {scientific_notation(s_2, 2)}$")
    # ax.set_xlim(-1e-4, 1e-4)
    # ax.set_xlabel(r"$\left(\Delta t\right)^2 - \langle \left(\Delta t\right)^2\rangle$")
    # ax.set_ylabel(r"Probability")
    # ax.grid(alpha=0.5)
    # ax.legend()
    # fig.savefig(FIG_DT_HIST)
    # plt.close(fig)
    #
    # ## PARABOLA IN PATHS
    # ##   - Plot Data + Fit
    # ##   - Plot Error
    #
    # xs, ys = get_path_u(output_n1.prod_output)
    # 
    # u = np.arange(xs.shape[0])
    # p = np.polyfit(u, ys, 2)
    # y_pred = p[0, :] * u[:, np.newaxis]**2 + p[1, :] * u[:, np.newaxis] + p[2, :]
    # fig = plt.figure(figsize=(8, 3))
    # gs = GridSpec(1, 2, width_ratios=[1, .5], wspace=0)
    # ax0 = fig.add_subplot(gs[0])
    # error = (y_pred - ys) / ys
    # n = error.flatten().shape[0] // 6000
    # ax0.plot(error.T.flatten()[::n], ".", color="black", alpha=0.2)
    # ax0.set_xlabel("sample points")
    # ax0.set_ylabel(r"$\frac{y_{pred} - y_i}{y_i}$", rotation=0, labelpad=20)
    # ax0.grid(alpha=0.5)
    # ax1 = fig.add_subplot(gs[1], sharey=ax0)
    # ax1.hist(error.flatten(), orientation="horizontal", density=True, bins=30, color="black", alpha=0.8)
    # ax1.set_xlabel(r"Probability")
    # ax1.set_ylabel(r"$\frac{y_{pred} - y_i}{y_i}$", rotation=0, labelpad=20)
    # ax1.yaxis.set_label_position("right")
    # ax1.yaxis.tick_right()
    # ax1.grid(alpha=0.5)
    # fig.savefig(FIG_PARABOLAS_RELATIVE_ERROR)
    # plt.close(fig)
    #
    # a = p[0, :]
    # b = p[1, :]
    # fig = plt.figure(figsize=(8, 3))
    # gs = GridSpec(1, 2, wspace=.3)
    # ax0 = fig.add_subplot(gs[0])
    # ax0.hist(a, density=True, bins=30, color="black", alpha=0.8)
    # ax0.set_xlabel("$a$")
    # ax0.set_ylabel("$p(a)$")
    # mu = np.mean(a)
    # sig = np.std(a)
    # t = ax0.annotate(
    #     rf"$\mu(a) = {mu:.02f}$""\n" 
    #     rf"$\sigma(a) = {sig:.03f}$",
    #     (0.55, 0.90), xycoords="axes fraction", verticalalignment="top")
    # t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))
    # ax1 = fig.add_subplot(gs[1])
    # ax1.hist(b, density=True, bins=30, color="black", alpha=0.8)
    # ax1.set_xlabel("$b$")
    # ax1.set_ylabel("$p(b)$")
    # mu = np.mean(b)
    # sig = np.std(b)
    # t = ax1.annotate(
    #     rf"$\mu(b) = {mu:.02f}$""\n" 
    #     rf"$\sigma(b) = {sig:.03f}$",
    #     (0.05, 0.90), xycoords="axes fraction", verticalalignment="top")
    # t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))
    # fig.savefig(FIG_PARABOLAS_AB)
    # plt.close(fig)
    #
    #
    # indices = np.argsort(np.abs(ys).max(axis=0))[::-1][[0, ys.shape[1]//4, ys.shape[1]*2//3, -1]]
    # # print(indices, np.abs(ys).max(axis=0)[indices])
    # fig = plt.figure(figsize=(8, 3))
    # gs = GridSpec(1, 2, wspace=0)
    # ax1 = fig.add_subplot(gs[0])
    # ax2 = fig.add_subplot(gs[1])
    # for i, (ax, xs) in enumerate(((ax1, xs/xs[[-1], :]), (ax2, xs))):
    #     for j, path_j in enumerate(indices):
    #         x_j = xs[:, path_j]
    #         y_j = ys[:, path_j]
    #         c = np.polyfit(x_j, y_j, 2)
    #         p = np.poly1d(c)
    #         xs_j = np.linspace(x_j[0], x_j[-1])
    #         line0, = ax.plot(xs_j, p(xs_j), '--', color="black", linewidth=1)
    #         line1, = ax.plot(x_j, y_j, ".", linewidth=0, alpha=.8,
    #                 color="red", markeredgewidth=1,
    #                 markersize=10, markeredgecolor="black")
    #         if j == 0:
    #             line0.set_label("Fit")
    #             line1.set_label("Data")
    #     if i == 0:
    #         ax.legend()
    #         ax.set_xlabel(r"$\frac{x}{x_{max}}$")
    #         ax.set_ylabel(r"$U$")
    #     else:
    #         ax.set_xlabel(r"$x$")
    #         ax.yaxis.set_label_position("right")
    #         ax.yaxis.tick_right()
    #     ax.grid(alpha=0.5)
    # fig.savefig(FIG_PARABOLAS)
    # plt.close(fig)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()
    for (i, params, output, n) in zip(
        range(3), 
        (LJ_N0, LJ_N1, LJ_N2),
        (output_n0, output_n1, output_n2),
        (n0, n1, n2)
    ):
        msd = get_msd(output.prod_output)[:, 0]
        dt = get_delta_time_from_msd(msd, params.temperature)
        time = dt * 2 ** np.arange(len(msd))

        other_msd = get_msd(output.other_prod_output)[:, 0]
        other_time = params.dt * 2 ** np.arange(len(other_msd))

        ln, = ax.loglog(other_time, other_msd, linewidth=1, color="black")
        if i == 0:
            ln.set_label("NVT")
        ax.loglog(time, msd, marker='.', linewidth=0, 
                  markeredgewidth=1, markersize=9, markeredgecolor="black",
                  alpha=.9, label=f"NVU RT $N={n}$")
    t = ax.annotate(
        rf"temperature = ${LJ_N0.temperature}$""\n"
        rf"$\rho$ = ${LJ_N0.rho}$", 
        (0.5, 0.15), xycoords="axes fraction")
    t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))
    ax.set_ylabel(r"$\langle \Delta r^2 \rangle$")
    ax.set_xlabel("$t$")
    ax.legend()
    ax.grid(alpha=.3)

    fig.savefig(FIG_MSD_N)
    plt.close(fig)


def lennard_jones(rdf_new: Set[str], rdf_all: bool) -> None:
    output_a = get_output(LJ_A, rdf_new=LJ_A.name in rdf_new or rdf_all)
    output_b = get_output(LJ_B, rdf_new=LJ_B.name in rdf_new or rdf_all)
    output_c = get_output(LJ_C, rdf_new=LJ_C.name in rdf_new or rdf_all)

    fsq, lap = rp.extract_scalars(output_a.prod_output, ["Fsq", "lapU"])
    steps = get_steps(output_a.prod_output)
    t_conf = fsq / lap
    sig = np.std(t_conf)
    mu = np.mean(t_conf)
    n = len(t_conf) // 1000
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot()
    ax.plot(steps[::n], t_conf[::n], linewidth=1, alpha=.8, color="black")
    t = ax.annotate(
        rf"$\mu(T_{{conf}}) = {mu:.02f}$""\n" 
        rf"$\sigma(T_{{conf}}) = {sig:.03f}$",
        (0.1, 0.90), xycoords="axes fraction", verticalalignment="top")
    t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))
    t = ax.annotate(
        rf"temperature = ${LJ_A.temperature:.02f}$""\n"
        rf"$\rho$ = ${LJ_A.rho}$", 
        (0.3, 0.90), xycoords="axes fraction", verticalalignment="top")
    t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))
    ax.grid(alpha=.3)
    ax.set_ylim(2, 2.8)
    ax.set_xlabel("steps")
    ax.set_ylabel(r"$T_{conf} = \dfrac{|\nabla U|^2}{\nabla^2 U}$")
    fig.savefig(FIG_LJ_TCONF)

    fig = plt.figure(figsize=(10, 7))
    gs = GridSpec(2, 4, wspace=.7, hspace=.2)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[1, 1:3])
    for i, (ax, output, params) in enumerate((
        (ax_a, output_a, LJ_A),
        (ax_b, output_b, LJ_B),
        (ax_c, output_c, LJ_C)
    )):
        n = output.prod_rdf["distances"].shape[0] // 200
        rdf = np.mean(output.prod_rdf["rdf"], axis=0)[::n]
        distances = output.prod_rdf["distances"][::n]
        other_rdf = np.mean(output.other_prod_rdf["rdf"], axis=0)
        ax.plot(output.other_prod_rdf["distances"], other_rdf, linewidth=1, color="black", label="NVT")
        ax.plot(distances, rdf, marker='.', linewidth=0, 
            markeredgewidth=.8, markersize=6, markeredgecolor="black",
            color="red", alpha=.9, label="NVU RT")
        t = ax.annotate(
            rf"temperature = ${params.temperature}$""\n"
            rf"$\rho$ = ${params.rho}$", 
            (0.4, 0.55), xycoords="axes fraction")
        t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))
        ax.grid(alpha=.3)
        ax.legend()
        ax.set_xlabel("$r$")
        ax.set_ylabel("$g(r)$")
        if i < 2:
            ax.set_xlim(0, 4)
    fig.savefig(FIG_LJ_RDF)

    fig = plt.figure(figsize=(10, 7))
    gs = GridSpec(2, 4, wspace=.7, hspace=.2)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4], sharex=ax_a, sharey=ax_a)
    ax_c = fig.add_subplot(gs[1, 1:3], sharex=ax_a, sharey=ax_a)
    for ax, output, params in (
        (ax_a, output_a, LJ_A),
        (ax_b, output_b, LJ_B),
        (ax_c, output_c, LJ_C)
    ):
        msd = get_msd(output.prod_output)[:, 0]
        dt = get_delta_time_from_msd(msd, params.temperature)
        time = dt * 2 ** np.arange(len(msd))

        other_msd = get_msd(output.other_prod_output)[:, 0]
        other_time = params.dt * 2 ** np.arange(len(other_msd))

        ax.loglog(other_time, other_msd, linewidth=1, color="black", label="NVT")
        ax.loglog(time, msd, marker='.', linewidth=0, 
                  markeredgewidth=1, markersize=9, markeredgecolor="black",
                  color="red", alpha=.9, label="NVU RT")
        t = ax.annotate(
            rf"temperature = ${params.temperature}$""\n"
            rf"$\rho$ = ${params.rho}$", 
            (0.5, 0.15), xycoords="axes fraction")
        t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))
        ax.set_ylabel(r"$\langle \Delta r^2 \rangle$")
        ax.set_xlabel("$t$")
        ax.legend()
        ax.grid(alpha=.3)
    fig.savefig(FIG_LJ_MSD)


def kob_andersen(rdf_new: Set[str], rdf_all: bool) -> None:
    outputs = [
        get_output(params, rdf_new=params.name in rdf_new or rdf_all)
        for params in KA_PARAMS
    ]

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 1, wspace=.5, hspace=.2)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    for ax, output, params in (
        (ax_a, outputs[0], KA_PARAMS[0]),
        (ax_b, outputs[-1], KA_PARAMS[-1]),
    ):
        n = len(output.prod_rdf["distances"]) // 180
        distances = output.prod_rdf["distances"][::n]
        other_rdf = np.mean(output.other_prod_rdf["rdf"], axis=0)
        rdf = np.mean(output.prod_rdf["rdf"], axis=0)[::n]
        ax.plot(output.other_prod_rdf["distances"], other_rdf, linewidth=1, color="black", label="NVT")
        for i in range(output.prod_rdf["rdf_ptype"].shape[1]):
            for j in range(output.prod_rdf["rdf_ptype"].shape[2]):
                if j > i:
                    break
                other_rdf_ij = np.mean(output.other_prod_rdf["rdf_ptype"][:, i, j, :], axis=0)
                ax.plot(output.other_prod_rdf["distances"], other_rdf_ij, linewidth=1, color="black", alpha=.8)
        ax.plot(distances, rdf, marker='.', linewidth=0, 
                markeredgewidth=1, markersize=9, markeredgecolor="black",
                color="red", alpha=.9, label="NVU RT")
        for i in range(output.prod_rdf["rdf_ptype"].shape[1]):
            for j in range(output.prod_rdf["rdf_ptype"].shape[2]):
                if j > i:
                    break
                rdf_ij = np.mean(output.prod_rdf["rdf_ptype"][:, i, j, :], axis=0)[::n]
                ax.plot(distances, rdf_ij, marker='.', linewidth=0, 
                        markeredgewidth=1, markersize=9, markeredgecolor="black",
                        alpha=.8, label=f"NVU RT {['A', 'B'][j]}-{['A', 'B'][i]}")
        t = ax.annotate(
            rf"temperature = ${params.temperature}$",
            (0.4, 0.62), xycoords="axes fraction")
        t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))
        ax.grid(alpha=.3)
        ax.legend()
        ax.set_xlabel("$r$")
        ax.set_ylabel("$g(r)$")
        ax.set_xlim(0, 4)
    fig.savefig(FIG_KA_RDF)

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 1, wspace=.2, hspace=.2)
    axs = [fig.add_subplot(gs[i]) for i in range(2)]
    for i, (output, params) in enumerate(zip(outputs, KA_PARAMS)):
        msd = get_msd(output.prod_output)
        # ax = fig.add_subplot(gs[i])
        n_msd, n_ptype = msd.shape
        other_msd = get_msd(output.other_prod_output)
        other_time = params.dt * 2 ** np.arange(other_msd.shape[0])

        for ax, j in zip(axs, range(n_ptype)):
            msd_j = msd[:, j]
            other_msd_j = other_msd[:, j]
            dt = get_delta_time_from_msd(msd_j, params.temperature)
            time = dt * 2 ** np.arange(n_msd)

            line0, = ax.loglog(other_time, other_msd_j, linewidth=1, color="black")
            line1, = ax.loglog(time, msd_j, marker='.', linewidth=0, 
                               markeredgewidth=1, markersize=9, markeredgecolor="black",
                               color="red" if j == 0 else "green", alpha=.9)
            if i == 0:
                line0.set_label("NVT")
                line1.set_label("NVU RT")
                t = ax.annotate(rf"temperature = ${params.temperature:.03f}$", (0.05, 0.8), xycoords="axes fraction")
                t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))

                t = ax.annotate(rf"{'A' if j == 0 else 'B'}", (0.4, 0.2), xycoords="axes fraction",
                                color="red" if j == 0 else "green")
                t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))
            if i == len(outputs) - 1:
                t = ax.annotate(rf"temperature = ${params.temperature:.03f}$", (0.6, 0.32), xycoords="axes fraction")
                t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))

            if i == 0:
                ax.grid(alpha=.3)
                ax.legend(loc="lower right")
                ax.set_ylabel(r"$\langle \Delta r^2 \rangle$")
                ax.set_xlabel("$t$")
                ax.set_ylim(1e-4, 1e2)
    fig.savefig(FIG_KA_MSD)


def asd(rdf_new: Set[str], rdf_all: bool) -> None:
    output = get_output(ASD, rdf_new=ASD.name in rdf_new or rdf_all)
    output_no_scale = get_output(ASD_NO_SCALE, rdf_new=ASD.name in rdf_new or rdf_all)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    n = len(output.prod_rdf["distances"]) // 110
    distances = output.prod_rdf["distances"][::n]
    other_rdf = np.mean(output.other_prod_rdf["rdf"], axis=0)
    rdf = np.mean(output.prod_rdf["rdf"], axis=0)[::n]
    ax.plot(output.other_prod_rdf["distances"], other_rdf, linewidth=1, color="black", label="NVT")
    for i in range(output.prod_rdf["rdf_ptype"].shape[1]):
        for j in range(output.prod_rdf["rdf_ptype"].shape[2]):
            if j != i:
                continue
            if j > i:
                break
            other_rdf_ij = np.mean(output.other_prod_rdf["rdf_ptype"][:, i, j, :], axis=0)
            ax.plot(output.other_prod_rdf["distances"], other_rdf_ij, linewidth=1, color="black", alpha=.8)
    ax.plot(distances, rdf, marker='.', linewidth=0, 
            markeredgewidth=1, markersize=9, markeredgecolor="black",
            color="red", alpha=.9, label="NVU RT")
    for i in range(output.prod_rdf["rdf_ptype"].shape[1]):
        for j in range(output.prod_rdf["rdf_ptype"].shape[2]):
            if j != i:
                continue
            if j > i:
                break
            rdf_ij = np.mean(output.prod_rdf["rdf_ptype"][:, i, j, :], axis=0)[::n]
            ax.plot(distances, rdf_ij, marker='.', linewidth=0, 
                    markeredgewidth=1, markersize=9, markeredgecolor="black",
                    alpha=.8, label=f"NVU RT {['A', 'B'][j]}-{['A', 'B'][i]}")
    # t = ax.annotate(
    #     rf"temperature = ${ASD.temperature}$",
    #     (0.4, 0.62), xycoords="axes fraction")
    # t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))
    ax.grid(alpha=.3)
    ax.legend()
    ax.set_xlabel("$r$")
    ax.set_ylabel("$g(r)$")
    fig.savefig(FIG_ASD_RDF)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    msd = get_msd(output_no_scale.prod_output)
    other_msd = get_msd(output_no_scale.other_prod_output)
    other_time = ASD.dt * 2 ** np.arange(len(other_msd))
    n_msd, n_type = msd.shape

    for i in range(n_type):
        mass: float = 1
        if ASD.pair_potential_name == "ASD" and i == 1:
            mass = ASD.pair_potential_params["b_mass"]  # type: ignore
        dt = get_delta_time_from_msd(msd[:, i], ASD.temperature, mass)
        time = np.mean(dt) * 2 ** np.arange(n_msd)

        color = "red" if i == 0 else "blue"

        ax.loglog(other_time, other_msd[:, i], linewidth=1, color="black", label="NVT")
        ax.loglog(time, msd[:, i], marker='.', linewidth=0, markeredgewidth=1, markersize=9, markeredgecolor="black",
                  color=color, alpha=.9, label=f"NVU RT {['A', 'B'][i]}")
    ax.grid(alpha=.3)
    ax.legend()
    ax.set_ylabel(r"$\langle \Delta r^2 \rangle$")
    ax.set_xlabel("$t$")
    fig.savefig(FIG_ASD_MSD_NO_SCALE)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    msd = get_msd(output.prod_output)
    other_msd = get_msd(output.other_prod_output)
    other_time = ASD.dt * 2 ** np.arange(len(other_msd))
    n_msd, n_type = msd.shape

    for i in range(n_type):
        mass: float = 1
        if ASD.pair_potential_name == "ASD" and i == 1:
            mass = ASD.pair_potential_params["b_mass"]  # type: ignore
        dt = get_delta_time_from_msd(msd[:, i], ASD.temperature, mass)
        time = np.mean(dt) * 2 ** np.arange(n_msd)

        color = "red" if i == 0 else "blue"

        ax.loglog(other_time, other_msd[:, i], linewidth=1, color="black", label="NVT")
        ax.loglog(time, msd[:, i], marker='.', linewidth=0, markeredgewidth=1, markersize=9, markeredgecolor="black",
                  color=color, alpha=.9, label=f"NVU RT {['A', 'B'][i]}")
    ax.grid(alpha=.3)
    ax.legend()
    ax.set_ylabel(r"$\langle \Delta r^2 \rangle$")
    ax.set_xlabel("$t$")
    fig.savefig(FIG_ASD_MSD)


def no_inertia(rdf_new: Set[str], rdf_all: bool) -> None:
    output = get_output(NI, rdf_new=NI.name in rdf_new or rdf_all)

    fig = plt.figure(figsize=(8, 3))
    gs = GridSpec(1, 2, wspace=0)

    ax = fig.add_subplot(gs[0])
    n = np.concatenate([np.arange(0, int(output.prod_rdf["distances"].shape[0]/4.5), 5),
                        np.arange(int(output.prod_rdf["distances"].shape[0]/4.5), output.prod_rdf["distances"].shape[0], 20)])
    rdf = np.mean(output.prod_rdf["rdf"], axis=0)[n]
    distances = output.prod_rdf["distances"][n]
    other_rdf = np.mean(output.other_prod_rdf["rdf"], axis=0)
    ax.plot(output.other_prod_rdf["distances"], other_rdf, linewidth=1, color="black", label="NVT")
    ax.plot(distances, rdf, marker='.', linewidth=0, 
        markeredgewidth=1, markersize=9, markeredgecolor="black",
        color="red", alpha=.9, label="NVU RT (no inertia)")
    ax.grid(alpha=.3)
    ax.legend()
    ax.set_xlabel("$r$")
    ax.set_ylabel("$g(r)$")

    ax = fig.add_subplot(gs[1])
    msd = get_msd(output.prod_output)[:, 0]
    dt = get_delta_time_from_msd(msd, NI.temperature)
    time = dt * 2 ** np.arange(len(msd))

    other_msd = get_msd(output.other_prod_output)[:, 0]
    other_time = NI.dt * 2 ** np.arange(len(other_msd))

    ax.loglog(other_time, other_msd, linewidth=1, color="black", label="NVT")
    ax.loglog(time, msd, marker='.', linewidth=0, 
              markeredgewidth=1, markersize=9, markeredgecolor="black",
              color="red", alpha=.9, label="NVU RT (no inertia)")
    ax.grid(alpha=.3)
    # ax.legend()
    ax.set_ylabel(r"$\langle \Delta r^2 \rangle$")
    ax.set_xlabel("$t$")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    fig.savefig(FIG_NO_INERTIA)


def main(
    run_method: bool, 
    run_lj: bool, 
    run_ka: bool, 
    run_no_inertia: bool,
    run_asd: bool,
    rdf_new: Set[str],
    rdf_all: bool,
) -> None:
    if run_method:
        try:
            method(rdf_new=rdf_new, rdf_all=rdf_all)
        except Exception:
            traceback.print_exc()
    if run_lj:
        try:
            lennard_jones(rdf_new=rdf_new, rdf_all=rdf_all)
        except Exception:
            traceback.print_exc()
    if run_ka:
        try:
            kob_andersen(rdf_new=rdf_new, rdf_all=rdf_all)
        except Exception:
            traceback.print_exc()
    if run_no_inertia:
        try:
            no_inertia(rdf_new=rdf_new, rdf_all=rdf_all)
        except Exception:
            traceback.print_exc()
    if run_asd:
        try:
            asd(rdf_new=rdf_new, rdf_all=rdf_all)
        except Exception:
            traceback.print_exc()


DATA_ROOT_FOLDER = "paper-output"
FIG_ROOT_FOLDER = "paper-figs"
if not os.path.exists(FIG_ROOT_FOLDER):
    os.makedirs(FIG_ROOT_FOLDER)
FIG_DT_OVER_STEPS = os.path.join(FIG_ROOT_FOLDER, "dt_over_steps.svg")
FIG_DT_CORRECTION = os.path.join(FIG_ROOT_FOLDER, "dt_correction.svg")
FIG_DT_HIST = os.path.join(FIG_ROOT_FOLDER, "dt_hist.svg")
FIG_MSD_N = os.path.join(FIG_ROOT_FOLDER, "msd_n.svg")
FIG_PARABOLAS = os.path.join(FIG_ROOT_FOLDER, "parabolas.svg")
FIG_PARABOLAS_AB = os.path.join(FIG_ROOT_FOLDER, "parabolas_ab.svg")
FIG_PARABOLAS_RELATIVE_ERROR = os.path.join(FIG_ROOT_FOLDER, "parabolas_relative_error.svg")
FIG_LJ_RDF = os.path.join(FIG_ROOT_FOLDER, "lennard_jones_rdf.svg")
FIG_LJ_MSD = os.path.join(FIG_ROOT_FOLDER, "lennard_jones_msd.svg")
FIG_LJ_TCONF = os.path.join(FIG_ROOT_FOLDER, "lennard_jones_tconf.svg")
FIG_KA_RDF = os.path.join(FIG_ROOT_FOLDER, "kob_andersen_rdf.svg")
FIG_KA_MSD = os.path.join(FIG_ROOT_FOLDER, "kob_andersen_msd.svg")
FIG_ASD_RDF = os.path.join(FIG_ROOT_FOLDER, "asd_rdf.svg")
FIG_ASD_MSD = os.path.join(FIG_ROOT_FOLDER, "asd_msd.svg")
FIG_ASD_MSD_NO_SCALE = os.path.join(FIG_ROOT_FOLDER, "asd_msd_no_scale.svg")
FIG_NO_INERTIA = os.path.join(FIG_ROOT_FOLDER, "no_inertia.svg")


LJ_N0 = SimulationVsNVT(
    name="LJ_N0",
    description=
"""Single-component Lennard-Jones""",
    root_folder=DATA_ROOT_FOLDER,
    rho=0.85,
    steps=2**20,
    steps_per_timeblock=2**15,
    scalar_output=2**8,
    temperature=1,
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
    nvu_params_initial_step=0.5/0.85**(1/3),
    nvu_params_initial_step_if_high=1,
    nvu_params_step=1,
    nvu_params_mode="reflection",
    nvu_params_save_path_u=True,
    nvu_params_raytracing_method="parabola",
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
LJ_N3 = dataclasses.replace(
    LJ_N0,
    name="LJ_N3",
    cells=[16, 16, 16],
)
LJ_N4 = dataclasses.replace(
    LJ_N0,
    name="LJ_N4",
    cells=[16, 16, 32],
)
LJ_N5 = dataclasses.replace(
    LJ_N0,
    name="LJ_N5",
    cells=[16, 32, 32],
)
LJ_N6 = dataclasses.replace(
    LJ_N0,
    name="LJ_N6",
    cells=[32, 32, 32],
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
    nvu_params_initial_step=0.5/0.85**(1/3),
    nvu_params_initial_step_if_high=1,
    nvu_params_step=1,
    nvu_params_mode="reflection",
    nvu_params_save_path_u=True,
    nvu_params_raytracing_method="parabola",
)

LJ_B = dataclasses.replace(
    LJ_A,
    name="LJ_B",
    rho=.427,
    temperature=1.10,
    nvu_params_initial_step=0.5/.427**(1/3),
)

LJ_C = dataclasses.replace(
    LJ_A,
    name="LJ_C",
    rho=.85,
    temperature=.28,
    nvu_params_initial_step=0.5/.85**(1/3),
)

kob_andersen_parameters: Dict[str, Union[npt.NDArray[np.float32], float]] = { 
    "eps": np.array([[1.00, 1.50],
                     [1.50, 0.50]], dtype=np.float32), 
    "sig": np.array([[1.00, 0.80],
                     [0.80, 0.88]], dtype=np.float32), 
}

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
    nvu_params_initial_step=0.5/1.2**(1/3),
    nvu_params_initial_step_if_high=1,
    nvu_params_step=1,
    nvu_params_mode="reflection",
    # nvu_params_save_path_u=True,
    nvu_params_raytracing_method="parabola-newton",
)

KA_PARAMS = [KA0]
KA_TEMPERATURES = [2.0, 0.80, 0.60, 0.50, 0.44, 0.42, 0.405]
for i in range(1, len(KA_TEMPERATURES)):
    if i <= 3:
        order = 25
    elif i == 4:
        order = 28
    elif i == 5:
        order = 30
    else:
        order = 32
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
        steps=2**order,
        steps_per_timeblock=2**(order - 5),
        scalar_output=2**(order - 12),
    )
    KA_PARAMS.append(p)

dataclasses.replace(
    KA_PARAMS[6],
    name=f"KA6_short",
    steps=2**30,
    steps_per_timeblock=2**25,
    scalar_output=2**(30 - 12),
)

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
    nvu_params_raytracing_method="bisection",
    nvu_params_max_steps=1000,
    nvu_params_max_initial_step_corrections=20,
    nvu_params_initial_step=0.5/.85**(1/3),
    nvu_params_initial_step_if_high=.1,
    nvu_params_step=.1,
    nvu_params_max_abs_val=2,
    nvu_params_threshold=1e-6,
    nvu_params_eps=1e-7,
)


asd_parameters: Dict[str, Union[npt.NDArray[np.float32], float]] = { 
    "sig": np.array([[1.000, 0.894],
                     [0.894, 0.788]], dtype=np.float32), 
    "eps": np.array([[1.000, 0.342],
                     [0.342, 0.117]], dtype=np.float32), 
    "bonds": np.array([[0.584, 3000.], ]),
    "b_mass": 0.195,
}

ASD = SimulationVsNVT(
    name="NVT_ASD",
    description="""ASD""",
    root_folder="paper-output/",
    rho=1.863,  # For ASD this is atomic density

    vel_temperature=1.44,
    temperature=0.465,
    cells=[8, 8, 8],
    tau=0.2,
    dt=0.002,  # For some reasons it needs very this dt
    # steps=4,
    # steps_per_timeblock=2,
    # scalar_output=0,
    steps=2**22,
    steps_per_timeblock=2**16,
    scalar_output=2**10,

    pair_potential_name="ASD",
    pair_potential_params=asd_parameters,

    nvu_params_max_abs_val=2,
    # 10^-3 is the safest option
    nvu_params_threshold=1e-5,
    nvu_params_eps=5e-6,
    nvu_params_max_steps=10,
    nvu_params_max_initial_step_corrections=20,
    # I dont really know how to tune this parameter
    nvu_params_initial_step=0.5/1.863**(1/3),
    nvu_params_initial_step_if_high=1e-3,
    nvu_params_step=1,
    # nvu_params_save_path_u=True,
    nvu_params_raytracing_method="parabola-newton",
    nvu_params_mode="reflection-mass_scaling",
)
ASD.temperature_eq = rp.make_function_ramp(  # type: ignore
    value0=10.000, x0=ASD.dt * ASD.steps * (1 / 8),
    value1=ASD.temperature, x1=ASD.dt * ASD.steps * (1 / 4))

ASD_NO_SCALE = dataclasses.replace(
    ASD, name="NVT_ASD_noscaling",
    nvu_params_mode="reflection",
    temperature_eq = rp.make_function_ramp(  # type: ignore
        value0=10.000, x0=ASD.dt * ASD.steps * (1 / 8),
        value1=ASD.temperature, x1=ASD.dt * ASD.steps * (1 / 4))
)


USE_BACKUP = {
    "KA6": dataclasses.replace(KA_PARAMS[6], name="KA6.back"),
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r", "--rdf", help="Recalculate rdf for simulations", nargs='*', default=[])
    parser.add_argument("-a", "--rdf_all", help="Recalculate rdf for all", action="store_true")
    parser.add_argument("-m", "--method", help="Run Method", action="store_true")
    parser.add_argument("-l", "--lj", help="Run Lennard-Jones", action="store_true")
    parser.add_argument("-k", "--ka", help="Run Kob-Andersen", action="store_true")
    parser.add_argument("-n", "--no_inertia", help="Run No inertia", action="store_true")
    parser.add_argument("-s", "--asd", help="Run ASD", action="store_true")
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
    else:
        for i in range(3):
            try:
                print(f"Trying device {i}")
                cuda.select_device(i)
                break
            except Exception:
                pass

    for name in args.rdf:
        if name not in KNOWN_SIMULATIONS:
            raise ValueError(f"Expected one of `{', '.join(KNOWN_SIMULATIONS.keys())}` for the run argument, but got `{name}`")

    main(
        run_method=args.method, 
        run_lj=args.lj, 
        run_ka=args.ka, 
        run_no_inertia=args.no_inertia,
        run_asd=args.asd,
        rdf_new=set(args.rdf),
        rdf_all=args.rdf_all,
    )

