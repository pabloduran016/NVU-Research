from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import sys
import os
from typing import Any, Dict, Literal, Union, List, Optional, Tuple, Callable
import numpy as np
import numpy.typing as npt
import rumdpy as rp
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from scipy import stats


__all__ = [
    "KNOWN_SIMULATIONS",
    "SimulationParameters",
    "SimulationVsNVE",
    "SimulationVsNVT",
    "run_NVTorNVE",
    "run_NVU_RT",
    "plot_nvu_vs_figures",
    "save_current_figures_to_pdf",
]


def run_NVTorNVE(params: "SimulationParameters") -> None:
    temp_eq = params.temperature_eq if params.temperature_eq is not None else params.temperature
    if params.initial_conf is None:
        conf = rp.Configuration(D=3)
        conf.make_lattice(rp.unit_cells.FCC, cells=params.cells, rho=params.rho)
        conf["m"] = 1.0
        if params.pair_potential_name == "Kob-Andersen":
            conf.ptype[::5] = 1     # Every fifth particle set to type 1 (4:1 mixture)
        elif params.pair_potential_name == "ASD":
            B_particles = range(1, conf.N, 2)
            conf.ptype[B_particles] = 1  # Setting particle type of B particles
            conf['m'][B_particles] = params.pair_potential_params["b_mass"]  # Setting masses of B particles
        if callable(temp_eq):
            temp0 = temp_eq(0)
        else:
            temp0 = temp_eq
        conf.randomize_velocities(T=temp0)
    else:
        conf, _  = load_conf_from_npz(params.initial_conf)
        
    if type(params) == SimulationVsNVT:
        kind = "NVT"
        integrator_eq = rp.integrators.NVT(temperature=temp_eq, tau=params.tau, dt=params.dt)
        integrator_prod = rp.integrators.NVT(temperature=params.temperature, tau=params.tau, dt=params.dt)
        prod_output = params.nvt_output
        conf_output = params.nvt_conf_output
    elif type(params) == SimulationVsNVE:
        kind = "NVE"
        integrator_eq = integrator_prod = rp.integrators.NVE(dt=params.dt)
        prod_output = params.nve_output
        conf_output = params.nve_conf_output
    else:
        raise ValueError("Expected vs NVE or vs NVT")

    print(f"========== {kind} EQ ==========")
    sim = rp.Simulation(
        conf, params.get_pair_potential(conf), integrator_eq,
        num_timeblocks=params.num_timeblocks, steps_per_timeblock=params.steps_per_timeblock,
        storage="memory",
        conf_output="none",
        scalar_output="none",  # type: ignore
        verbose=True)
    for block in sim.timeblocks():
        print("block=", block, sim.status(per_particle=True))
    print(sim.summary())
    print(f"========== {kind} PROD ==========")
    sim = rp.Simulation(
        conf, params.get_pair_potential(conf), integrator_prod,
        num_timeblocks=params.num_timeblocks, 
        steps_per_timeblock=params.steps_per_timeblock,
        scalar_output=params.scalar_output,
        storage=prod_output, 
        verbose=True)
    for block in sim.timeblocks():
        print("block=", block, sim.status(per_particle=True))
    print(sim.summary())

    other_output_prod = rp.tools.load_output(prod_output)
    positions = other_output_prod["block"][:, :, 0, :, :]
    _, _, _n, d = positions.shape
    u, = rp.extract_scalars(other_output_prod, ["U"], first_block=0, D=d)
    target_u = np.mean(u[len(u)*3//4:])

    sim = rp.Simulation(
        conf, params.get_pair_potential(conf), integrator_prod,
        num_timeblocks=params.num_timeblocks, steps_per_timeblock=params.steps_per_timeblock,
        storage="memory",
        conf_output="none",
        scalar_output="none",  # type: ignore
        verbose=True)
    for block in sim.timeblocks():
        ev = rp.Evaluater(conf, params.get_pair_potential(conf))
        ev.evaluate()
        conf_u = np.sum(conf["u"])
        if conf_u <= target_u:
            save_conf_to_npz(conf_output, conf, target_u)
            return
    print("ERROR: Could not find a suitable configuration for NVU", file=sys.stderr)


def run_NVU_RT(params: "SimulationParameters", do_nvu_eq: bool) -> None:
    if type(params) == SimulationVsNVT:
        # kind = "NVT"
        prod_output = params.nvt_output
        conf_output = params.nvt_conf_output
    elif type(params) == SimulationVsNVE:
        # kind = "NVE"
        prod_output = params.nve_output
        conf_output = params.nve_conf_output
    else:
        raise ValueError("Expected vs NVE or vs NVT")

    if not os.path.exists(prod_output):
        print("WARNING: NVE configuration not found", file=sys.stderr)
        return

    # target_u = np.mean(u)
    other_prod_output = rp.tools.load_output(prod_output)
    conf, target_u = load_conf_from_npz(conf_output)
    integrator = rp.integrators.NVU_RT(
        target_u=target_u,
        max_abs_val=params.nvu_params_max_abs_val,
        threshold=params.nvu_params_threshold,
        eps=params.nvu_params_eps,
        max_steps=params.nvu_params_max_steps,
        max_initial_step_corrections=params.nvu_params_max_initial_step_corrections,
        initial_step=params.nvu_params_initial_step,
        initial_step_if_high=params.nvu_params_initial_step_if_high,
        step=params.nvu_params_step,
        debug_print=True,
        mode=params.nvu_params_mode,
        save_path_u=params.nvu_params_save_path_u,
        raytracing_method=params.nvu_params_raytracing_method
    )
    if do_nvu_eq:
        print("========== NVU EQ ==========")
        sim = rp.Simulation(
            conf, params.get_pair_potential(conf), integrator,
            num_timeblocks=params.num_timeblocks, 
            steps_per_timeblock=params.steps_per_timeblock,
            storage=params.nvu_eq_output,
            scalar_output=params.scalar_output,
            verbose=True,
            save_time=True,
        )
        for block in sim.timeblocks():
            print("block=", block, sim.status(per_particle=True))
        print(sim.summary())
        save_conf_to_npz(params.nvu_eq_conf_output, conf, target_u)
    else:
        conf, _target_u = load_conf_from_npz(params.nvu_eq_conf_output)

    print("========== NVU PROD ==========")
    sim = rp.Simulation(
        conf, params.get_pair_potential(conf), integrator,
        num_timeblocks=params.num_timeblocks, steps_per_timeblock=params.steps_per_timeblock,
        storage=params.nvu_output, 
        scalar_output=params.scalar_output,
        verbose=True,
        save_time=True,
    )
    for block in sim.timeblocks():
        print("block=", block, sim.status(per_particle=True))
    print(sim.summary())


def save_conf_to_npz(path: str, conf: rp.Configuration, target_u: float) -> None:
    np.savez(path, r=conf["r"], v=conf["v"], ptype=conf.ptype, target_u=target_u, simbox_initial=conf.simbox.lengths)


def load_conf_from_npz(path: str) -> Tuple[rp.Configuration, float]:
    conf_data = np.load(path)
    n, d = conf_data["r"].shape
    conf = rp.Configuration(N=n, D=d)
    conf["r"] = conf_data["r"]
    conf["m"] = 1.0
    conf["v"] = conf_data["v"]
    conf.ptype = conf_data["ptype"]
    conf.simbox = rp.Simbox(D=d, lengths=conf_data["simbox_initial"])
    return conf, float(conf_data["target_u"])



KNOWN_SIMULATIONS: Dict[str, "SimulationParameters"] = {}

@dataclass(kw_only=True)
class SimulationParameters:
    name: str
    description: str
    root_folder: str
    rho: float
    steps: int
    steps_per_timeblock: int
    scalar_output: int
    temperature: float | Callable[[float], float]
    temperature_eq: Optional[float | Callable[[float], float]] = None
    initial_conf: Optional[str] = None

    cells: List[int]

    pair_potential_name: Literal["LJ", "Kob-Andersen", "ASD"]
    pair_potential_params: Dict[str, Union[npt.NDArray[np.float32], float]]

    nvu_params_max_abs_val: float
    nvu_params_threshold: float
    nvu_params_eps: float
    nvu_params_max_steps: int
    nvu_params_max_initial_step_corrections: int
    nvu_params_initial_step: float
    nvu_params_initial_step_if_high: float
    nvu_params_step: float
    nvu_params_mode: Literal["reflection", "no-inertia"] = "reflection"
    nvu_params_save_path_u: bool = False
    nvu_params_raytracing_method: Literal["parabola", "parabola-newton", "bisection"] = "bisection"

    nvu_eq_conf_output = property(lambda self: os.path.join(self.folder, "nvu-rt_eq.npz"))
    nvu_eq_output = property(lambda self: os.path.join(self.folder, "nvu-rt_eq.h5"))
    nvu_output = property(lambda self: os.path.join(self.folder, "nvu-rt_prod.h5"))
    output_figs = property(lambda self: os.path.join(self.folder, "figures.pdf"))
    nvu_prod_rdf = property(lambda self: os.path.join(self.folder, "nvu-prod_rdf.npz"))
    other_prod_rdf = property(lambda self: os.path.join(self.folder, "other-prod_rdf.npz"))
    info_output = property(lambda self: os.path.join(self.folder, "info.txt"))
    folder = property(lambda self: os.path.join(self.root_folder, self.name))

    def __post_init__(self) -> None:
        self.num_timeblocks = self.steps//self.steps_per_timeblock

        if self.name in KNOWN_SIMULATIONS:
            raise ValueError(f"Simulation with name `{self.name}` already exists")

        KNOWN_SIMULATIONS[self.name] = self

    def init(self) -> None:
        if not os.path.exists(self.folder):
            print("Making dirs:", self.folder)
            os.makedirs(self.folder)

        self.create_info_file()

    def get_pair_potential(self, configuration: rp.Configuration) -> rp.PairPotential:
        if self.pair_potential_name == "LJ" or self.pair_potential_name == "Kob-Andersen":
            params = ("sig", "eps", "cut")
            if set(self.pair_potential_params.keys()) != set(params):
                raise ValueError(f"Parameters for potential `{self.pair_potential_name}` are {', '.join(params)}, but got {', '.join(self.pair_potential_params.keys())}")
            if self.pair_potential_name == "Kob-Andersen":
                for name, val in self.pair_potential_params.items():
                    if type(val) != np.ndarray:
                        raise ValueError(f"For potential Kob-Andersen every parameter has to be a (2, 2) numpy array, but got type {type(val)}")
                    if val.shape != (2, 2):
                        raise ValueError(f"For potential Kob-Andersen every parameter has to have shape (2, 2), but got {val.shape} for parameter {name}")
            pair_f = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
            param_values = tuple(self.pair_potential_params[name] for name in params)
            return rp.PairPotential(pair_f, param_values, max_num_nbs=1000)
        if self.pair_potential_name == "ASD":
            params = ("sig", "eps", "cut", "bonds", "b_mass")
            if set(self.pair_potential_params.keys()) != set(params):
                raise ValueError(f"Parameters for potential `{self.pair_potential_name}` are {', '.join(params)}, but got {', '.join(self.pair_potential_params.keys())}")
            for name, val in self.pair_potential_params.items():
                if name == "b_mass":
                    if type(val) != float:
                        raise ValueError(f"For potential ASD parameter `b_mass` has to be a float, but got type {type(val)}")
                    continue
                if type(val) != np.ndarray:
                    raise ValueError(f"For potential ASD parameter {name} has to be a (2, 2) numpy array, but got type {type(val)}")
                if name in ("sig", "eps", "cut"):
                    if val.shape != (2, 2):
                        raise ValueError(f"For potential ASD parameter {name} has to have shape (2, 2), but got {val.shape} for parameter {name}")
                elif name == "bonds":
                    if val.shape != (1, 2):
                        raise ValueError(f"For potential ASD parameter {name} has to have shape (1, 2), but got {val.shape} for parameter {name}")
                else:
                    assert False, "Unreachable"

            bond_potential = rp.harmonic_bond_function
            bond_params = self.pair_potential_params["bonds"]
            bond_indices = [[i, i + 1, 0] for i in range(0, configuration.N - 1, 2)]  # dumbells: i(even) and i+1 bonded with type 0
            bonds = rp.Bonds(bond_potential, bond_params, bond_indices)
            pair_f = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
            exclusions = bonds.get_exclusions(configuration)
            param_values = tuple(self.pair_potential_params[name] for name in ("sig", "eps", "cut"))
            return rp.PairPotential(pair_f, param_values, exclusions=exclusions, max_num_nbs=1000)
        else:
            raise ValueError(f"Unknwon potential name `{self.pair_potential_name}`")

    def create_info_file(self) -> None:
        with open(self.info_output, "w") as f:
            f.write(self.info())

    def info(self) -> str:
        info = f"{self.name}\n\n{self.description}\n\n"
        info += f"{self.pair_potential_name}\n"
        for name, val in self.pair_potential_params.items():
            info += f"{name} = {val}\n"
        info += f"\n"
        for name, val in (
            ("rho", self.rho),
            ("temperature", self.temperature),
            ("blocks", f"{self.num_timeblocks} = 2^{np.log2(self.num_timeblocks)}"),
            ("steps", f"{self.steps} = 2^{np.log2(self.steps)}"),
            ("steps_per_timeblock", f"{self.steps_per_timeblock} = 2^{np.log2(self.steps_per_timeblock)}"),
            ("scalar_output", f"{self.scalar_output}; scalars_per_block (approx) = {0 if self.scalar_output == 0 else self.steps_per_timeblock//self.scalar_output}"),
            ("cells", self.cells),
        ):
            info += f"{name} = {val}\n"
        info += f"\n"
        info += f"NVU\n"
        for name, val in (
            ("max_abs_val", self.nvu_params_max_abs_val),
            ("threshold", self.nvu_params_threshold),
            ("eps", self.nvu_params_eps),
            ("max_steps", self.nvu_params_max_steps),
            ("max_initial_step_corrections", self.nvu_params_max_initial_step_corrections),
            ("initial_step", self.nvu_params_initial_step),
            ("initial_step_if_high", self.nvu_params_initial_step_if_high),
            ("step", self.nvu_params_step),
            ("mode", self.nvu_params_mode),
            ("raytracing_method", self.nvu_params_raytracing_method),
        ):
            info += f"{name} = {val}\n"
        return info


@dataclass(kw_only=True)
class SimulationVsNVT(SimulationParameters):
    tau: float
    dt: float

    nvt_output = property(lambda self: os.path.join(self.folder, "nvt_prod.h5"))
    nvt_conf_output = property(lambda self: os.path.join(self.folder, "nvt.npz"))

    def info(self) -> str:
        info = "NVU vs NVT\n"
        info += super().info()
        info += f"\n"
        info += f"NVT\n"
        for name, val in (
            ("tau", self.tau),
            ("dt", self.dt),
        ):
            info += f"{name} = {val}\n"
        return info


@dataclass(kw_only=True)
class SimulationVsNVE(SimulationParameters):
    dt: float

    nve_output = property(lambda self: os.path.join(self.folder, "nve_prod.h5"))
    nve_conf_output = property(lambda self: os.path.join(self.folder, "nve.npz"))

    def info(self) -> str:
        info = "NVU vs NVE\n"
        info += super().info()
        info += f"\n"
        info += f"NVE\n"
        for name, val in (
            ("dt", self.dt),
        ):
            info += f"{name} = {val}\n"
        return info


def plot_nvu_vs_figures(params: SimulationParameters) -> None:
    if type(params) == SimulationVsNVT:
        kind = "NVT"
        other_prod_output_path = params.nvt_output
    elif type(params) == SimulationVsNVE:
        kind = "NVE"
        other_prod_output_path = params.nve_output
    else:
        raise ValueError("Expected vs NVE or vs NVT")
    
    other_prod_output = rp.tools.load_output(other_prod_output_path)

    _, _, _, n, d = other_prod_output["block"].shape

    other_u, other_k, = rp.extract_scalars(other_prod_output, ["U", "K"], first_block=0, D=d)
    _conf, target_u = load_conf_from_npz(params.nvu_eq_conf_output)
    other_du_rel = (other_u - target_u) / abs(target_u)

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(f"{kind} potential energy U")
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)

    prod_step = np.arange(len(other_u)) * other_prod_output["steps_between_output"]
    ax0.plot(prod_step, other_du_rel, linewidth=1, alpha=.8, color="black")
    ax0.set_xlabel("steps")
    ax0.set_ylabel(r"$\frac{U - U_0}{|U_0|}$")
    ax0.grid()
    ax1.hist(other_u, bins=20, alpha=.8, color="black")
    ax0.set_xlabel(r"$U$")
    ax1.axvline(target_u)
    ax1.grid()

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(f"{kind} Temperature")
    ax = fig.add_subplot()
    dof = d * n - d  # degrees of freedom
    temp = 2 * other_k / dof
    ax.plot(prod_step, temp, linewidth=1, alpha=.8, color="black")
    ax.grid()

    if not os.path.exists(params.nvu_output):
        print("WARNING: NVU PROD output not found", file=sys.stderr)
        return

    nvu_prod_output = rp.tools.load_output(params.nvu_output)
    nvu_eq_output = rp.tools.load_output(params.nvu_eq_output)

    nvu_prod_u, prod_dt, prod_its, prod_fsq, prod_lap, prod_cos_v_f, prod_time, = \
        rp.extract_scalars(nvu_prod_output, ["U", "dt", "its", "Fsq", "lapU", "cos_v_f", "time", ], 
                           integrator_outputs=rp.integrators.NVU_RT.outputs)
    prod_cos_v_f[prod_cos_v_f > 1] = 1
    prod_cos_v_f[prod_cos_v_f < -1] = -1
    prod_correction = (np.pi / 2 - np.arccos(prod_cos_v_f)) / prod_cos_v_f
    prod_dt = prod_dt * prod_correction

    nvu_prod_du_rel = (nvu_prod_u - target_u) / abs(target_u)
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Potential energy NVU")
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)
    ax0.semilogy(prod_step, np.abs(nvu_prod_du_rel), linewidth=0, marker='o', color="black")
    ax0.set_xlabel(r"$step$")
    ax0.set_ylabel(r"$\frac{U - U_0}{|U_0|}$")
    ax0.grid()
    ax1.hist(nvu_prod_du_rel, bins=30, color="black", alpha=.8)
    ax1.set_xlabel(r"$\frac{U - U_0}{|U_0|}$")
    ax1.grid()

    other_rdf = calculate_rdf(other_prod_output)
    nvu_rdf = calculate_rdf(nvu_prod_output)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()
    fig.suptitle("$g(r)$")
    ax.plot(other_rdf['distances'], np.mean(other_rdf['rdf'], axis=0), linewidth=1, color="black", alpha=.8, label=kind)
    ax.plot(nvu_rdf['distances'], np.mean(nvu_rdf['rdf'], axis=0), linewidth=0, marker='.', markersize=5, markeredgewidth=0, color="blue", alpha=.8, label="NVU")
    ax.grid()
    ax.legend()
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$g(r)$")

    nvu_msd = rp.tools.calc_dynamics(nvu_prod_output, first_block=0)["msd"][:, 0]
    other_msd = rp.tools.calc_dynamics(other_prod_output, first_block=0)["msd"][:, 0]
    nvu_time = np.mean(prod_dt) * 2 ** np.arange(len(nvu_msd))
    other_time = other_prod_output["attrs"]['dt'] * 2 ** np.arange(len(other_msd))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()
    fig.suptitle("MSD")
    ax.loglog(other_time, other_msd, linewidth=1, color="black", alpha=.8, label=kind)
    ax.loglog(nvu_time, nvu_msd, linewidth=0, marker='.', markersize=5, markeredgewidth=0, color="blue", alpha=.8, label="NVU")
    ax.grid()
    ax.legend()
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$MSD$")

    nblocks, nconfs, _, _, _ = nvu_prod_output["block"].shape
    step_conf = np.concatenate([params.steps_per_timeblock * i + 2 ** np.concatenate([[0], np.arange(nconfs-1)]) for i in range(nblocks)])
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("Time NVU")
    ax = fig.add_subplot()
    res = stats.linregress(prod_step, prod_time)
    xs = np.linspace(np.min(prod_step), np.max(prod_step))
    ax.plot(xs, xs * res.slope + res.intercept, ':', color="red", alpha=.5, label=rf"m = {res.slope}")  # type: ignore
    ax.plot(prod_step, prod_time, '--', color="black", alpha=.5, label="scalar time")
    ax.plot(step_conf, nvu_prod_output["time"].flatten(), linewidth=0, marker='.', markersize=5, markeredgewidth=0, color="blue", alpha=.5, label="conf time")
    ax.grid()
    ax.legend()
    ax.set_title(r"$t(step)$")
    ax.set_xlabel(r"$step$")
    ax.set_ylabel(r"$t$")

    fig = plt.figure(figsize=(10, 8))
    mean_dt, std_dt = np.mean(prod_dt), np.std(prod_dt)
    fig.suptitle(rf"$\Delta t$. $\mu={mean_dt:.04f}$; $\sigma={std_dt:.06f}={std_dt/mean_dt*100:.02f}\%$")
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)
    ax0.plot(prod_step, prod_dt, linewidth=1, color="black", alpha=.8)
    ax0.set_ylabel(r"$\Delta t$")
    ax0.set_xlabel(r"$step$")
    ax0.grid()
    ax1.hist(prod_dt[~np.isnan(prod_dt)], bins=20, color="black", alpha=.8)
    ax1.set_xlabel(r"$\Delta t$")

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(r"Correction to delta time so that $\Delta t' = \Delta t \cdot \frac{\alpha}{\cos \theta}$. $\alpha$ comp. of $\theta$")
    ax0 = fig.add_subplot(2, 1, 1)
    ax0.plot(prod_correction, color="black", alpha=.5)
    ax1 = fig.add_subplot(2, 1, 2)
    ax1.hist(prod_correction[~np.isnan(prod_correction)], color="black", alpha=.5)


    fig = plt.figure(figsize=(10, 8))
    mean_its, std_its = np.mean(prod_its), np.std(prod_its)
    fig.suptitle(rf"Iterations. $\mu={mean_its:.01f}$; $\sigma={std_its:.02f}={std_its/mean_its*100:.02f}\%$")
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)
    ax0.plot(prod_step, prod_its, linewidth=1, color="black", alpha=.8)
    ax0.set_ylabel(r"iterations")
    ax0.set_xlabel(r"$step$")
    ax0.grid()
    ax1.hist(prod_its, bins=20, color="black", alpha=.8)
    ax1.set_xlabel(r"iterations")
    ax1.grid()

    if "path_u" in nvu_prod_output:
        nblocks, nsaved_per_block, npoints, n = nvu_prod_output["path_u"].shape
        path_u = nvu_prod_output["path_u"].reshape(nblocks*nsaved_per_block, npoints, n)
        xs = path_u[:, :, 0].T  # (npoints, npaths)
        ys = path_u[:, :, 1].T  # (npoints, npaths)
        # Harmonic approximation:
        u = np.arange(npoints)/(n - 1) - 0.5
        p = np.polyfit(u, ys, 2)
        y_pred = p[0, :] * u[:, np.newaxis]**2 + p[1, :] * u[:, np.newaxis] + p[2, :]
        fig = plt.figure(figsize=(10, 8))
        r2 = 1 - np.sum((y_pred - ys)**2) / np.sum((y_pred - y_pred.mean(axis=0))**2)
        fig.suptitle(rf"$R^2 = {r2}$ when aproximating $U(\lambda)$ to a parabola")
        ax = fig.add_subplot()
        rsd = ((y_pred - ys)**2 / (ys - ys.mean(axis=0))**2).flatten()
        ax.hist(rsd[(~np.isnan(rsd)) & (np.abs(rsd) < float("inf"))], bins=30, color="black", alpha=0.5)
        # ax.plot(rsd, marker='.', markeredgewidth=0, linewidth=0, markersize=5,
        #         color="black", alpha=0.5)
        # ax.hist((y_pred/ys).flatten(), bins=20, color="black", alpha=0.5)

    
    fig = plt.figure(figsize=(10, 8))
    mean_vf, std_vf = np.mean(prod_cos_v_f), np.std(prod_cos_v_f)
    fig.suptitle(rf"$cos(v,\,f)$. $\mu={mean_vf:.03f}$; $\sigma={std_vf:.05f}={std_vf/mean_vf*100:.02f}\%$")
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)
    ax0.plot(prod_step, prod_cos_v_f, linewidth=1, color="black", alpha=.8)
    ax0.set_ylabel(r"$cos(v,\,f)$")
    ax0.set_xlabel(r"$step$")
    ax0.grid()
    ax1.hist(prod_cos_v_f[~np.isnan(prod_cos_v_f)], bins=20, color="black", alpha=.8)
    ax1.set_xlabel(r"$cos(v,\,f)$")
    ax1.grid()
    
    fig = plt.figure(figsize=(10, 8))
    t_conf = prod_fsq / prod_lap
    mean_t_conf, std_t_conf = np.mean(t_conf), np.std(t_conf)
    fig.suptitle(rf"$k_BT_{{conf}}$. $\mu={mean_t_conf:.03f}$; $\sigma={std_t_conf:.04f}={std_t_conf/mean_t_conf*100:.02f}\%$")
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)
    ax0.plot(prod_step, t_conf, linewidth=1, color="black", alpha=.8)
    ax0.set_ylabel(r"$k_BT_{conf}$")
    ax0.set_xlabel(r"$step$")
    ax0.grid()
    ax1.hist(t_conf[~np.isnan(t_conf)], bins=20, color="black", alpha=.8)
    ax1.set_xlabel(r"$k_BT_{conf}$")
    ax1.grid()

    fig = plt.figure(figsize=(10, 8))
    kappa = prod_lap / np.sqrt(prod_fsq)
    mean_kappa, std_kappa = np.nanmean(kappa), np.nanstd(kappa)
    fig.suptitle(rf"$\kappa$. $\mu={mean_kappa:.01f}$; $\sigma={std_kappa:.02f}={std_kappa/mean_kappa*100:.02f}\%$")
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)
    ax0.plot(prod_step, kappa, linewidth=1, color="black", alpha=.8)
    ax0.set_ylabel(r"$\kappa$")
    ax0.set_xlabel(r"$step$")
    ax0.grid()
    kappa_ok = kappa[(~np.isnan(kappa)) & (np.abs(kappa) < float("inf"))]
    ax1.hist(kappa_ok, bins=20, color="black", alpha=.8)
    ax1.set_xlabel(r"$\kappa$")
    ax1.grid()

    fig = plt.figure(figsize=(10, 8))
    x, y = prod_dt, prod_cos_v_f
    fig.suptitle(rf"Correlation: $\Delta t$ vs $\cos(v,\,f)$. $cov = {get_cov(x, y):.04f}$")
    ax = fig.add_subplot()
    ax.plot(x, y, linewidth=0, marker='.', markersize=5, markeredgewidth=0, color="black", alpha=.8)
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$cos(v,\,f)$")
    ax.grid()
    
    fig = plt.figure(figsize=(10, 8))
    x, y = prod_dt, t_conf
    fig.suptitle(rf"Correlation: $\Delta t$ vs $k_BT_{{conf}}$. $cov = {get_cov(x, y):.04f}$")
    ax = fig.add_subplot()
    ax.plot(x, y, linewidth=0, marker='.', markersize=5, markeredgewidth=0, color="black", alpha=.8)
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$k_BT_{conf}$")
    ax.grid()

    fig = plt.figure(figsize=(10, 8))
    x, y = prod_dt, kappa
    fig.suptitle(rf"Correlation: $\Delta t$ vs $\kappa$. $cov = {get_cov(x, y):.04f}$")
    ax = fig.add_subplot()
    ax.plot(x, y, linewidth=0, marker='.', markersize=5, markeredgewidth=0, color="black", alpha=.8)
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$\kappa$")
    ax.grid()

    fig = plt.figure(figsize=(10, 8))
    x, y = prod_cos_v_f, kappa
    fig.suptitle(rf"Correlation: $\cos(v,\,f)$ vs $\kappa$. $cov = {get_cov(x, y):.04f}$")
    ax = fig.add_subplot()
    ax.plot(x[~np.isnan(x)], y[~np.isnan(x)], linewidth=0, marker='.', markersize=5, markeredgewidth=0, color="black", alpha=.8)
    ax.set_xlabel(r"$\cos(v,\,f)$")
    ax.set_ylabel(r"$\kappa$")
    ax.grid()

    nvu_eq_u, eq_dt, eq_its, eq_fsq, eq_lap, eq_cos_v_f, eq_time = \
        rp.extract_scalars(nvu_eq_output, ["U", "dt", "its", "Fsq", "lapU", "cos_v_f", "time", ], 
                           integrator_outputs=rp.integrators.NVU_RT.outputs, first_block=1)
    eq_cos_v_f[eq_cos_v_f > 1] = 1
    eq_cos_v_f[eq_cos_v_f < 1] = -1
    eq_step = np.arange(len(nvu_eq_u)) * nvu_eq_output["steps_between_output"]

    nvu_eq_du_rel = (nvu_eq_u - target_u) / abs(target_u)
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Potential energy NVU EQ")
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)
    ax0.semilogy(eq_step, np.abs(nvu_eq_du_rel), linewidth=0, marker='o', color="black")
    ax0.set_xlabel(r"$step$")
    ax0.set_ylabel(r"$\frac{U - U_0}{|U_0|}$")
    ax0.grid()
    ax1.hist(nvu_eq_du_rel, bins=30, color="black", alpha=.8)
    ax1.set_xlabel(r"$\frac{U - U_0}{|U_0|}$")
    ax1.grid()

    nblocks, nconfs, _, _, _ = nvu_eq_output["block"][1:, :, :, :, :].shape
    step_conf = np.concatenate([params.steps_per_timeblock * i + 2 ** np.concatenate([[0], np.arange(nconfs-1)]) for i in range(nblocks)])
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("Time NVU EQ")
    ax = fig.add_subplot()
    res = stats.linregress(eq_step, eq_time)
    xs = np.linspace(np.min(eq_step), np.max(eq_step))
    ax.plot(xs, xs * res.slope + res.intercept, ':', color="red", alpha=.5, label=rf"m = {res.slope}")  # type: ignore
    ax.plot(eq_step, eq_time, '--', color="black", alpha=.5, label="scalar time")
    ax.plot(step_conf, nvu_eq_output["time"][1:, :].flatten(), linewidth=0, marker='.', markersize=5, markeredgewidth=0, color="blue", alpha=.5, label="conf time")
    ax.grid()
    ax.legend()
    ax.set_title(r"$t(step)$")
    ax.set_xlabel(r"$step$")
    ax.set_ylabel(r"$t$")

    fig = plt.figure(figsize=(10, 8))
    mean_dt, std_dt = np.mean(eq_dt), np.std(eq_dt)
    fig.suptitle(rf"$\Delta t$. $\mu={mean_dt:.04f}$; $\sigma={std_dt:.06f}={std_dt/mean_dt*100:.02f}\%$")
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)
    ax0.plot(eq_step, eq_dt, linewidth=1, color="black", alpha=.8)
    ax0.set_ylabel(r"$\Delta t$")
    ax0.set_xlabel(r"$step$")
    ax0.grid()
    ax1.hist(eq_dt, bins=20, color="black", alpha=.8)
    ax1.set_xlabel(r"$\Delta t$")

    fig = plt.figure(figsize=(10, 8))
    mean_its, std_its = np.mean(eq_its), np.std(eq_its)
    fig.suptitle(rf"NVU EQ Iterations. $\mu={mean_its:.01f}$; $\sigma={std_its:.02f}={std_its/mean_its*100:.02f}\%$")
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)
    ax0.plot(eq_step, eq_its, linewidth=1, color="black", alpha=.8)
    ax0.set_ylabel(r"iterations")
    ax0.set_xlabel(r"$step$")
    ax0.grid()
    ax1.hist(eq_its, bins=20, color="black", alpha=.8)
    ax1.set_xlabel(r"iterations")
    ax1.grid()
    
    fig = plt.figure(figsize=(10, 8))
    mean_vf, std_vf = np.mean(eq_cos_v_f), np.std(eq_cos_v_f)
    fig.suptitle(rf"NVU EQ. $cos(v,\,f)$. $\mu={mean_vf:.03f}$; $\sigma={std_vf:.05f}={std_vf/mean_vf*100:.02f}\%$")
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)
    ax0.plot(eq_step, eq_cos_v_f, linewidth=1, color="black", alpha=.8)
    ax0.set_ylabel(r"$cos(v,\,f)$")
    ax0.set_xlabel(r"$step$")
    ax0.grid()
    ax1.hist(eq_cos_v_f, bins=20, color="black", alpha=.8)
    ax1.set_xlabel(r"$cos(v,\,f)$")
    ax1.grid()
    
    fig = plt.figure(figsize=(10, 8))
    t_conf = eq_fsq / eq_lap
    mean_t_conf, std_t_conf = np.nanmean(t_conf[np.abs(t_conf) < float("inf")]
), np.nanstd(t_conf[np.abs(t_conf) < float("inf")]
)
    fig.suptitle(rf"NVU EQ. $k_BT_{{conf}}$. $\mu={mean_t_conf:.03f}$; $\sigma={std_t_conf:.04f}={std_t_conf/mean_t_conf*100:.02f}\%$")
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)
    ax0.plot(eq_step[np.abs(t_conf) < float("inf")], t_conf[np.abs(t_conf) < float("inf")], linewidth=1, color="black", alpha=.8)
    ax0.set_ylabel(r"$k_BT_{conf}$")
    ax0.set_xlabel(r"$step$")
    ax0.grid()
    ax1.hist(t_conf[np.abs(t_conf) < float("inf")], bins=20, color="black", alpha=.8)
    ax1.set_xlabel(r"$k_BT_{conf}$")
    ax1.grid()

    fig = plt.figure(figsize=(10, 8))
    kappa = eq_lap / np.sqrt(eq_fsq)
    mean_kappa, std_kappa = np.nanmean(kappa), np.nanstd(kappa)
    fig.suptitle(rf"NVU EQ. $\kappa$. $\mu={mean_kappa:.01f}$; $\sigma={std_kappa:.02f}={std_kappa/mean_kappa*100:.02f}\%$")
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)
    ax0.plot(eq_step, kappa, linewidth=1, color="black", alpha=.8)
    ax0.set_ylabel(r"$\kappa$")
    ax0.set_xlabel(r"$step$")
    ax0.grid()
    kappa_ok = kappa[(~np.isnan(kappa)) & (np.abs(kappa) < float("inf"))]
    ax1.hist(kappa_ok, bins=20, color="black", alpha=.8)
    ax1.set_xlabel(r"$\kappa$")
    ax1.grid()

    fig = plt.figure(figsize=(10, 8))
    x, y = eq_dt, eq_cos_v_f
    fig.suptitle(rf"NVU EQ. Correlation: $\Delta t$ vs $\cos(v,\,f)$. $cov = {get_cov(x, y):.04f}$")
    ax = fig.add_subplot()
    ax.plot(x, y, linewidth=0, marker='.', markersize=5, markeredgewidth=0, color="black", alpha=.8)
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$cos(v,\,f)$")
    ax.grid()
    
    fig = plt.figure(figsize=(10, 8))
    x, y = eq_dt, t_conf
    fig.suptitle(rf"NVU EQ. Correlation: $\Delta t$ vs $k_BT_{{conf}}$. $cov = {get_cov(x, y):.04f}$")
    ax = fig.add_subplot()
    ax.plot(x, y, linewidth=0, marker='.', markersize=5, markeredgewidth=0, color="black", alpha=.8)
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$k_BT_{conf}$")
    ax.grid()

    fig = plt.figure(figsize=(10, 8))
    x, y = eq_dt, kappa
    fig.suptitle(rf"NVU EQ. Correlation: $\Delta t$ vs $\kappa$. $cov = {get_cov(x, y):.04f}$")
    ax = fig.add_subplot()
    ax.plot(x, y, linewidth=0, marker='.', markersize=5, markeredgewidth=0, color="black", alpha=.8)
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$\kappa$")
    ax.grid()

    fig = plt.figure(figsize=(10, 8))
    x, y = eq_cos_v_f, kappa
    fig.suptitle(rf"NVU EQ. Correlation: $\cos(v,\,f)$ vs $\kappa$. $cov = {get_cov(x, y):.04f}$")
    ax = fig.add_subplot()
    ax.plot(x, y, linewidth=0, marker='.', markersize=5, markeredgewidth=0, color="black", alpha=.8)
    ax.set_xlabel(r"$\cos(v,\,f)$")
    ax.set_ylabel(r"$\kappa$")
    ax.grid()



def get_cov(x: npt.NDArray[np.float32], y: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: 
    (xx, xy), (_yx, yy) = np.cov(x, y)
    c = xy / np.sqrt(xx*yy)
    return c


def calculate_rdf(output: dict[str, Any]) -> dict[str, Any]:
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

    cal_rdf.save_average()
    rdf = cal_rdf.read()
    return rdf

def save_current_figures_to_pdf(path0: str):
    name, ext = os.path.splitext(path0)
    if os.path.exists(path0):
        i = 1
        while os.path.exists(f"{name}({i}){ext}"):
            i += 1
        path = f"{name}({i}){ext}"
    else:
        path = path0
    print(f"Saving figures to {path}")
    pages = PdfPages(path)
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pages, format='pdf')  # type: ignore
    pages.close()

