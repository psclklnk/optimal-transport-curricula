import os
import pickle
import numpy as np
from pathlib import Path
from util import add_plot
import matplotlib.pyplot as plt
from deep_sprl.experiments import EMazeExperiment
from deep_sprl.environments.emaze import generate_e_maze

FONT_SIZE = 8
TICK_SIZE = 6

plt.rc('text.latex', preamble=r'\usepackage{amsmath}'
                              r'\newcommand{\currot}{\textsc{currot}}'
                              r'\newcommand{\gradient}{\textsc{gradient}}'
                              r'\newcommand{\sprl}{\textsc{sprl}}'
                              r'\newcommand{\alpgmm}{\textsc{alp-gmm}}'
                              r'\newcommand{\goalgan}{\textsc{goalgan}}'
                              r'\newcommand{\acl}{\textsc{acl}}'
                              r'\newcommand{\vds}{\textsc{vds}}'
                              r'\newcommand{\plr}{\textsc{plr}}')
plt.rcParams.update({
    "text.usetex": True,
    "font.serif": "Helvetica",
    "font.family": "serif"
})


def performance_plot(experiment, base_log_dir, learners, methods, labels, colors=None, params=None, ax=None,
                     ncol=3, bbox_to_anchor=(0.135, 1.03), single_column: bool = False):
    if ax is None:
        f = plt.figure(figsize=(5.7 if single_column else 3.0, 1.8))
        ax = plt.Axes(f, [0.1 if single_column else 0.14, 0.175, 0.82, 0.7 if single_column else 0.5])
        f.add_axes(ax)
    else:
        f = plt.gcf()

    if params is None:
        params = [{}] * len(learners)

    if colors is None:
        colors = [f"C{i}" for i in range(len(learners))]

    lines = []
    for learner, method, cur_params, color in zip(learners, methods, params, colors):
        exp = experiment(base_log_dir, method, learner, cur_params, seed=0)
        log_dir = os.path.dirname(exp.get_log_dir())
        lines.append(add_plot(log_dir, ax, color))

    # This is the optimal return
    ax.hlines(0.63089, 0, 100, colors="red", linestyles="--")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50000 / experiment.STEPS_PER_ITER, 100000 / experiment.STEPS_PER_ITER,
                   150000 / experiment.STEPS_PER_ITER, 200000 / experiment.STEPS_PER_ITER])
    ax.set_xticklabels(["$0$", "$50$K", "$100$K", "$150$K", "$200$K"])
    ax.set_yticks([0., 0.2, 0.4, 0.6])
    ax.set_yticklabels(["$0$", "$0.2$", "$0.4$", "$0.6$"])
    ax.set_ylim(-0.05, 0.65)

    ax.set_ylabel(r"Episodic Return", fontsize=FONT_SIZE, labelpad=2.)
    ax.set_xlabel(r"Step", fontsize=FONT_SIZE, labelpad=2.)
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    f.legend(lines, labels,
             fontsize=FONT_SIZE, loc='upper left', bbox_to_anchor=bbox_to_anchor, ncol=ncol,
             columnspacing=0.4, handlelength=0.7 if single_column else 0.9, handletextpad=0.25)

    ax.grid()

    return f, ax


def generated_interpolation_vis_v2(experiment, method, learner, params, labels, seeds, steps, single_column=False):
    f = plt.figure(figsize=(2.3 if single_column else 3.0, 0.9 if single_column else 1.1))

    vis = generate_e_maze(20, 200)[1]
    bg_img = np.zeros((20, 20, 3))
    bg_img[vis == 0] = 1

    for i, (par, label, cur_steps, seed) in enumerate(zip(params, labels, steps, seeds)):
        exp = experiment("../logs", method, learner, par, seed=seed)

        ax = plt.Axes(f, [i * 0.33 + 0.01, 0.01, 0.325, 0.82])
        f.add_axes(ax)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        ax.imshow(bg_img)
        ax.set_title(label, fontsize=FONT_SIZE, pad=3 if single_column else 2.5)

        context_trace = Path(exp.get_log_dir()) / "context_trace.pkl"
        with open(context_trace, "rb") as file:
            distributions = pickle.load(file)
            if isinstance(distributions, tuple):
                distributions = distributions[1]

        cm = plt.get_cmap('cividis')
        for i, step in enumerate(cur_steps):
            dist = np.reshape(distributions[step], (20, 20))
            dist_img = np.array(cm(i / (len(cur_steps) - 1))) * np.ones((20, 20, 4))
            dist_img[..., -1] = np.where(dist > 1e-3, 0.5, 1.) * dist + np.where(dist > 1e-3, 0.5, 0.)

            ax.imshow(dist_img)

    return f, ax


def generated_interpolation_vis(experiment, method, learner, params, seed, steps, single_column: bool = False):
    f = plt.figure(figsize=(0.8, 1.8) if single_column else (3.0, 1.1))

    vis = generate_e_maze(20, 200)[1]
    bg_img = np.zeros((20, 20, 3))
    bg_img[vis == 0] = 1

    for i, (dist_fn, label, cur_steps) in enumerate(zip(["shortest_path", "opt_perf", "euclidean"],
                                                        [r"$d_{\text{S}}(c_1, c_2)$", r"$d_{\text{P}^*}(c_1, c_2)$",
                                                         r"$d_{\text{E}}(c_1, c_2)$"], steps)):
        params["dist_fn"] = dist_fn
        exp = experiment("../logs", method, learner, params, seed=seed)

        if single_column:
            ax = plt.Axes(f, [0.01, (2 - i) * 0.33 + 0.01, 0.98, 0.325])
        else:
            ax = plt.Axes(f, [i * 0.33 + 0.01, 0.01, 0.325, 0.88])
        f.add_axes(ax)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        ax.imshow(bg_img, aspect=0.725 if single_column else 1.)
        if not single_column:
            ax.set_title(label, fontsize=FONT_SIZE, pad=2.5)

        context_trace = Path(exp.get_log_dir()) / "context_trace.pkl"
        with open(context_trace, "rb") as file:
            distributions = pickle.load(file)
            if isinstance(distributions, tuple):
                distributions = distributions[1]

        cm = plt.get_cmap('cividis')
        for i, step in enumerate(cur_steps):
            dist = np.reshape(distributions[step], (20, 20))
            dist_img = np.array(cm(i / (len(cur_steps) - 1))) * np.ones((20, 20, 4))
            dist_img[..., -1] = np.where(dist > 1e-3, 0.5, 1.) * dist + np.where(dist > 1e-3, 0.5, 0.)

            ax.imshow(dist_img, aspect=0.725 if single_column else 1.)

        ax.plot((-0.5, 2.5), (2.5, 2.5), color=cm(0.))
        ax.plot((2.5, 2.5), (2.5, -0.5), color=cm(0.))

        ax.plot((8.5, 8.5), (-0.5, 2.5), color="red")
        ax.plot((8.5, 11.5), (2.5, 2.5), color="red")
        ax.plot((11.5, 11.5), (2.5, -0.5), color="red")

    return f, ax


def env_figure(experiment, single_column: bool = False):
    f = plt.figure(figsize=(0.95, 1.8) if single_column else (3.0, 1.1))

    vis = generate_e_maze(20, 200)[1]
    bg_img = np.zeros((20, 20, 3))
    bg_img[vis == 0] = 1

    for i, (dist_fn, label) in enumerate(zip(["shortest_path", "opt_perf", "euclidean"],
                                             [r"$d_{\text{S}}(c_1, c_2)$", r"$d_{\text{P}^*}(c_1, c_2)$",
                                              r"$d_{\text{E}}(c_1, c_2)$"])):
        exp = experiment("../logs", "gradient", "ppo", {"GRADIENT_ENT": 1e-4, "dist_fn": dist_fn}, seed=1)

        if single_column:
            extent = [0.17, (2 - i) * 0.33 + 0.01, 0.825, 0.325]
        else:
            extent = [i * 0.33 + 0.01, 0.01, 0.325, 0.88]
        ax = plt.Axes(f, extent)
        f.add_axes(ax)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        ax.imshow(bg_img, aspect=0.725 if single_column else 1.)
        if single_column:
            ax.set_ylabel(label, fontsize=FONT_SIZE, labelpad=2.5)
        else:
            ax.set_title(label, fontsize=FONT_SIZE, pad=2.5)

        cm = plt.get_cmap('cividis')
        env, __ = exp.create_environment()
        gradient_teacher = env.teacher
        for idx in list(range(0, len(gradient_teacher.prec_alphas), 2)) + [len(gradient_teacher.prec_alphas) - 1]:
            dist = np.reshape(gradient_teacher.interpolations[idx], (20, 20))
            dist_img = np.array(cm(gradient_teacher.prec_alphas[idx])) * np.ones((20, 20, 4))
            dist_img[..., -1] = np.where(dist > 1e-3, 0.5, 1.) * dist + np.where(dist > 1e-3, 0.5, 0.)
            ax.imshow(dist_img, aspect=0.725 if single_column else 1.)

        ax.plot((-0.5, 2.5), (2.5, 2.5), color=cm(0.))
        ax.plot((2.5, 2.5), (2.5, -0.5), color=cm(0.))

        ax.plot((8.5, 8.5), (-0.5, 2.5), color="red")
        ax.plot((8.5, 11.5), (2.5, 2.5), color="red")
        ax.plot((11.5, 11.5), (2.5, -0.5), color="red")

    return f, ax


def print_performance(method, learner, params):
    for param in params:
        exp = EMazeExperiment("../logs", method, learner, param, seed=1)
        base_log_dir = Path(exp.get_log_dir()).parent

        seeds = [int(d.split("-")[1]) for d in os.listdir(base_log_dir) if d.startswith("seed-")]
        max_perfs = []
        for seed in seeds:
            seed_dir = "seed-" + str(seed)
            seed_log_dir = os.path.join(base_log_dir, seed_dir)

            if os.path.exists(os.path.join(seed_log_dir, "performance.pkl")):
                with open(os.path.join(seed_log_dir, "performance.pkl"), "rb") as f:
                    # max_perfs.append(np.max(pickle.load(f)))
                    max_perfs.append(pickle.load(f)[-1])

        mu_max_perf = np.mean(max_perfs)
        se_max_perf = np.std(max_perfs) / np.sqrt(len(max_perfs))
        print(f"{mu_max_perf :.2f} \\pm {se_max_perf :.2f}")


if __name__ == "__main__":
    single_column = True
    dir_name = "figures_sc" if single_column else "figures"
    os.makedirs(dir_name, exist_ok=True)

    f, ax = performance_plot(EMazeExperiment, "../logs",
                             learners=["ppo", "ppo", "ppo", "ppo", "ppo", "ppo", "ppo", "ppo"],
                             methods=["default", "random", "wasserstein", "wasserstein", "wasserstein",
                                      "gradient", "gradient", "gradient"],
                             colors=["C2", "C3", "C0", "C4", "C5", "C1", "C6", "C7"],
                             labels=[r"Default", r"Random", r"$\textrm{\currot}_{d_{\text{S}}}$",
                                     r"$\textrm{\currot}_{d_{\text{P}^*}}$",
                                     r"$\textrm{\currot}_{d_{\text{E}}}$",
                                     r"$\textrm{\gradient}_{d_{\text{S}}}$",
                                     r"$\textrm{\gradient}_{d_{\text{P}^*}}$",
                                     r"$\textrm{\gradient}_{d_{\text{E}}}$"],
                             params=[{"dist_fn": "shortest_path"}, {"dist_fn": "shortest_path"},
                                     {"dist_fn": "shortest_path", "ENT_LB": 0.},
                                     {"dist_fn": "opt_perf", "ENT_LB": 0.},
                                     {"dist_fn": "euclidean", "ENT_LB": 0.},
                                     {"GRADIENT_ENT": 0., "dist_fn": "shortest_path"},
                                     {"GRADIENT_ENT": 0., "dist_fn": "opt_perf"},
                                     {"GRADIENT_ENT": 0., "dist_fn": "euclidean"}])

    plt.savefig(f"{dir_name}/emaze_performance.pdf")
    plt.close(f)

    f, ax = env_figure(EMazeExperiment, single_column=single_column)
    plt.savefig(f"{dir_name}/emaze_env.pdf")
    plt.close(f)

    f, ax = generated_interpolation_vis(EMazeExperiment, "wasserstein", "ppo",
                                        {"dist_fn": "shortest_path", "ENT_LB": 0.},
                                        seed=1, steps=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 15, 17, 20],
                                                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 15, 17, 20],
                                                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 15, 17, 20]],
                                        single_column=single_column)
    plt.savefig(f"{dir_name}/emaze_currot_interpolations.pdf")
    plt.close(f)

    f, ax = generated_interpolation_vis(EMazeExperiment, "wasserstein", "ppo",
                                        {"dist_fn": "shortest_path", "ENT_LB": 2.},
                                        seed=2, steps=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 15, 17, 20, 50],
                                                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 15, 17, 20, 50],
                                                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 15, 17, 20, 50]],
                                        single_column=single_column)
    plt.savefig(f"{dir_name}/emaze_currot_interpolations_entropy.pdf")
    plt.close(f)

    print("wasserstein")
    for dist_fn in ["shortest_path", "opt_perf", "euclidean"]:
        print(dist_fn)
        print_performance("wasserstein", "ppo",
                          [{"dist_fn": dist_fn, "ENT_LB": 0.}, {"dist_fn": dist_fn, "ENT_LB": 0.5},
                           {"dist_fn": dist_fn, "ENT_LB": 1.0}, {"dist_fn": dist_fn, "ENT_LB": 2.0}])

    print("cur_perf")
    print_performance("wasserstein", "ppo",
                      [{"dist_fn": "cur_perf", "ENT_LB": 0.}, {"dist_fn": "cur_perf", "ENT_LB": 0.5},
                       {"dist_fn": "cur_perf", "ENT_LB": 1.0}, {"dist_fn": "cur_perf", "ENT_LB": 2.0}])

    print("gradient")
    for dist_fn in ["shortest_path", "opt_perf", "euclidean"]:
        print(dist_fn)
        print_performance("gradient", "ppo",
                          [{"dist_fn": dist_fn, "GRADIENT_ENT": 0.}, {"dist_fn": dist_fn, "GRADIENT_ENT": 1e-8},
                           {"dist_fn": dist_fn, "GRADIENT_ENT": 1e-4}, {"dist_fn": dist_fn, "GRADIENT_ENT": 1e-2}])

    print("cur_perf")
    print_performance("gradient", "ppo",
                      [{"dist_fn": "cur_perf", "GRADIENT_ENT": 0.}, {"dist_fn": "cur_perf", "GRADIENT_ENT": 1e-8}])

    f, ax = generated_interpolation_vis_v2(EMazeExperiment, "wasserstein", "ppo",
                                           params=[{"dist_fn": "cur_perf", "ENT_LB": 0.5},
                                                   {"dist_fn": "cur_perf", "ENT_LB": 0.5},
                                                   {"dist_fn": "cur_perf", "ENT_LB": 0.5}],
                                           seeds=[5, 10, 2],
                                           labels=[r"\currot ($d_{\text{P}}$)",
                                                   r"\currot ($d_{\text{P}}$)",
                                                   r"\currot ($d_{\text{P}}$)"],
                                           steps=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 15, 17],
                                                  [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 15, 17],
                                                  [0, 1, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22,
                                                   35]], single_column=True)
    plt.savefig(f"{dir_name}/currot_cur_perf_interpolations.svg")
    plt.close(f)

    f, ax = generated_interpolation_vis_v2(EMazeExperiment, "gradient", "ppo",
                                           params=[{"dist_fn": "cur_perf", "GRADIENT_ENT": 1e-8},
                                                   {"dist_fn": "cur_perf", "GRADIENT_ENT": 1e-8},
                                                   {"dist_fn": "cur_perf", "GRADIENT_ENT": 1e-8}],
                                           seeds=[1, 5, 10],
                                           labels=[r"\gradient ($d_{\text{P}}$)",
                                                   r"\gradient ($d_{\text{P}}$)",
                                                   r"\gradient ($d_{\text{P}}$)"],
                                           steps=[[0, 2, 4, 5, 6, 7, 8, 10, 13, 15, 17, 22, 30, 40],
                                                  [0, 2, 4, 5, 6, 7, 8, 10, 13, 15, 17, 22, 30, 40],
                                                  [0, 2, 4, 5, 6, 7, 8, 10, 13, 15, 17, 22, 30, 45]],
                                           single_column=True)
    plt.savefig(f"{dir_name}/gradient_cur_perf_interpolations.pdf")
    plt.close(f)

    f, ax = performance_plot(EMazeExperiment, "../logs",
                             learners=["ppo", "ppo", "ppo", "ppo"],
                             methods=["wasserstein", "wasserstein", "gradient", "gradient"],
                             colors=["C0", "C4", "C1", "C6"],
                             labels=[r"\currot ($d_{\text{P}}$, $H_{\text{LB}}{=}0$)",
                                     r"\currot ($d_{\text{P}}$, $H_{\text{LB}}{=}0.5$)",
                                     r"\gradient ($d_{\text{P}}$, $\lambda{=}0$)",
                                     r"\gradient ($d_{\text{P}}$, $\lambda{=}®10^{-8}$)"],
                             params=[{"dist_fn": "cur_perf", "ENT_LB": 0.},
                                     {"dist_fn": "cur_perf", "ENT_LB": 0.5},
                                     {"GRADIENT_ENT": 0., "dist_fn": "cur_perf"},
                                     {"GRADIENT_ENT": 1e-8, "dist_fn": "cur_perf"}], ncol=2,
                             bbox_to_anchor=(0.025, 1.02))

    plt.savefig(f"{dir_name}/emaze_cur_perf_performance.pdf")
    plt.close(f)
