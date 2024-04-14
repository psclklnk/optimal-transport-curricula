import os
import numpy as np
import matplotlib.pyplot as plt
from misc.util import get_seed_performance

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
    "font.family": "serif",
    "font.serif": ["Helvetica"],
})

FONT_SIZE = 8
TICK_SIZE = 6


def high_dim_performance(methods, colors, extra_opts, text, base_log_dir="logs", ax=None, ylabel: bool = True):
    from deep_sprl.experiments.point_mass_nd_experiment import PointMassNDExperiment

    legend = False
    if ax is None:
        f = plt.figure(figsize=(3.1, 1.4))
        ax = f.add_axes([0.1, 0.22, 0.88, 0.61])
        legend = True
    else:
        f = None

    lines = []
    for method, color, extra_opt in zip(methods, colors, extra_opts):
        perfs = []
        for idx in range(0, 10):
            opts = {"DIM": idx + 1, }
            opts.update(extra_opt)

            exp = PointMassNDExperiment(base_log_dir, method, "ppo", opts, seed=0)
            log_dir = os.path.dirname(exp.get_log_dir())
            perfs.append(get_seed_performance(log_dir, max_seed=None)[1][:, -1])
        perfs = np.array(perfs)
        mid = np.mean(perfs, axis=1)
        lines.append(ax.plot(3 * np.arange(1, 11), mid, color=color)[0])
        sem = np.std(perfs, axis=1) / np.sqrt(perfs.shape[1])
        low = mid - 2 * sem
        high = mid + 2 * sem
        ax.fill_between(3 * np.arange(1, 11), low, high, color=color, alpha=0.5)

    lines.append(ax.hlines(1.53, 3, 30, linestyles="--", color="C2"))
    ax.grid(zorder=-1)
    ax.text(3.5, 4.3, text, fontsize=FONT_SIZE)
    ax.set_xlabel("Context Dimension", fontsize=FONT_SIZE, labelpad=2)
    ax.set_xlim([3, 30])
    ax.set_ylim([0.3, 6.6])
    if ylabel:
        ax.set_ylabel("Cum. Disc. Ret.", fontsize=FONT_SIZE, labelpad=2)
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, pad=1)

    if legend:
        f.legend(lines, [r"\currot", r"\gradient", r"\alpgmm", r"\currot", "Default"],
                 fontsize=FONT_SIZE, loc='upper left',
                 bbox_to_anchor=(-0.005, 1.03) if single_column else (-0.02, 1.03),
                 ncol=5, columnspacing=0.6, handlelength=0.9, handletextpad=0.25)

    return lines


if __name__ == "__main__":
    base_log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "logs")
    single_column = True
    dir_name = "figures_sc" if single_column else "figures"

    f = plt.figure(figsize=(6.5, 1.4))
    axs = [f.add_axes([0.04, 0.22, 0.22, 0.61]),
           f.add_axes([0.283, 0.22, 0.22, 0.61]),
           f.add_axes([0.526, 0.22, 0.22, 0.61]),
           f.add_axes([0.77, 0.22, 0.22, 0.61])]

    high_dim_performance(["wasserstein", "gradient", "alp_gmm"], ["C0", "C1", "C9"],
                         [{"MAX": True}, {"MAX": True}, {"MAX": True}],
                         r"Using $\mathbf{c}_{\text{max}}$, $\mu_l$",
                         base_log_dir=base_log_dir, ax=axs[0], ylabel=True)
    lines = high_dim_performance(["wasserstein", "gradient", "alp_gmm", "wasserstein", "gradient"],
                                 ["C0", "C1", "C9", "C3", "C4"],
                                 [{}, {}, {}, {"BETTER_INIT": True}, {"BETTER_INIT": True}],
                                 r"Using $\mathbf{c}_{\text{min}}$, $\mu_l$",
                                 base_log_dir=base_log_dir, ax=axs[1], ylabel=False)
    high_dim_performance(["wasserstein", "gradient", "alp_gmm"], ["C3", "C4", "C9"],
                         [{"TARGET_SLICES": True, "BETTER_INIT": True},
                          {"TARGET_SLICES": True, "BETTER_INIT": True},
                          {}],
                         r"Using $\mathbf{c}_{\text{min}}$, $\mu_h$",
                         base_log_dir=base_log_dir, ax=axs[3], ylabel=False)
    high_dim_performance(["wasserstein", "gradient", "alp_gmm"], ["C0", "C1", "C9"],
                         [{"TARGET_SLICES": True, "MAX": True}, {"TARGET_SLICES": True, "MAX": True},
                          {"MAX": True}],
                         r"Using $\mathbf{c}_{\text{max}}$, $\mu_h$",
                         base_log_dir=base_log_dir, ax=axs[2], ylabel=False)

    f.legend(lines, [r"\currot", r"\gradient", r"\alpgmm", r"\currot$^*$", r"\gradient$^*$", "Default"],
             fontsize=FONT_SIZE, loc='upper left',
             bbox_to_anchor=(0.2, 1.03) if single_column else (0.25, 1.03),
             ncol=6, columnspacing=0.6, handlelength=0.9, handletextpad=0.25)

    plt.savefig(f"{dir_name}/point_mass_high_dim.pdf", dpi=300)
    plt.close(f)
