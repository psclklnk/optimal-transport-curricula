import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rc('text.latex', preamble=r'\usepackage{amsmath}'
                              r'\newcommand{\currot}{\textsc{currot}}'
                              r'\newcommand{\gradient}{\textsc{gradient}}'
                              r'\newcommand{\sprl}{\textsc{sprl}}'
                              r'\newcommand{\alpgmm}{\textsc{alp-gmm}}'
                              r'\newcommand{\goalgan}{\textsc{goalgan}}'
                              r'\newcommand{\acl}{\textsc{acl}}'
                              r'\newcommand{\vds}{\textsc{vds}}'
                              r'\newcommand{\plr}{\textsc{plr}}'
                              r'\newcommand{\her}{\textsc{her}+\textsc{sac}}'
                              r'\newcommand{\sac}{\textsc{sac}}')
plt.rcParams.update({
    "text.usetex": True,
    "font.serif": "Helvetica",
    "font.family": "serif"
})

FONT_SIZE = 8
TICK_SIZE = 6

def main(log_dir, her_log_dir):
    base_dir = Path(__file__).resolve().parent
    log_dir = base_dir / log_dir
    her_log_dir = base_dir / her_log_dir
    method = "sac"
    lines = []
    labels = ["Oracle", r"Oracle (\her)", "Random", r"Random (\her)", "Random-LT", r"Random-LT (\her)",
              r"\currot (\her)", r"\currot", r"\gradient (\her)", r"\gradient"]

    fig = plt.figure(figsize=(3.1, 1.7))
    ax = plt.Axes(fig, [0.12, 0.18, 0.855, 0.55])
    fig.add_axes(ax)

    for path, color, style in zip([f"{log_dir}/sparse_goal_reaching/default/{method}",
                                   f"{her_log_dir}/sparse_goal_reaching_her_2/default/{method}",

                                   f"{log_dir}/sparse_goal_reaching/random/{method}",
                                   f"{her_log_dir}/sparse_goal_reaching_her_2_offline/random/{method}",

                                   f"{her_log_dir}/sparse_goal_reaching_target_slice/random/{method}",
                                   f"{her_log_dir}/sparse_goal_reaching_her_2_target_slice/random/{method}",

                                   f"{log_dir}/sparse_goal_reaching/wasserstein/{method}_DELTA=0.8_ENT_LB=0.0_METRIC_EPS=1.2",
                                   f"{her_log_dir}/sparse_goal_reaching_her_2_offline/wasserstein/{method}_DELTA=0.8_ENT_LB=0.0_METRIC_EPS=1.2",

                                   f"{log_dir}/sparse_goal_reaching/gradient/{method}_GRADIENT_DELTA=0.6_GRADIENT_ENT=0.01_GRADIENT_EPS=0.05",
                                   f"{her_log_dir}/sparse_goal_reaching_her_2_offline/gradient/{method}_GRADIENT_DELTA=0.6_GRADIENT_ENT=0.01_GRADIENT_EPS=0.05"],
                                  ["black", "black", "C3", "C3", "C5", "C5", "C0", "C0", "C1", "C1"],
                                  ["dotted", "-", "dotted", "-", "dotted", "-", "dotted", "-", "dotted", "-"]):
        log_dir = Path(path).resolve()
        performances = []
        for performance_file in log_dir.glob("seed-*/performance.pkl"):
            with open(performance_file, "rb") as f:
                performances.append(pickle.load(f))
        print(len(performances))
        performances = np.stack(performances, axis=-1)

        l, = ax.plot(np.arange(performances.shape[0]), np.mean(performances, axis=-1), color=color, linestyle=style,
                     linewidth=2)
        se = np.std(performances, axis=-1) / np.sqrt(performances.shape[1])
        ax.fill_between(np.arange(performances.shape[0]),
                        np.mean(performances, axis=-1) - 2 * se, np.mean(performances, axis=-1) + 2 * se,
                        alpha=0.3, color=color, linestyle=style)
        lines.append(l)

    ax.set_ylim(0, 1)
    ax.set_xticks([0, 10, 20, 30, 40], ["0", "1M", "2M", "3M", "4M"])
    ax.set_xlim([0, 40])
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_SIZE)
    ax.set_xlabel(r"Step", fontsize=FONT_SIZE, labelpad=2.)
    ax.set_ylabel(r"Success Rate", fontsize=FONT_SIZE, labelpad=2.)
    ax.set_xlabel(r"Step", fontsize=FONT_SIZE, labelpad=2.)
    ax.grid()
    ax.legend(handles=[lines[1], lines[3], lines[5], lines[6], lines[8],
                       Line2D([0], [0], color='grey', alpha=0, linestyle="-", label='HER'),
                       Line2D([0], [0], color='grey', linestyle="-", label='HER'),
                       Line2D([0], [0], color='grey', linestyle="dotted", label='Default')],
              labels=[labels[0], labels[2], labels[4], labels[7], labels[9], "", r"\her", r"\sac"], fontsize=FONT_SIZE,
              loc='upper left', bbox_to_anchor=(-0.145, 1.505), ncol=4, columnspacing=0.6, handlelength=1.3,
              handletextpad=0.25)

    plt.savefig("her_ablation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main("../logs", "../logs_her")
