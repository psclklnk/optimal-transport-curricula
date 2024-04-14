import os
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from util import add_plot
import matplotlib.pyplot as plt
from deep_sprl.experiments import UnlockPickupExperiment
from deep_sprl.environments.unlockpickup.unlock_pickup_context import UnlockPickupContext, get_context_distance

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


def visualize_context_distances(base_log_dir: Path, method: str, learner: str, parameters=None, save_dir=None,
                                single_column=False):
    if parameters is None:
        parameters = {}

    exp = UnlockPickupExperiment(str(base_log_dir), method, learner, parameters, seed=1)
    exp_dir = Path(exp.get_log_dir()).parent
    traces = []
    for seed_dir in exp_dir.glob("seed-*"):
        trace = [], [], []
        iterations = list(sorted(seed_dir.glob("iteration-*"), key=lambda x: int(x.name.split("-")[-1])))
        for iteration_dir in iterations:
            with open(iteration_dir / "context_trace.pkl", "rb") as f:
                undisc_rews, disc_rews, contexts = pickle.load(f)
                trace[0].extend(undisc_rews)
                trace[1].extend(disc_rews)
                trace[2].extend([c.to_array() for c in contexts])
        traces.append(trace)

    # We now generate a visualization of the average context distance over time as well as the average performance
    # on the chosen contexts
    distance_path = Path("distances.npy")
    dist_fn = get_context_distance()

    if not distance_path.exists():
        distance_trace = []
        for trace in tqdm(traces):
            # Iterate over the intial contexts and compute the minimum distance
            context_trace = np.array(trace[2])
            y_pos_trace = UnlockPickupContext.get_door_y_pos(context_trace)
            target_samples = np.reshape(exp.TARGET_SAMPLES, (-1, 8))
            target_door_y_pos = UnlockPickupContext.get_door_y_pos(target_samples)
            assert target_door_y_pos.shape[0] % 4 == 0

            distances = np.zeros(context_trace.shape[0])
            for i in range(4):
                start = i * (target_door_y_pos.shape[0] // 4)
                end = (i + 1) * (target_door_y_pos.shape[0] // 4)

                assert np.all(target_door_y_pos[start:end] == i + 1)

                mask = y_pos_trace == (i + 1)
                distances[mask] = np.min(dist_fn(context_trace[mask, None], target_samples[None, start:end]), axis=-1)
            distance_trace.append(distances)

        with open(distance_path, "wb") as f:
            pickle.dump(distance_trace, f)
    else:
        with open(distance_path, "rb") as f:
            distance_trace = pickle.load(f)

    start_distance_path = Path("start_distances.npy")
    if not start_distance_path.exists():
        start_distance_trace = []
        for trace in tqdm(traces):
            # Iterate over the intial contexts and compute the minimum distance
            context_trace = np.array(trace[2])
            y_pos_trace = UnlockPickupContext.get_door_y_pos(context_trace)
            target_samples = np.reshape(exp.INIT_SAMPLES, (-1, 8))
            target_door_y_pos = UnlockPickupContext.get_door_y_pos(target_samples)
            assert target_door_y_pos.shape[0] % 4 == 0

            distances = np.zeros(context_trace.shape[0])
            for i in range(4):
                start = i * (target_door_y_pos.shape[0] // 4)
                end = (i + 1) * (target_door_y_pos.shape[0] // 4)

                assert np.all(target_door_y_pos[start:end] == i + 1)

                mask = y_pos_trace == (i + 1)
                distances[mask] = np.min(dist_fn(context_trace[mask, None], target_samples[None, start:end]), axis=-1)
            start_distance_trace.append(distances)

        with open(start_distance_path, "wb") as f:
            pickle.dump(start_distance_trace, f)
    else:
        with open(start_distance_path, "rb") as f:
            start_distance_trace = pickle.load(f)

    run_colors = ["C0" if np.mean(rewards[dist == 0][-50:]) < 0.1 else "C1"
                  for rewards, dist in zip([np.array(t[0]) for t in traces], distance_trace)]

    from scipy.signal import convolve
    from deep_sprl.environments.unlockpickup.unlock_pickup_env import visualize_contexts
    f = plt.figure(figsize=(1.6, 3.0))
    if single_column:
        axs = np.array([f.add_axes([0.28, 0.53, 0.71, 0.42]), f.add_axes([0.28, 0.06, 0.71, 0.42])])
    else:
        axs = np.array([f.add_axes([0.145, 0.11, 0.345, 0.88]), f.add_axes([0.65, 0.11, 0.345, 0.88])])
    for start_dist, contexts, rewards, color in zip(start_distance_trace, [np.array(t[2]) for t in traces],
                                                    [np.array(t[0]) for t in traces], run_colors):
        filter_width = int(0.1 * start_dist.shape[0])
        box_filter = np.ones(filter_width) / filter_width
        left_room = UnlockPickupContext.get_agent_pos(contexts)[:, 0] < 5
        door_closed = ~UnlockPickupContext.is_door_open(contexts)

        left_room = convolve(left_room, box_filter, mode="valid")
        door_closed = convolve(door_closed, box_filter, mode="valid")

        axs[0].plot(np.linspace(0, 1, left_room.shape[0]), left_room, color=color)
        axs[1].plot(np.linspace(0, 1, left_room.shape[0]), door_closed, color=color)

    axs[0].set_xticks([], [])
    axs[1].set_xticks([], [])
    if not single_column:
        axs[0].set_xlabel("Training Progress", fontsize=FONT_SIZE)
    axs[1].set_xlabel("Training Progress", fontsize=FONT_SIZE)
    axs[0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    axs[1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    axs[0].set_ylabel("Fraction Left Room", fontsize=FONT_SIZE)
    axs[1].set_ylabel("Fraction Door Closed", fontsize=FONT_SIZE)
    axs[0].grid()
    axs[1].grid()
    plt.savefig(f"{save_dir}/plr_contexts.pdf")

    n_visited = []
    target_n_visited = []
    target_success_visited = []
    for i, (reward_trace, __, context_trace) in enumerate(traces):
        visited_contexts = set()
        visited_trace = []
        target_visited_contexts = set()
        target_visited_trace = []
        target_success_contexts = set()
        target_success_visited_trace = []
        for reward, context, dist in zip(reward_trace, context_trace, distance_trace[i]):
            visited_contexts.add(UnlockPickupContext(context).to_tuple())
            if dist == 0:
                target_visited_contexts.add(UnlockPickupContext(context).to_tuple())
                if reward > 0:
                    target_success_contexts.add(UnlockPickupContext(context).to_tuple())
            visited_trace.append(len(visited_contexts))
            target_visited_trace.append(len(target_visited_contexts))
            target_success_visited_trace.append(len(target_success_contexts))
        n_visited.append(np.array(visited_trace))
        target_n_visited.append(np.array(target_visited_trace))
        target_success_visited.append(np.array(target_success_visited_trace))

    f = plt.figure(figsize=(3.2, 3.0))
    axs = np.array([[f.add_axes([0.145, 0.53, 0.345, 0.42]), f.add_axes([0.645, 0.53, 0.345, 0.42])],
                    [f.add_axes([0.145, 0.06, 0.345, 0.42]), f.add_axes([0.645, 0.06, 0.345, 0.42])]])
    for i, (dist, n_vis, tn_vis, tn_suc, rewards, color) in enumerate(zip(distance_trace, n_visited, target_n_visited,
                                                                          target_success_visited,
                                                                          [t[0] for t in traces],
                                                                          run_colors)):
        # We filter the data with a box filter of 5% of the width
        filter_width = int(0.1 * dist.shape[0])
        box_filter = np.ones(filter_width) / filter_width
        mu_rewards = convolve(rewards * (dist == 0.), box_filter, mode="valid")
        dist = convolve(dist == 0., box_filter, mode="valid")
        mu_rewards = mu_rewards / dist
        n_vis = convolve(n_vis, box_filter, mode="valid")
        tn_vis = convolve(tn_vis, box_filter, mode="valid")
        rewards = convolve(rewards, box_filter, mode="valid")

        axs[0, 0].plot(np.linspace(0, 1, dist.shape[0]), dist, color=color)
        axs[0, 0].hlines(np.prod(exp.TARGET_SAMPLES.shape[:-1]) / exp.ALL_CONTEXTS.shape[0], 0, 1,
                         color="red", linestyle="--")
        axs[0, 1].plot(np.linspace(0, 1, n_vis.shape[0]), n_vis, color=color)
        axs[0, 1].plot(np.linspace(0, 1, tn_vis.shape[0]), tn_vis, color=color, linestyle="--")
        axs[0, 1].plot(np.linspace(0, 1, tn_suc.shape[0]), tn_suc, color=color, linestyle=":")
        axs[1, 0].plot(np.linspace(0, 1, rewards.shape[0]), rewards, color=color)
        axs[1, 1].plot(np.linspace(0, 1, mu_rewards.shape[0]), mu_rewards, color=color)

    axs[0, 0].set_ylabel("Fraction $\mu(c) {>} 0$", fontsize=FONT_SIZE)
    axs[0, 1].set_ylabel(r"\# Visited Contexts", fontsize=FONT_SIZE)
    axs[1, 0].set_ylabel(r"Ep. Ret. on $p(c)$", fontsize=FONT_SIZE)
    axs[1, 1].set_ylabel(r"Ep. Ret. on $\mu(c){>}0$", fontsize=FONT_SIZE)
    axs[0, 0].set_ylim((0, axs[0, 0].get_ylim()[1]))
    # axs[0, 1].hlines(-np.log(1 / np.prod(exp.TARGET_SAMPLES.shape[:-1])), 0, 1, linestyle="--", color="red")
    axs[1, 0].set_ylim((0, axs[1, 0].get_ylim()[1]))
    axs[1, 1].set_ylim(axs[1, 0].get_ylim())

    axs[0, 0].set_xticks([], [])
    axs[0, 1].set_xticks([], [])
    axs[1, 0].set_xticks([], [])
    axs[1, 1].set_xticks([], [])
    axs[1, 0].set_xlabel("Training Progress", fontsize=FONT_SIZE)
    axs[1, 1].set_xlabel("Training Progress", fontsize=FONT_SIZE)

    axs[0, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    axs[0, 1].ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
    axs[0, 1].yaxis.offsetText.set_fontsize(TICK_SIZE)
    # axs[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))
    # axs[1, 1].yaxis.offsetText.set_fontsize(TICK_SIZE)

    axs[0, 0].grid()
    axs[0, 1].grid()
    axs[1, 0].grid()
    axs[1, 1].grid()

    if save_dir is None:
        plt.show()
    else:
        plt.savefig(f"{save_dir}/plr_unlock_pickup_investigation.pdf")
        plt.close(f)


def performance_plot(base_log_dir, learners, methods, labels, colors=None, ax=None,
                     figsize=(3.2, 1.2), handlelength=0.42, handletextpad=0.2,
                     axis_extent=[0.15, 0.23, 0.84, 0.57], legend_offset=0.01, ncol=3):
    if ax is None:
        f = plt.figure(figsize=figsize)
        ax = plt.Axes(f, axis_extent)
        f.add_axes(ax)
    else:
        f = plt.gcf()

    if colors is None:
        colors = [f"C{i}" for i in range(len(learners))]

    lines = []
    for learner, method, color in zip(learners, methods, colors):
        exp = UnlockPickupExperiment(base_log_dir, method, learner, {}, seed=0)
        log_dir = os.path.dirname(exp.get_log_dir())
        lines.append(add_plot(log_dir, ax, color))

    ax.set_xlim(0, 245)
    ax.set_xticks([0, 500000 / UnlockPickupExperiment.STEPS_PER_ITER, 1000000 / UnlockPickupExperiment.STEPS_PER_ITER,
                   1500000 / UnlockPickupExperiment.STEPS_PER_ITER, 2000000 / UnlockPickupExperiment.STEPS_PER_ITER])
    ax.set_xticklabels(["$0$", "$0.5$M", "$1$M", "$1.5$M", "$2$M"])

    ax.set_ylabel(r"Cum. Disc. Ret.", fontsize=FONT_SIZE, labelpad=2., y=0.42)
    ax.set_xlabel(r"Step", fontsize=FONT_SIZE, labelpad=1.)
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax.tick_params(axis='x', which='major', pad=2)
    ax.grid()

    f.legend(lines, labels,
             fontsize=FONT_SIZE, loc='upper left', bbox_to_anchor=(legend_offset, 1.03), ncol=ncol, columnspacing=0.3,
             handlelength=handlelength, handletextpad=handletextpad)
    return f, ax


def environment_visualization(save_path, single_column: bool = False):
    contexts = [np.array([1, 1, 2, 8, 3, 4, 2, 0], dtype=np.int64),
                np.array([4, 2, 4, 9, 1, 4, 2, 0], dtype=np.int64),
                np.array([4, 3, 3, 7, 2, 4, 3, 1], dtype=np.int64),
                np.array([6, 3, 3, 7, 3, 4, 3, 1], dtype=np.int64)]

    exp = UnlockPickupExperiment("../logs", "default", "dqn", {}, seed=1)
    exp.eval_env.env.render_mode = "rgb_array"

    f = plt.figure(figsize=(2.9, 1.7) if single_column else (3.0, 1.7))
    for i, context in enumerate(contexts):
        ax = plt.Axes(f, [0.005 + (i % 2) * 0.5, 0.005 + (1 - (i // 2)) * 0.5, 0.49, 0.49])
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        f.add_axes(ax)
        exp.eval_env.env.context = UnlockPickupContext(context)
        exp.eval_env.env.reset()
        ax.imshow(exp.eval_env.env.render())

    plt.savefig(save_path, dpi=300)
    plt.close(f)


if __name__ == "__main__":
    single_column = True
    dir_name = "figures_sc" if single_column else "figures"
    visualize_context_distances(Path("../logs"), "plr", "dqn",
                                {}, save_dir=dir_name, single_column=single_column)

    environment_visualization(f"{dir_name}/unlockpickup_env.pdf", single_column=single_column)

    f, ax = performance_plot("../logs", learners=["dqn", "dqn", "dqn", "dqn", "dqn", "dqn", "dqn"],
                             methods=["default", "random", "wasserstein", "gradient", "acl", "plr", "vds"],
                             labels=[r"Default", r"Random", r"\currot", r"\gradient", r"\acl", r"\plr", r"\vds"],
                             colors=["C2", "C3", "C0", "C1", "C4", "C5", "C6"],
                             figsize=(2.75, 1.7) if single_column else (3.2, 1.2),
                             handlelength=0.42 if single_column else 0.42,
                             handletextpad=0.2 if single_column else 0.2,
                             axis_extent=[0.145, 0.16, 0.845, 0.62] if single_column else [0.15, 0.23, 0.84, 0.57],
                             legend_offset=0.22 if single_column else 0.01,
                             ncol=4 if single_column else 6)
    plt.savefig(f"{dir_name}/unlockpickup_performance.pdf")
    plt.close(f)
