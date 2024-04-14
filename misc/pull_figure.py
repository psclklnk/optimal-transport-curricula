import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import linear_sum_assignment
from deep_sprl.teachers.util import NadarayaWatsonPy
from pathlib import Path

FONT_SIZE = 8
TICK_SIZE = 6

plt.rc('text.latex', preamble=r'\usepackage{amsmath}'
                              r'\usepackage{amssymb}'
                              r'\usepackage{amsthm}'
                              r'\usepackage{contour}'
                              r'\usepackage{amsfonts}'
                              r'\newcommand{\currot}{\textsc{currot}}'
                              r'\newcommand{\gradient}{\textsc{gradient}}'
                              r'\newcommand{\sprl}{\textsc{sprl}}'
                              r'\DeclareMathOperator*{\argmin}{arg\,min}')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Helvetica"],
})


def get_interpolator():
    # We realize the agent performance using an interpolation between points
    contexts = np.array([[0, 0], [0.1, 0], [0, 0],
                         [0, 0.1], [0, 0.2], [0, 0.25], [0, 0.3],
                         [0.55, 0.41], [0.38, 0.38],
                         [0.23, 0.02],
                         [0.23, 0.35],
                         [0.02, 0.23],
                         [0.35, 0.23],
                         [1., 1.],
                         [0., 1.], [0.1, 1.], [0.2, 1.], [0.3, 1.],
                         [1., 0.],
                         [0.4, 0.88],
                         [0.17, 0.54],
                         [0.4, 0.47],
                         [0.03, 0.5], [0.15, 0.49], [0.3, 0.6],
                         [0.7, 0.3], [0.62, 0.8],
                         [0., 0.], [0.1, 0.], [0.2, 0.05], [0.3, 0.05]])
    performances = np.array([1., 1., 1.,
                             0.95, 0.9, 0.5, 0.3,
                             0.8, 0.8,
                             0.98,
                             0.98,
                             0.98,
                             0.98,
                             0.05,
                             0., 0., 0., 0.,
                             0.,
                             0.07,
                             0.07,
                             0.8,
                             0.12, 0.05, 0.3,
                             0.8, 0.13,
                             1, 0.95, 0.87, 0.8])

    return NadarayaWatsonPy(contexts, performances, 0.2)


def create_agent_performance():
    interpolator = get_interpolator()

    X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    Z = interpolator.predict_individual(np.stack((X, Y), axis=-1))

    return X, Y, Z


def gmm_likelihood(samples, hard=False):
    np.random.seed(0)
    add_samples = []
    for i in range(0, 10 * samples.shape[0]):
        start_idx, end_idx = np.random.permutation(samples.shape[0])[:2]
        alpha = np.random.uniform(0, 1)
        add_samples.append(alpha * samples[start_idx] + (1 - alpha) * samples[end_idx])

    full_samples = np.concatenate((samples, np.array(add_samples)), axis=0)

    X, Y = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    log_pdfs = []
    for full_sample in full_samples:
        if hard:
            tmp = np.where(np.linalg.norm(np.stack((X, Y), axis=-1) - full_sample[None, None, :], axis=-1) <= 0.025,
                           1., 1e-3)
            log_pdfs.append(np.log(tmp / np.sum(tmp)))
        else:
            log_pdfs.append(scipy.stats.multivariate_normal.logpdf(np.stack((X, Y), axis=-1),
                                                                   full_sample, (0.05 ** 2) * np.eye(2)))
    return X, Y, np.mean(np.exp(log_pdfs), axis=0)


def gmm_plot(ax, samples, color, n_sample_rounds=10):
    # Sample in the convex hull of the points
    if n_sample_rounds == 0:
        full_samples = samples
    else:
        np.random.seed(0)
        add_samples = []
        while len(add_samples) < n_sample_rounds * samples.shape[0]:
            start_idx, end_idx = np.random.permutation(samples.shape[0])[:2]
            if np.linalg.norm(samples[start_idx] - samples[end_idx]) < 0.45:
                alpha = np.random.uniform(0, 1)
                add_samples.append(alpha * samples[start_idx] + (1 - alpha) * samples[end_idx])

        full_samples = np.concatenate((samples, np.array(add_samples)), axis=0)

    X, Y = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    dists = np.sum(np.square(np.stack((X, Y), axis=-1)[:, :, None, :] - full_samples[None, None, :, ]), axis=-1)
    Z = np.exp(-(0.5 / (0.05 ** 2)) * np.min(dists, axis=-1))

    # We render this as an image
    image = np.zeros(Z.shape + (4,))
    image[..., :-1] = color[:3]
    image[Z > 0.5, -1] = 0.5

    ax.contour(X, Y, Z, levels=[0.3], linestyles="--", colors=[color])


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) / 255 for i in range(0, lv, lv // 3))


def gradient_plot(ax, init_samples, target_samples, delta):
    # Compute the barycenter with fulfills delta
    rows, cols = linear_sum_assignment(np.linalg.norm(init_samples[:, None] - target_samples[None, :], axis=-1) ** 2)
    interpolator = get_interpolator()
    alphas = np.linspace(0, 1, 100)
    performances = np.zeros_like(alphas)
    empirical_distributions = []
    for i, alpha in enumerate(alphas):
        empirical_distribution = (1 - alpha) * init_samples + alpha * target_samples[cols]
        performances[i] = np.mean(interpolator.predict_individual(empirical_distribution))
        empirical_distributions.append(empirical_distribution)

    opt_idx = np.argmin(np.square(performances - delta))
    cmap = matplotlib.cm.get_cmap("inferno")

    # Plot the targets and initial samples
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    gmm_plot(ax, target_samples, hex_to_rgb(colors[2]))
    gmm_plot(ax, init_samples, hex_to_rgb(colors[0]))
    ax.scatter(target_samples[:, 0], target_samples[:, 1], color="C2", zorder=2, edgecolor="black")
    ax.scatter(init_samples[:, 0], init_samples[:, 1], color="C0", zorder=2, edgecolor="black")
    for idx in [25, opt_idx, 80]:
        if idx == opt_idx:
            opaqueness = 1.
        else:
            opaqueness = 0.5

        # Additionally, we do a GMM plot around the samples
        gmm_plot(ax, empirical_distributions[idx], cmap(performances[idx]))
        ax.scatter(empirical_distributions[idx][:, 0], empirical_distributions[idx][:, 1],
                   color=cmap(performances[idx]), zorder=2, edgecolor="black", alpha=opaqueness)

    # Plot the geodesics of the assignment
    ax.plot(np.stack((init_samples[:, 0], target_samples[cols, 0])),
            np.stack((init_samples[:, 1], target_samples[cols, 1])), linestyle="--", color="yellow", zorder=1)


def sprl_plot(ax, init_samples, target_samples, delta):
    X, Y, init_likelihood = gmm_likelihood(init_samples)
    __, __, target_likelihood = gmm_likelihood(target_samples, hard=True)

    interpolator = get_interpolator()
    performances = interpolator.predict_individual(np.stack((X, Y), axis=-1))

    # ax.contour(X, Y, init_likelihood)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax.scatter(init_samples[:, 0], init_samples[:, 1], color="C0", zorder=3, edgecolor="black")
    gmm_plot(ax, init_samples, hex_to_rgb(colors[0]))
    ax.scatter(target_samples[:, 0], target_samples[:, 1], color="C2", zorder=3, edgecolor="black")
    gmm_plot(ax, target_samples, hex_to_rgb(colors[2]))

    # ax.contour(X, Y, target_likelihood)

    def logsumexp(x):
        x_max = np.max(x)
        return np.log(np.sum(np.exp(x - x_max))) + x_max

    def sprl_dist(target_ll, perfs, delt):
        def expected_perf(target_ll, perfs, eta):
            unnormed_dist = target_ll + eta * perfs
            normed_dist = unnormed_dist - logsumexp(unnormed_dist)
            return np.sum(np.exp(normed_dist) * perfs)

        from scipy.optimize import brentq
        eta_opt = brentq(lambda eta: expected_perf(target_ll, perfs, eta) - delt, 0., 1000.)
        unnormed_dist = target_ll + eta_opt * perfs
        return np.exp(unnormed_dist - logsumexp(unnormed_dist))

    cmap = matplotlib.cm.get_cmap("inferno")
    traj = [np.mean(init_samples, axis=0)]
    for tmp, seed in zip([0.82, 0.75, delta], [0, 4, 5]):
        dist = sprl_dist(np.log(target_likelihood), performances, tmp)
        flat_cum_dist = np.cumsum(np.reshape(dist, -1))
        traj.append(np.sum(dist[..., None] * np.stack((X, Y), axis=-1), axis=(0, 1)))

        np.random.seed(seed)
        flat_sample_idxs = np.argmin(np.random.uniform(0, 1., size=5)[:, None] >= flat_cum_dist[None, :], axis=-1)
        sample_idxs = np.unravel_index(flat_sample_idxs, X.shape)
        ax.scatter(X[sample_idxs], Y[sample_idxs], color=cmap(tmp), zorder=3, edgecolor="black",
                   alpha=1. if tmp == delta else 0.5)
        gmm_plot(ax, np.stack((X[sample_idxs], Y[sample_idxs]), axis=-1), cmap(tmp))
    traj.append(np.mean(target_samples, axis=0))

    traj = np.array(traj)

    from scipy import interpolate
    f, u = interpolate.splprep([traj[:, 0], traj[:, 1]], s=0)
    # create interpolated lists of points
    xint, yint = interpolate.splev(np.linspace(0, 1, 100), f)

    traj = np.stack((xint, yint), axis=-1)
    ax.plot(traj[:, 0], traj[:, 1], color="yellow", linewidth=4, zorder=2)
    der = traj[-1] - traj[-2]
    der = der / np.linalg.norm(der)
    ax.arrow(traj[-1, 0] - 0.04 * der[0], traj[-1, 1] - 0.04 * der[1],
             0.01 * der[0], 0.01 * der[1], width=0.001, color="yellow",
             head_width=0.15, head_length=0.075, overhang=0.3)
    ax.plot([0.6625, 0.5625], [0.5, 0.545], color="yellow", zorder=2, linewidth=2)
    ax.plot([0.6325, 0.5325], [0.45, 0.495], color="yellow", zorder=2, linewidth=2)


def currot_plot(ax, init_samples, target_samples, delta, single_column: bool = False):
    rows, cols = linear_sum_assignment(np.linalg.norm(init_samples[:, None] - target_samples[None, :], axis=-1) ** 2)
    interpolator = get_interpolator()

    X, Y = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    search_points = np.stack((X, Y), axis=-1)
    perfs = interpolator.predict_individual(search_points)

    particles = []
    deltas = [0.82, 0.75, delta]
    all_deltas = np.linspace(np.min(interpolator.predict_individual(init_samples)),
                             np.min(interpolator.predict_individual(target_samples)), 50)
    all_particles = []
    for tmp in all_deltas:
        idxs_x, idxs_y = np.where(perfs > tmp)

        valid_positions = search_points[idxs_x, idxs_y]
        new_particles_idxs = np.argmin(
            np.sum(np.square(target_samples[cols, None, :] - valid_positions[None]), axis=-1),
            axis=-1)
        all_particles.append(valid_positions[new_particles_idxs])

    for tmp in deltas:
        particles.append(all_particles[np.argmin(np.square(tmp - all_deltas))])

    cmap = matplotlib.cm.get_cmap("inferno")

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    gmm_plot(ax, init_samples, hex_to_rgb(colors[0]))
    ax.scatter(init_samples[:, 0], init_samples[:, 1], color="C0", zorder=2, edgecolor="black")
    for p, d in zip(particles, deltas):
        gmm_plot(ax, p, cmap(d))
        ax.scatter(p[:, 0], p[:, 1], color=cmap(d), zorder=2, edgecolor="black",
                   alpha=1. if d == delta else 0.5)
    gmm_plot(ax, target_samples, hex_to_rgb(colors[2]))
    ax.scatter(target_samples[:, 0], target_samples[:, 1], color="C2", zorder=2, edgecolor="black")

    mu_traj = np.mean(np.array([init_samples] + all_particles), axis=1)
    from scipy import interpolate
    f, u = interpolate.splprep([mu_traj[:, 0], mu_traj[:, 1]], s=1e-3)
    # create interpolated lists of points
    xint, yint = interpolate.splev(np.linspace(0, 1, 100), f)

    mu_traj = np.stack((xint, yint), axis=-1)

    der = mu_traj[-1] - mu_traj[-2]
    der = der / np.linalg.norm(der)
    ax.arrow(mu_traj[-1, 0] - 0.04 * der[0], mu_traj[-1, 1] - 0.04 * der[1],
             0.01 * der[0], 0.01 * der[1], width=0.001, color="yellow",
             head_width=0.15, head_length=0.075, overhang=0.3)
    ax.plot(mu_traj[:, 0], mu_traj[:, 1], color="yellow", linewidth=4, zorder=1)

    ax.text(0.15 if single_column else 0.0, 0.94, r"\textbf{Target Tasks}", color=colors[2], fontsize=FONT_SIZE)
    ax.text(0.0, 0.12, r"\textbf{Initial Tasks}", color=colors[0], fontsize=FONT_SIZE)

    # ax.plot(np.stack((init_samples[:, 0], particles[0][:, 0])),
    #         np.stack((init_samples[:, 1], particles[0][:, 1])), linestyle="--", color="yellow", zorder=1)
    # for p1, p2 in zip(particles[:-1], particles[1:]):
    #     ax.plot(np.stack((p1[:, 0], p2[:, 0])), np.stack((p1[:, 1], p2[:, 1])), linestyle="--", color="yellow",
    #             zorder=1)
    # ax.plot(np.stack((particles[-1][:, 0], target_samples[cols, 0])),
    #         np.stack((particles[-1][:, 1], target_samples[cols, 1])), linestyle="--", color="yellow", zorder=1)


def main(path=None, single_column: bool = False):
    # Boundaries are in [0, 1] x [0, 1]
    # Define target values
    delta = 0.6
    target_samples = np.array([[0.82, 0.75], [0.92, 0.8], [0.6, 0.95],
                               [0.79, 0.91], [0.93, 0.56]])

    np.random.seed(0)
    interpolator = get_interpolator()
    X, Y = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    idxs1, idxs2 = np.where(interpolator.predict_individual(np.stack((X, Y), axis=-1)) > 0.9)
    np.random.seed(0)
    sel = np.random.randint(0, idxs1.shape[0], size=5)
    init_samples = np.stack((X, Y), axis=-1)[idxs1[sel], idxs2[sel]]

    # Compute the values
    f = plt.figure(figsize=(5.0, 2.0) if single_column else (3.5, 2.0))
    ax = f.add_axes([0.4675, 0.06, 0.425, 0.88])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_title(r"\sprl", fontsize=FONT_SIZE, pad=2.)
    ax.set_xlabel(r"Task Parameter $c_1$", fontsize=FONT_SIZE, labelpad=1)

    ax2 = f.add_axes([0.035, 0.06, 0.425, 0.88])
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_xticks([], [])
    ax2.set_yticks([], [])
    ax2.set_title(r"\currot", fontsize=FONT_SIZE, pad=2.)
    ax2.set_xlabel(r"Task Parameter $c_1$", fontsize=FONT_SIZE, labelpad=1)
    ax2.set_ylabel(r"Task Parameter $c_2$", fontsize=FONT_SIZE, labelpad=1)

    cax = f.add_axes([0.90, 0.06, 0.03, 0.88])
    X, Y, Z = create_agent_performance()
    ax.contourf(X, Y, Z, 10, cmap="inferno")
    # ax.contour(X, Y, Z, levels=[delta], linestyles="--", colors=["red"])
    ax2.contourf(X, Y, Z, 10, cmap="inferno")
    # ax2.contour(X, Y, Z, levels=[delta], linestyles="--", colors=["red"])

    # gradient_plot(ax, init_samples, target_samples, delta)
    sprl_plot(ax, init_samples, target_samples, delta)
    currot_plot(ax2, init_samples, target_samples, delta, single_column=single_column)

    cmap = matplotlib.cm.get_cmap("inferno")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cb1 = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cax.set_yticks([0., 1.])
    cax.set_yticklabels(["low", "high"])  # , [r"$0.0$", r"$0.2$", r"$0.4$", r"$\delta$", r"$0.8$", r"$1.0$"])
    cax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, pad=0)
    cax.set_ylabel("Agent Performance", fontsize=FONT_SIZE, labelpad=-15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=300)


if __name__ == "__main__":
    single_column = True
    dir_name = "figures_sc" if single_column else "figures"
    Path(dir_name).mkdir(exist_ok=True)
    main(f"{dir_name}/pull_figure.pdf", single_column=True)
