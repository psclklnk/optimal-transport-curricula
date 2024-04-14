import os
import torch
import pickle
import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize, NonlinearConstraint, linprog
from deep_sprl.teachers.spl.exact_currot import generate_wasserstein_constraint, ExactGradient, ExactCurrOT
from deep_sprl.experiments.emaze_experiment import PerformanceOracle

FONT_SIZE = 8
TICK_SIZE = 6

plt.rc('text.latex', preamble=r'\usepackage{amsmath}'
                              r'\usepackage{amssymb}'
                              r'\usepackage{amsthm}'
                              r'\usepackage{amsfonts}'
                              r'\newcommand{\currot}{\textsc{currot}}'
                              r'\newcommand{\gradient}{\textsc{gradient}}'
                              r'\DeclareMathOperator*{\argmin}{arg\,min}')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Helvetica"],
})


def gauss_log_pdf(xs, mean, var):
    logs = -0.5 * (np.square(xs - mean) / var)
    return logs - logsumexp(logs)


def compute_projection(xs, mu, values, delta):
    def objective(params):
        log_pdf = gauss_log_pdf(xs, params[0], params[1])
        pdf = np.exp(log_pdf)
        return np.sum(pdf * (log_pdf - np.log(mu)))

    def perf_con(params):
        log_pdf = gauss_log_pdf(xs, params[0], params[1])
        return np.sum(np.exp(log_pdf) * values)

    con = NonlinearConstraint(perf_con, delta, np.inf)
    res = minimize(objective, np.array([0.5, 0.25]), method="trust-constr", constraints=[con],
                   bounds=[(0., 1.), (1e-10, 50.)])
    return res.x


class ValueFunction(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            return np.mean(self.net(torch.from_numpy(x).float()).detach().numpy().astype(x.dtype))
        else:
            return self.net(x)

    def predict_individual(self, samples, with_gradient=False):
        samples_torch = torch.from_numpy(samples).float()
        if with_gradient:
            samples_torch = samples_torch.requires_grad_(True)

        perf = self.net(samples_torch)
        if with_gradient:
            grad = torch.autograd.grad(perf, samples_torch, grad_outputs=torch.ones_like(perf))[0]
            return np.squeeze(perf.detach().numpy().astype(samples.dtype)), \
                grad.detach().numpy().astype(samples.dtype)
        else:
            return np.squeeze(perf.detach().numpy().astype(samples.dtype))


def train_vf(vf, data_x, data_y, max_iter=10000):
    data_x = torch.from_numpy(data_x).float()[:, None]
    data_y = torch.from_numpy(data_y).float()

    optimizer = torch.optim.Adam(vf.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    rel_err = torch.mean(torch.abs(data_y - torch.squeeze(vf.forward(data_x))) / torch.clamp_min(torch.abs(data_y), 1.))
    count = 0
    while rel_err > 0.05 and count < max_iter:
        loss = loss_fn(torch.squeeze(vf.forward(data_x)), data_y)
        print("Relative Error: %.3e" % rel_err)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rel_err = torch.mean(
            torch.abs(data_y - torch.squeeze(vf.forward(data_x))) / torch.clamp_min(torch.abs(data_y), 1.))
        count += 1


def get_vf():
    torch.random.manual_seed(0)
    vf = ValueFunction()
    np.random.seed(1)
    train_x = np.random.uniform(0, 1, 10)
    train_y = np.minimum(1., 1 / (10 * train_x))
    train_vf(vf, train_x, train_y)
    return vf


def logsumexp(x, axis=None):
    x_max = np.max(x, axis=axis)
    return np.log(np.sum(np.exp(x - x_max), axis=axis)) + x_max


def expected_value(target_ll, values, eta):
    interp_ll = target_ll + eta * values
    interp_ll -= logsumexp(interp_ll)
    return np.sum(np.exp(interp_ll) * values)


def wasserstein_spdl(xs, target_ll, values, delta):
    interpolator = WassersteinSPDL(delta, np.exp(target_ll))
    return interpolator(np.abs(xs[:, None] - xs[None, :]) ** 2, values)


def spdl(xs, target_ll, values, delta):
    # Compute the right value of eta
    if expected_value(target_ll, values, 0) >= delta:
        eta = 0.
    else:
        eta = brentq(lambda x: expected_value(target_ll, values, x) - delta, 2000., 0.)
    interp_ll = target_ll + eta * values
    interp_ll -= logsumexp(interp_ll)

    return np.exp(interp_ll)


def smooth_prob(probs: np.ndarray):
    smoothed_probs = np.zeros_like(probs)

    # Create a filter
    from scipy import signal
    b, a = signal.butter(8, 0.1)

    # Get the connected regions with non-zero support
    start_idx = 0
    end_idx = 1
    for i in range(probs.shape[0]):
        # Filter out little holes
        if np.any(probs[end_idx:end_idx + 2] > 1e-4):
            end_idx += 1
        else:
            if end_idx - start_idx > 1:
                # Process
                smoothed_probs[start_idx:end_idx] = signal.filtfilt(b, a, probs[start_idx:end_idx],
                                                                    padlen=int(0.5 * end_idx - start_idx),
                                                                    padtype="constant")
            else:
                smoothed_probs[start_idx:end_idx] = probs[start_idx:end_idx]

            start_idx = end_idx
            end_idx = end_idx + 1

    if end_idx != probs.shape[0]:
        if end_idx - start_idx > 1:
            # Process
            smoothed_probs[start_idx:end_idx] = signal.filtfilt(b, a, probs[start_idx:end_idx], method="gust")
        else:
            smoothed_probs[start_idx:end_idx] = probs[start_idx:end_idx]

    return smoothed_probs


def ep_interpolation_plot(path=None, single_column: bool = False):
    xs = np.linspace(0, 1, 300)
    deltas = [0.96, 0.8, 0.4, 0.1, 0.]

    target_ll = -20 * np.square(xs - 0.95)
    target_ll -= logsumexp(target_ll)

    scale = 0.025
    values = 0.8 / (1 + np.exp(20 * (xs - 0.3))) + 0.2 * (1 - xs)

    # Generate the different plots for the different eta
    f = plt.figure(figsize=(2.55 if single_column else 3., 1.7))
    axs = []

    cax = None if single_column else f.add_axes([0.865, 0.08, 0.03, 0.82])
    cmap = cm.get_cmap("inferno")
    labels = [
        r"$\argmin_{p} D_{\text{KL}}(p(c) \| \mu(c))\ \text{s.t.}\ \mathbb{E}_{p} \left[ J(\pi, c) \right] \geq \delta$",
        r"$\argmin_{p} \mathcal{W}_2(p(c), \mu(c))\ \text{s.t.}\ \mathbb{E}_{p} \left[ J(\pi, c) \right] \geq \delta$"]
    for row, (interpolator, color, label) in enumerate(zip([spdl, wasserstein_spdl], ["C2", "C3"], labels)):
        ax = f.add_axes([0.01, 0.08 + 0.47 * (1 - row), 0.98 if single_column else 0.845, 0.35])
        axs.append(ax)

        vl, = ax.plot(xs, values, color="C2")
        ax.fill_between(xs, 0, values, color="C2", alpha=0.7)
        ax.set_title(label, fontsize=FONT_SIZE, pad=3.5)

        for i, delta in enumerate(deltas):
            interp_prob = interpolator(xs, target_ll, values, delta)

            color = cmap(delta)

            mask = xs < 0.4
            xs1 = xs[mask]
            xs2 = xs[~mask]
            ni1 = interp_prob[mask] / scale
            ni2 = interp_prob[~mask] / scale

            pm1 = ni1 > 1e-2
            pm1[1:-1] = np.logical_or(pm1[2:], pm1[:-2])
            pm2 = ni2 > 1e-2
            pm2[1:-1] = np.logical_or(pm2[2:], pm2[:-2])
            il, = ax.plot(xs1[pm1], ni1[pm1], color=color, zorder=2 + i)
            il, = ax.plot(xs2[pm2], ni2[pm2], color=color, zorder=2 + len(deltas) - 1 - i)
            ax.fill_between(xs1[pm1], 0., ni1[pm1], color=color, alpha=0.7, zorder=2 + i)
            ax.fill_between(xs2[pm2], 0., ni2[pm2], color=color, alpha=0.7,
                            zorder=2 + len(deltas) - 1 - i)

        if row == 1:
            ax.set_xlabel(r"$c$", fontsize=FONT_SIZE, labelpad=-1)

        ax.set_xticklabels([])
        ax.grid()
        ax.set_xlim([0, 1])
        ax.set_xticks([0., 0.25, 0.5, 0.75, 1.])
        ax.set_ylim([0, 1])
        ax.set_yticks([0., 0.25, 0.5, 0.75, 1.])
        ax.set_yticklabels([])
        ax.tick_params('both', length=0, width=0, which='major')

    axs[0].tick_params('x', length=0, width=0, which='major')
    axs[1].tick_params('x', length=0, width=0, which='major')
    axs[0].tick_params('y', which="major", pad=0)
    axs[1].tick_params('y', which="major", pad=0)

    if cax is not None:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        cb1 = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
        cb1.set_label(r"$\delta$", fontsize=FONT_SIZE, labelpad=-1)
        cax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(f)


def kl_interpolation_plot2(path=None, single_column: bool = False):
    xs = np.linspace(0, 1, 1000)
    target_lls1 = []
    target_lls2 = []
    alphass = [[0., 0.45, 0.5, 0.55, 1.], [0., 0.25, 0.5, 0.75, 1.]]

    target_ll1 = np.log(np.where(xs > 0.75, 1, 1e-5))
    target_ll1 -= logsumexp(target_ll1)
    target_lls1.append(target_ll1)

    target_ll2 = np.log(np.where(xs < 0.25, 1, 1e-5))
    target_ll2 -= logsumexp(target_ll2)
    target_lls2.append(target_ll2)

    target_ll1 = -500 * np.square(xs - 0.9)
    target_ll1 -= logsumexp(target_ll1)
    target_lls1.append(target_ll1)

    target_ll2 = -500 * np.square(xs - 0.1)
    target_ll2 -= logsumexp(target_ll2)
    target_lls2.append(target_ll2)

    f = plt.figure(figsize=(2.6 if single_column else 3.1, 1.3))
    cax = None if single_column else f.add_axes([0.86, 0.1, 0.03, 0.78])
    cmap = cm.get_cmap("inferno")
    for row, (target_ll1, target_ll2, alphas) in enumerate(zip(target_lls1, target_lls2, alphass)):
        # Compute normalization constants
        linit = np.exp(target_ll1) / (xs[1] - xs[0])
        ltarget = np.exp(target_ll2) / (xs[1] - xs[0])
        min_l = min(np.min(linit), np.min(ltarget))
        max_l = max(np.max(linit), np.max(ltarget))

        # Generate the different plots for the different eta
        ax = f.add_axes([0.01, 0.1 + 0.4 * row, 0.98, 0.38])

        for i, alpha in enumerate(alphas):
            interp_ll = target_ll1 * alpha + target_ll2 * (1 - alpha)
            interp_ll -= logsumexp(interp_ll)

            linterp = np.exp(interp_ll) / (xs[1] - xs[0])
            norm_interp = (linterp - min_l) / (max_l - min_l)

            color = cmap(0.95 * (i / (len(alphas) - 1)))

            # We reverse the zorder for the different plots to have the correct overlay behavior
            if row == 1:
                ax.plot(xs, norm_interp, color=color, zorder=2 + i)
                ax.fill_between(xs, -1, norm_interp, color=color, alpha=0.8, edgecolor=color, linewidth=2, zorder=2 + i)
            else:
                mask = xs < 0.5
                xs1 = xs[mask]
                ni1 = norm_interp[mask]
                xs2 = xs[~mask]
                ni2 = norm_interp[~mask]
                ax.plot(xs1[ni1 > 1e-2], ni1[ni1 > 1e-2], color=color, zorder=2 + i)
                ax.fill_between(xs1[ni1 > 1e-2], -1, ni1[ni1 > 1e-2], color=color, alpha=0.8,
                                edgecolor=color, linewidth=2, zorder=2 + i)
                ax.plot(xs2[ni2 > 1e-2], ni2[ni2 > 1e-2], color=color, zorder=len(alphas) - i + 2)
                ax.fill_between(xs2[ni2 > 1e-2], -1, ni2[ni2 > 1e-2], color=color, alpha=0.8,
                                edgecolor=color, linewidth=2, zorder=len(alphas) - i + 2)

            ax.set_xlim([0, 1])
            ax.set_xticks([0., 0.25, 0.5, 0.75, 1.])
            ax.set_ylim([0, 1])
            ax.set_yticks([0., 0.25, 0.5, 0.75, 1.])

        if row == 1:
            ax.set_title(r"$p_1(c)^{\alpha} p_2(c)^{1-\alpha}$", fontsize=FONT_SIZE, pad=3)

        if row == 0:
            ax.set_xlabel(r"$c$", fontsize=FONT_SIZE, labelpad=-1)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params('both', length=0, width=0, which='major')
        ax.grid()

    if cax is not None:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        cb1 = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
        cb1.set_label(r"$\alpha$", fontsize=FONT_SIZE, labelpad=-1)
        cax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def currot_gradient_interpolation(path=None):
    xs = np.linspace(0, 1, 300)
    deltas = [0.95, 0.85, 0.3, 0.05, 0.]

    init_prob = np.zeros_like(xs)
    init_prob[0] = 1

    target_ll = -20 * np.square(xs - 0.95)
    target_ll -= logsumexp(target_ll)

    scale = 0.025
    values = 0.8 / (1 + np.exp(20 * (xs - 0.3))) + 0.2 * (1 - xs)

    # We first compute the different interpolations
    if os.path.exists("cg_interpolations.pkl"):
        with open("cg_interpolations.pkl", "rb") as f:
            gradient_interps, currot_interps = pickle.load(f)
    else:
        gradient = ExactGradient(init_prob, lambda: np.abs(xs[:, None] - xs[None, :]),
                                 PerformanceOracle(lambda: values), 0., np.exp(target_ll), ent_eps=0.)
        currot = ExactCurrOT(init_prob, lambda: np.abs(xs[:, None] - xs[None, :]),
                             PerformanceOracle(lambda: values), 0., np.exp(target_ll))

        gradient_interps = []
        currot_interps = []
        for i, delta in enumerate(deltas):
            gradient.delta = delta
            currot.barycenter_computer.delta = delta

            gradient.update_distribution(xs, values)
            gradient_interps.append(gradient.current_distribution)

            currot.update_distribution(xs, values)
            currot_interps.append(currot.current_distribution)

        with open("cg_interpolations.pkl", "wb") as f:
            pickle.dump((gradient_interps, currot_interps), f)

    # We smooth the gradient interpolations since the LP barycenters tend to have high-frequency chirps
    gradient_interps = [smooth_prob(gi) for gi in gradient_interps]

    # Generate the different plots for the different eta
    f = plt.figure(figsize=(3, 1.7))
    axs = []

    cax = f.add_axes([0.865, 0.08, 0.03, 0.82])
    cmap = cm.get_cmap("inferno")

    labels = [r"\gradient", r"\currot"]
    for row, (interpolations, label) in enumerate(zip([gradient_interps, currot_interps], labels)):
        ax = f.add_axes([0.01, 0.08 + 0.47 * (1 - row), 0.845, 0.35])
        axs.append(ax)

        vl, = ax.plot(xs, values, color="C2")
        ax.fill_between(xs, 0, values, color="C2", alpha=0.7)

        for i, (interp_prob, delta) in enumerate(zip(interpolations, deltas)):
            color = cmap(delta)

            il, = ax.plot(xs, interp_prob / scale, color=color, zorder=2 + len(deltas) - i)
            ax.fill_between(xs, 0., interp_prob / scale, color=color, alpha=0.8, zorder=2 + len(deltas) - i)

        ax.set_title(label, fontsize=FONT_SIZE, pad=3.5)
        if row == 1:
            ax.set_xlabel(r"$c$", fontsize=FONT_SIZE, labelpad=-1)

        ax.set_xticklabels([])
        ax.grid()
        ax.set_xlim([0, 1])
        ax.set_xticks([0., 0.25, 0.5, 0.75, 1.])
        ax.set_ylim([0, 1])
        ax.set_yticks([0., 0.25, 0.5, 0.75, 1.])
        ax.set_yticklabels([])
        ax.tick_params('both', length=0, width=0, which='major')

    axs[0].tick_params('x', length=0, width=0, which='major')
    axs[1].tick_params('x', length=0, width=0, which='major')
    axs[0].tick_params('y', which="major", pad=0)
    axs[1].tick_params('y', which="major", pad=0)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cb1 = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cb1.set_label(r"$\delta$", fontsize=FONT_SIZE, labelpad=-1)
    cax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(f)


def wasserstein_interpolation_plot(path=None):
    from deep_sprl.teachers.spl.exact_currot import EntropicWassersteinBarycenter
    from tqdm import tqdm

    xs = np.linspace(0, 1, 500)
    target_probs1 = []
    target_probs2 = []
    alphass = [[0., 0.25, 0.5, 0.75, 1.], [0., 0.25, 0.5, 0.75, 1.]]

    target_prob1 = np.where(xs > 0.75, 1., 0.)
    target_prob1 /= np.sum(target_prob1)
    target_probs1.append(target_prob1)

    target_prob2 = np.where(xs < 0.25, 1., 0.)
    target_prob2 /= np.sum(target_prob2)
    target_probs2.append(target_prob2)

    target_prob1 = np.exp(-500 * np.square(xs - 0.9))
    target_prob1[target_prob1 < 1e-3] = 0
    target_prob1 /= np.sum(target_prob1)
    target_probs1.append(target_prob1)

    target_prob2 = np.exp(-500 * np.square(xs - 0.1))
    target_prob2[target_prob2 < 1e-3] = 0
    target_prob2 /= np.sum(target_prob2)
    target_probs2.append(target_prob2)

    f = plt.figure(figsize=(3.1, 1.3))
    cax = f.add_axes([0.86, 0.1, 0.03, 0.78])
    cmap = cm.get_cmap("inferno")
    for row, (target_prob1, target_prob2, alphas) in enumerate(zip(target_probs1, target_probs2, alphass)):
        # Compute normalization constants
        linit = target_prob1 / (xs[1] - xs[0])
        ltarget = target_prob2 / (xs[1] - xs[0])
        min_l = min(np.min(linit), np.min(ltarget))
        max_l = max(np.max(linit), np.max(ltarget))

        # Generate the different plots for the different eta
        ax = f.add_axes([0.01, 0.1 + 0.4 * row, 0.84, 0.38])
        interpolator = EntropicWassersteinBarycenter(1e-8, target_prob1)
        for i, alpha in tqdm(enumerate(alphas)):
            color = cmap(0.95 * (i / (len(alphas) - 1)))
            interp_prob = interpolator(np.abs(xs[None, :] - xs[:, None]) ** 2, alpha, target_prob2)
            linterp = interp_prob / (xs[1] - xs[0])
            norm_interp = (linterp - min_l) / (max_l - min_l)
            ax.plot(xs, norm_interp, color=color, zorder=2 + i)
            ax.fill_between(xs, -1, norm_interp, color=color, alpha=0.8, edgecolor=color, linewidth=2, zorder=2 + i)

        ax.set_xlim([0, 1])
        ax.set_xticks([0., 0.25, 0.5, 0.75, 1.])
        ax.set_ylim([0, 1])
        ax.set_yticks([0., 0.25, 0.5, 0.75, 1.])

        if row == 1:
            ax.set_title(r"$\mathcal{B}([\alpha, 1{-}\alpha], [p_1, p_2])$", fontsize=FONT_SIZE, pad=3)

        if row == 0:
            ax.set_xlabel(r"$c$", fontsize=FONT_SIZE, labelpad=-1)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid()
        ax.tick_params('both', length=0, width=0, which='major')

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cb1 = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cb1.set_label(r"$\alpha$", fontsize=FONT_SIZE, labelpad=-1)
    cax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


class WassersteinSPDL:

    def __init__(self, delta: float, mu: np.ndarray):
        self.delta = delta
        self.mu = mu

        self.n_mu = np.sum(self.mu > 0)
        self.n = mu.shape[0]
        wcon = generate_wasserstein_constraint(self.n_mu, self.n)
        self.con = np.concatenate((wcon, np.zeros((wcon.shape[0], self.n))), axis=-1)
        self.con[self.n_mu:, -self.n:] = -np.eye(self.n)

        self.con_val = np.concatenate((self.mu[self.mu > 0], np.zeros(self.n)), axis=0)
        self.bounds = [(0, None) for _ in range(self.con.shape[1])]

    def __call__(self, metric: np.ndarray, performances: np.ndarray):
        c = np.zeros(self.con.shape[1])
        c[:-self.n] = np.reshape(metric[self.mu > 0, :], -1)

        performance_con = np.zeros((1, self.con.shape[1]))
        performance_con[0, -self.n:] = -performances

        res = linprog(c, A_eq=self.con, b_eq=self.con_val, bounds=self.bounds, method="highs-ds",
                      A_ub=performance_con, b_ub=-np.array([self.delta]))
        return res.x[-self.n:]


if __name__ == "__main__":
    single_column = True
    dir_name = "figures_sc" if single_column else "figures"
    os.makedirs(dir_name, exist_ok=True)
    kl_interpolation_plot2(path=f"{dir_name}/kl_interpolation.pdf", single_column=single_column)
    wasserstein_interpolation_plot(path=f"{dir_name}/analytic_wasserstein_interpolation.pdf")
    ep_interpolation_plot(path=f"{dir_name}/expected_performance_interpolation.pdf", single_column=single_column)
    currot_gradient_interpolation(path=f"{dir_name}/currot_gradient_diff.pdf")
