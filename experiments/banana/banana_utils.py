from typing import Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gpflow

import sys

sys.path.append("../../")
from src.streaming_sparse_gp.osvgpc import OSVGPC


def load_banana_dataset() -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.loadtxt(
        "../data/banana_train_x.txt",
        delimiter=","
    )
    train_y = np.loadtxt(
        "../data/banana_train_y.txt", delimiter=","
    )
    train_y[train_y == -1] = 0

    test_x = np.loadtxt(
        "../data/banana_test_x.txt",
        delimiter=","
    )
    test_y = np.loadtxt(
        "../data/banana_test_y.txt", delimiter=","
    )
    test_y[test_y == -1] = 0

    return train_x, train_y, test_x, test_y


def plot_banana(pred_mu, pred_var, pred_prob, inducing_pnts, data, xtest, ytest, vmin=0., vmax=1.,
                plot_inducing=False, plot_probability=False, plot_colorbar=False, previous_data=None):
    if plot_probability:
        camp0_color = ["C1", "white"]
        camp1_color = ["white", "C0"]
    else:
        camp0_color = ["white", "C1"]
        camp1_color = ["C0", "white"]

    cmap0 = matplotlib.colors.LinearSegmentedColormap.from_list("", camp0_color)
    cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", camp1_color)
    colors0 = cmap0(np.linspace(0, 1., 128))
    colors1 = cmap1(np.linspace(0, 1., 128))
    colors = np.append(colors0, colors1, axis=0)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', colors)

    X, Y = data

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    for i, mark, color in [[1, 'o', 'C0'], [0, 's', 'C1']]:
        ind = Y[:, 0] == i
        ax.scatter(X[ind, 0], X[ind, 1], s=100, alpha=.5, edgecolor='k', marker=mark, color=color)

    # Plotting prevous data ghosted out
    if previous_data is not None:
        X_prev, Y_prev = previous_data

        for i, mark, color in [[1, 'o', 'C0'], [0, 's', 'C1']]:
            ind = Y_prev[:, 0] == i
            ax.scatter(X_prev[ind, 0], X_prev[ind, 1], s=100, alpha=.1, edgecolor='k', marker=mark, color=color)

    if plot_inducing and inducing_pnts is not None:
        ax.scatter(inducing_pnts[:, 0], inducing_pnts[:, 1], s=40, color='k')

    # Scale background
    if plot_probability:
        foo = pred_prob.numpy()
    else:
        foo = pred_mu.numpy() > 0.5
        foo = foo.astype(float)
        foo = (2. * foo - 1.) * np.sqrt(pred_var.numpy())
    if vmax is None:
        vmax = np.max(np.sqrt(pred_var.numpy()))
        vmin = -vmax
    im = ax.imshow(foo.reshape(100, 100).transpose(), extent=[-2.8, 2.8, -2.8, 2.8],
                   origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    ax.axis('equal')

    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.8, 2.8)
    ax.contour(xtest, ytest, pred_mu.numpy().reshape(100, 100), levels=[.5],
               colors='k', linewidths=4.)

    if plot_colorbar:
        plt.colorbar(im, ax=ax)

    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.8, 2.8)


def optimize_streaming_model(optimizer, model, train_data: Tuple[np.ndarray, np.ndarray],
                             iterations: int = 100, mu=None, Su=None, Kaa=None, Zopt=None, first_init=True):
    """
    Optimize Bui model
    """

    def optimization_step_adam():
        optimizer.minimize(model.training_loss, model.trainable_variables)

    def optimization_step_scipy():
        optimizer.minimize(model.training_loss, model.trainable_variables, options={'maxiter': iterations})

    def optimization_step():
        if isinstance(optimizer, gpflow.optimizers.Scipy):
            optimization_step_scipy()
        else:
            for _ in range(iterations):
                optimization_step_adam()

    def init_Z(cur_Z, new_X, use_old_Z=True):
        if use_old_Z:
            Z = np.copy(cur_Z)
        else:
            M = cur_Z.shape[0]
            M_old = int(0.7 * M)
            M_new = M - M_old
            old_Z = cur_Z[np.random.permutation(M)[0:M_old], :]
            new_Z = new_X[np.random.permutation(new_X.shape[0])[0:M_new], :]
            Z = np.vstack((old_Z, new_Z))
        return Z

    use_old_z = True

    X, y = train_data

    if first_init:
        if isinstance(optimizer, gpflow.optimizers.Scipy):
            gpflow.optimizers.Scipy().minimize(
                model.training_loss_closure((X, y)), model.trainable_variables, options={'maxiter': iterations})
        else:
            for _ in range(iterations):
                optimizer.minimize(model.training_loss_closure((X, y)), model.trainable_variables)
    else:
        Zinit = init_Z(Zopt, X, use_old_z)
        model = OSVGPC((X, y), gpflow.kernels.Matern52(), gpflow.likelihoods.Bernoulli(), mu, Su, Kaa,
                       Zopt, Zinit)
        optimization_step()

    Zopt = model.inducing_variable.Z.numpy()
    mu, Su = model.predict_f(Zopt, full_cov=True)
    if len(Su.shape) == 3:
        Su = Su[0, :, :] + 1e-4 * np.eye(mu.shape[0])
    Kaa = model.kernel(model.inducing_variable.Z)

    return mu, Su, Kaa, Zopt, model
