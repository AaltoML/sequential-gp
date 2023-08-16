import pickle
import argparse
import gpflow
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import sys
sys.path.append("../..")
sys.path.append("..")

from magnetometer_utils import load_data, transform_room2video, get_transformed_grid, transform_grid2room
from exp_utils import convert_data_to_online
from src.models.tsvgp_cont import t_SVGP_cont, OnlineGP
from src.models.utils import piv_chol, memory_picker

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot streaming plots for the magntometer experiment.")
    parser.add_argument("-o", type=str, default=None, required=True)
    args = parser.parse_args()

    model_path = args.o
    n_inducing_variable = 100

    with open(model_path, "rb") as f:
        dict_params = pickle.load(f)

    kernel = gpflow.kernels.Sum([gpflow.kernels.Constant(), gpflow.kernels.Matern52()])
    likelihood = gpflow.likelihoods.Gaussian()
    inducing_variable = -2 + np.zeros((n_inducing_variable, 2)) + np.random.rand(n_inducing_variable * 2).reshape((-1, 2))
    model = t_SVGP_cont(kernel, likelihood, inducing_variable)

    model.kernel.kernels[0].variance = dict_params['.kernel.kernels[0].variance']
    model.kernel.kernels[1].lengthscales = dict_params['.kernel.kernels[1].lengthscales']
    model.kernel.kernels[1].variance = dict_params['.kernel.kernels[1].variance']
    model.likelihood.variance = dict_params['.likelihood.variance']

    print("Model loaded successfully!!!")

    train_data, _ = load_data("../data/invensense", train_id=[3])
    online_data = convert_data_to_online(train_data[0], n_sets=20, shuffle=False)

    # Fixing hyper parameters
    gpflow.utilities.set_trainable(model.kernel.kernels[0].variance, False)
    gpflow.utilities.set_trainable(model.kernel.kernels[1].lengthscales, False)
    gpflow.utilities.set_trainable(model.kernel.kernels[1].variance, False)
    gpflow.utilities.set_trainable(model.likelihood.variance, False)

    memory = (online_data[0][0][:1], online_data[0][1][:1])
    online_gp = OnlineGP(model, opt_hypers=tf.optimizers.Adam(), n_steps=2, lambda_lr=0.9, memory=memory,
                         Z_picker=piv_chol, memory_picker=memory_picker, num_mem=10)

    path_x = None
    path_y = None

    # init Z
    first_batch = online_data[0][0]
    mean_first_batch = np.mean(first_batch, axis=0)
    var_first_batch = np.var(first_batch, axis=0)
    cov_first_batch = np.diag(var_first_batch.reshape(-1))
    mean_first_batch = mean_first_batch.reshape(-1)

    inducing_variable = np.random.multivariate_normal(mean_first_batch, cov_first_batch, n_inducing_variable)
    model.inducing_variable.Z.assign(inducing_variable)

    for i, batch_data in enumerate(online_data):
        online_gp.update_with_new_batch(batch_data, train_hyps=False, train_mem=True, n_hyp_opt_steps=5)

        if path_x is None:
            path_x = batch_data[0]
            path_y = batch_data[1]
        else:
            path_x = np.concatenate([path_x, batch_data[0]], axis=0)
            path_y = np.concatenate([path_y, batch_data[1]], axis=0)

        # only plot every 5th batch
        if (i != 0) and (i+1) % 5 != 0:
            continue

        Z_new = model.inducing_variable.Z.numpy()

        # Prediction over grid
        xtest, ytest = np.mgrid[-1.3:1.3:100j, -1.3:1.3:100j]
        xtest_transformed, ytest_transformed = transform_grid2room(xtest, ytest)
        zz = np.concatenate([xtest_transformed[..., None], ytest_transformed[..., None]], axis=1)

        pred_m_grid, pred_S_grid = online_gp.model.predict_f(zz)
        pred_m_grid = pred_m_grid.numpy().reshape((100, -1))

        pred_S_grid = pred_S_grid.numpy()
        alpha_map = np.exp(-np.sqrt(pred_S_grid)).reshape((100, 100))
        alpha_map = alpha_map - np.min(alpha_map)
        alpha_map = alpha_map/np.max(alpha_map)
        # alpha_map = 1 - alpha_map

        # Test points
        transformed_x1test, transformed_x2test = transform_room2video(xtest_transformed, ytest_transformed)
        transformed_x1test = np.reshape(transformed_x1test, xtest.shape)
        transformed_x2test = np.reshape(transformed_x2test, ytest.shape)

        # Path
        path_transformed_x0, path_transformed_x1 = transform_room2video(path_x[:, 0], path_x[:, 1])
        path_transformed_x = np.concatenate([path_transformed_x0[..., None], path_transformed_x1[..., None]], axis=1)

        # Grid
        g1, g2 = get_transformed_grid()

        # Inducing variables
        transformed_Z_0, transformed_Z_1 = transform_room2video(Z_new[:, 0], Z_new[:, 1])
        transformed_Z_0 = np.reshape(transformed_Z_0, Z_new[:, 0].shape)
        transformed_Z_1 = np.reshape(transformed_Z_1, Z_new[:, 1].shape)

        # Plotting
        # plt.clf()
        _, axs = plt.subplots(1, 1)
        plt.plot(path_transformed_x[:, 0], path_transformed_x[:, 1])
        pcol = plt.pcolormesh(transformed_x1test, transformed_x2test, pred_m_grid, alpha=alpha_map.reshape(-1),
                              vmin=10, vmax=90, shading='gouraud', cmap="jet")
        pcol.set_edgecolor('face')

        plt.scatter(transformed_Z_0, transformed_Z_1, color="black")

        plt.plot(g1, g2, color="gray", alpha=0.2)
        plt.plot(g1.T, g2.T, color="gray", alpha=0.2)

        plt.xlim([0, 1920])
        plt.ylim([0, 1080])
        axs.set_aspect("equal")
        plt.axis('off')
        plt.gca().invert_yaxis()
        # plt.savefig('robot' + str(i + 1) + '.png', bbox_inches='tight', pad_inches=0, dpi=200)
        plt.show()
