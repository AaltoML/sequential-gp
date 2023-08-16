import os
import pickle
import gpflow
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

sys.path.append("../..")
sys.path.append("..")

from magnetometer_utils import load_data, transform_room2video, get_transformed_grid, transform_grid2room
from exp_utils import convert_data_to_online
from src.models.tsvgp_cont import t_SVGP_cont, OnlineGP
from src.models.utils import memory_picker, piv_chol


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot online plots for the magntometer experiment.")
    parser.add_argument("-o", type=str, default=None, required=True)
    args = parser.parse_args()
    model_dir = args.o
    streaming = False

    if not os.path.exists(model_dir):
        raise Exception("Model directory is invalid!!!")

    model_names = []
    for f in os.listdir(model_dir):
        if "online_magnetometer.pkl" in f:
            model_names.append(f)

    # sort by model id
    model_names.sort()

    n_inducing_variable = 100
    if streaming:
        train_data, _ = load_data("../data/invensense", train_id=[3])
        train_data = convert_data_to_online(train_data[0], n_sets=20, shuffle=False)
    else:
        train_data, _ = load_data("../data/invensense", train_id=[1, 2, 4, 5])

    for i, model_name in enumerate(model_names):
        model_path = os.path.join(model_dir, model_name)
        with open(model_path, "rb") as f:
            dict_params = pickle.load(f)

        # Loading model
        kernel = gpflow.kernels.Sum([gpflow.kernels.Constant(), gpflow.kernels.Matern52()])
        likelihood = gpflow.likelihoods.Gaussian()
        inducing_variable = -2 + np.zeros((n_inducing_variable, 2)) + np.random.rand(n_inducing_variable * 2).reshape(
            (-1, 2))
        model = t_SVGP_cont(kernel, likelihood, inducing_variable)
        model.kernel.kernels[0].variance = dict_params['.kernel.kernels[0].variance']
        model.kernel.kernels[1].lengthscales = dict_params['.kernel.kernels[1].lengthscales']
        model.kernel.kernels[1].variance = dict_params['.kernel.kernels[1].variance']
        model.likelihood.variance = dict_params['.likelihood.variance']
        model.inducing_variable.Z = dict_params['.inducing_variable.Z']
        Z = model.inducing_variable.Z.numpy().copy()
        model.lambda_1.assign(dict_params['.sites.lambda_1'])
        model.lambda_2.assign(dict_params['.sites._lambda_2'])
        print("Model loaded successfully!!!")

        # Prediction over grid
        xtest, ytest = np.mgrid[-1.:1.:100j, -1.:1.:100j]
        xtest_transformed, ytest_transformed = transform_grid2room(xtest, ytest)
        zz = np.concatenate([xtest_transformed[..., None], ytest_transformed[..., None]], axis=1)

        pred_m_grid, pred_S_grid = model.predict_f(zz)
        pred_m_grid = pred_m_grid.numpy().reshape((100, -1))

        pred_S_grid = pred_S_grid.numpy()
        if not streaming:
            alpha_map = np.sqrt(pred_S_grid).reshape((100, 100))
        else:
            alpha_map = np.exp(-1 * np.sqrt(pred_S_grid).reshape((100, 100)))
        alpha_map = alpha_map - np.min(alpha_map)
        alpha_map = alpha_map / np.max(alpha_map)
        alpha_map = 1 - alpha_map

        # Test points
        transformed_x1test, transformed_x2test = transform_room2video(xtest_transformed, ytest_transformed)
        transformed_x1test = np.reshape(transformed_x1test, xtest.shape)
        transformed_x2test = np.reshape(transformed_x2test, ytest.shape)

        # Path
        path_transformed_x0, path_transformed_x1 = transform_room2video(train_data[i][0][:, 0],
                                                                        train_data[i][0][:, 1])
        path_transformed_x = np.concatenate([path_transformed_x0[..., None], path_transformed_x1[..., None]], axis=1)

        # Grid
        g1, g2 = get_transformed_grid()

        # Inducing variables
        idx, _ = np.where((Z < -3.0) | (Z > 5.0))
        Z = np.delete(Z, idx, axis=0)
        transformed_Z_0, transformed_Z_1 = transform_room2video(Z[:, 0], Z[:, 1])
        transformed_Z_0 = np.reshape(transformed_Z_0, Z[:, 0].shape)
        transformed_Z_1 = np.reshape(transformed_Z_1, Z[:, 1].shape)

        # Plotting
        plt.clf()
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
        plt.show()
