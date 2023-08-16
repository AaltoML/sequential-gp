import os

import numpy as np
import pandas as pd

A_room2video = np.array([70.7791, -388.6396, 618.0954,
                         -66.1362, 26.6245, 665.3620,
                         0, 0, 1.0000]).reshape((3, 3))

C_room2video = np.array([0.1597, -0.0318, 1.0000]).reshape((1, 3))

A_grid2room = np.array([0.0118, 2.7777, 1.3689,
                        -2.2243, -0.0967, -1.0929,
                        0, 0, 1.0000]).reshape((3, 3))

C_grid2room = np.array([0.0160, -0.1450, 1.0000]).reshape((1, 3))


def load_data(main_dir: str, train_id: list = None, test_id: list = None) -> [list, list]:
    """
    Load magnetometer data.

    Main source of data is: https://github.com/AaltoML/magnetic-data

    Note: The function involves some constants that are specific to the data.
    """
    data_train = []
    data_test = None

    if train_id is None:
        train_id = [1, 2, 4, 5]

    if test_id is None:
        test_id = [1]

    for i in train_id:
        loc_path = os.path.join(main_dir, str(i) + "-loc.csv")
        mag_path = os.path.join(main_dir, str(i) + "-mag.csv")

        loc_data = pd.read_csv(loc_path).to_numpy()
        mag_data = pd.read_csv(mag_path).to_numpy()

        # take norm of mag data
        mag_data_norm = np.sqrt(np.sum(np.square(mag_data), axis=-1))[..., None]
        data_combined = np.concatenate([loc_data, mag_data_norm], axis=1)

        data_train.append([data_combined[:, :-1], data_combined[:, -1:]])

    for i in test_id:
        loc_path = os.path.join(main_dir, str(i) + "-loc.csv")
        mag_path = os.path.join(main_dir, str(i) + "-mag.csv")

        loc_data = pd.read_csv(loc_path).to_numpy()
        mag_data = pd.read_csv(mag_path).to_numpy()

        # take norm of mag data
        mag_data_norm = np.sqrt(np.sum(np.square(mag_data), axis=-1))[..., None]

        data_combined = np.concatenate([loc_data, mag_data_norm], axis=1)

        if data_test is None:
            data_test = [np.array(data_combined[:, :-1]), np.array(data_combined[:, -1:])]
        else:
            data_test = [np.concatenate([data_test[0], np.array(data_combined[:, :-1])], axis=0),
                         np.concatenate([data_test[1], np.array(data_combined[:, -1:])], axis=0)]

    return data_train, data_test


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
Below function are for plotting purposes and comes from original Matlab scripts.

They are for transformation between room, video, grid.
"""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def get_transformed_grid():
    z1 = np.concatenate([np.linspace(-1, 1, 32), np.nan * np.ones((1,))])
    z2 = z1.copy()

    g1, g2 = np.meshgrid(z1, z2)
    Z = np.concatenate([g1.reshape((-1, 1)), g2.reshape((-1, 1))], axis=1)

    var1 = A_grid2room @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T
    var2 = C_grid2room @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T

    Z = np.divide(var1, var2).T
    Z = Z[:, :2]

    var1 = A_room2video @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T
    var2 = C_room2video @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T
    Y = np.divide(var1, var2).T

    g1 = np.reshape(Y[:, 0], g1.shape)
    g2 = np.reshape(Y[:, 1], g2.shape)

    return g1, g2


def transform_grid2video(x, y):
    Z = np.concatenate([x.reshape((-1, 1)), y.reshape((-1, 1))], axis=1)
    var1 = A_grid2room @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T
    var2 = C_grid2room @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T

    Z = np.divide(var1, var2).T
    Z = Z[:, :2]

    var1 = A_room2video @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T
    var2 = C_room2video @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T
    Y = np.divide(var1, var2).T
    return Y[:, 0], Y[:, 1]


def transform_room2video(x, y):
    Z = np.concatenate([x.reshape((-1, 1)), y.reshape((-1, 1))], axis=1)
    var1 = A_room2video @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T
    var2 = C_room2video @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T
    Y = np.divide(var1, var2).T

    return Y[:, 0], Y[:, 1]


def transform_grid2room(x, y):
    Z = np.concatenate([x.reshape((-1, 1)), y.reshape((-1, 1))], axis=1)
    var1 = A_grid2room @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T
    var2 = C_grid2room @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T
    Y = np.divide(var1, var2).T
    return Y[:, 0], Y[:, 1]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
