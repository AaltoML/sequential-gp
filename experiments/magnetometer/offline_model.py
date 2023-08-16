import logging
import os
import pickle

import gpflow
import numpy as np
import hydra
from omegaconf import DictConfig
from hydra.utils import call, instantiate

import sys
sys.path.append("..")
sys.path.append("../../")

from exp_utils import get_hydra_output_dir

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../configs/", config_name="magnetometer_offline_experiment")
def run_experiment(cfg: DictConfig):
    """
    Initialize and run the experiment.
    """
    all_train_data, test_data = instantiate(cfg.dataset.dataloader)()
    log.info("Data loaded successfully!!!")

    output_dir = get_hydra_output_dir()

    # Merge all train_data and test_data into one
    train_data = None
    for data in all_train_data:
        if train_data is None:
            train_data = (data[0], data[1])
        else:
            train_data = (np.concatenate([train_data[0], data[0]]), np.concatenate([train_data[1], data[1]]))

    # Set up inducing variables
    n_inducing_variable = int(np.sqrt(cfg.n_inducing_variable).item())
    xx, yy = np.linspace(-2, 5, n_inducing_variable), np.linspace(-4, 2, n_inducing_variable)
    z1, z2 = np.meshgrid(xx, yy)
    zz = np.vstack((z1.flatten(), z2.flatten())).T
    inducing_variable = zz.tolist()
    cfg.model.inducing_variable = inducing_variable
    cfg.model.num_data = train_data[0].shape[0]

    model = instantiate(cfg.model)
    model.kernel.kernels[0].variance.assign(500)

    elbo_vals, nlpd_vals, rmse_vals = call(cfg.optimize)(model=model, train_data=train_data, test_data=test_data,
                                                         optimizer=instantiate(cfg.optimizer), debug=True)

    log.info(f"Test NLPD: {nlpd_vals[-1]}")
    log.info(f"Test RMSE: {rmse_vals[-1]}")

    log.info("Optimization successfully done!!!")

    parameters = gpflow.utilities.parameter_dict(model)
    with open(os.path.join(output_dir, "model_offline_magnetometer.pkl"), "wb") as f:
        pickle.dump(parameters, f)


if __name__ == '__main__':
    run_experiment()
