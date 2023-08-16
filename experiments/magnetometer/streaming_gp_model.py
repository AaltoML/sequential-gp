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

from exp_utils import get_hydra_output_dir, convert_data_to_online

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../configs/", config_name="magnetometer_streaming_experiment")
def run_experiment(cfg: DictConfig):
    """
    Initialize and run the experiment.
    """
    all_train_data, test_data = instantiate(cfg.dataset.dataloader)(train_id=[3], test_id=[1, 2, 4, 5])
    log.info("Data loaded successfully!!!")

    output_dir = get_hydra_output_dir()

    # Set up inducing variables
    n_inducing_variable = int(np.sqrt(cfg.n_inducing_variable).item())

    xx, yy = np.linspace(-2, 5, n_inducing_variable), np.linspace(-4, 2, n_inducing_variable)
    z1, z2 = np.meshgrid(xx, yy)
    zz = np.vstack((z1.flatten(), z2.flatten())).T
    inducing_variable = zz.tolist()
    cfg.model.inducing_variable = inducing_variable

    online_data = convert_data_to_online(all_train_data[0], n_sets=20)
    model = instantiate(cfg.model)(data=online_data[0])
    model.kernel.kernels[0].variance.assign(500)

    optimizer = instantiate(cfg.optimizer)
    nlpd_vals, rmse_vals, _ = call(cfg.optimize)(optimizer=optimizer, model=model, train_data=online_data,
                                                 test_data=test_data)

    log.info(f"Test NLPD: {nlpd_vals[-1]}")
    log.info(f"Test RMSE: {rmse_vals[-1]}")
    log.info("Optimization successfully done!!!")

    parameters = gpflow.utilities.parameter_dict(model)
    with open(os.path.join(output_dir, "model_streaming_magnetometer.pkl"), "wb") as f:
        pickle.dump(parameters, f)


if __name__ == '__main__':
    run_experiment()
