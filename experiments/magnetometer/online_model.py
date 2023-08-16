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

from exp_utils import get_hydra_output_dir, convert_data_to_online
from src.models.tsvgp_cont import piv_chol, fixed_Z, random_picker

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../configs/", config_name="magnetometer_online_experiment")
def run_experiment(cfg: DictConfig):
    """
    Initialize and run the experiment.
    """
    if cfg.streaming:
        all_train_data, test_data = instantiate(cfg.dataset.dataloader)(train_id=[3], test_id=[1, 2, 4, 5])
        train_data = convert_data_to_online(all_train_data[0], n_sets=20)
    else:
        train_data, test_data = instantiate(cfg.dataset.dataloader)(train_id=[1, 2, 4, 5], test_id=[1])
    log.info("Data loaded successfully!!!")

    output_dir = get_hydra_output_dir()

    n_inducing_variable = int(np.sqrt(cfg.n_inducing_variable).item())
    xx, yy = np.linspace(-2, 5, n_inducing_variable), np.linspace(-4, 2, n_inducing_variable)
    z1, z2 = np.meshgrid(xx, yy)
    zz = np.vstack((z1.flatten(), z2.flatten())).T
    inducing_variable = zz.tolist()
    cfg.model.inducing_variable = inducing_variable

    model = instantiate(cfg.model)
    model.kernel.kernels[0].variance.assign(500)
    gpflow.set_trainable(model.inducing_variable.Z, True)

    memory = (train_data[0][0][:1], train_data[0][1][:1])
    online_gp = instantiate(cfg.online_gp)(model=model, memory=memory, opt_hypers=instantiate(cfg.optimizer),
                                           Z_picker=fixed_Z, memory_picker=random_picker)

    for i, set_data in enumerate(train_data):
        if cfg.streaming:
            test_data = test_data
            nlpd_vals, rmse_vals, _ = call(cfg.optimize)(online_gp=online_gp, train_data=[set_data],
                                                         test_data=test_data, debug=False)
        else:
            test_data = set_data
            set_data = convert_data_to_online(set_data, n_sets=20)
            nlpd_vals, rmse_vals, _ = call(cfg.optimize)(online_gp=online_gp, train_data=set_data,
                                                         test_data=test_data, debug=True)

        log.info(f"------------------------------------------")
        log.info(f"Set {i}")
        log.info(f"Test NLPD: {nlpd_vals[-1]}")
        log.info(f"Test RMSE: {rmse_vals[-1]}")
        log.info(f"------------------------------------------")

        parameters = gpflow.utilities.parameter_dict(model)
        with open(os.path.join(output_dir, "model" + str(i) + "_online_magnetometer.pkl"), "wb") as f:
            pickle.dump(parameters, f)


if __name__ == '__main__':
    run_experiment()
