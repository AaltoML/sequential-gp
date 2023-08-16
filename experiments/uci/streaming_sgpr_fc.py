"""
@inproceedings{BuiNguTur17,
  title =  {Streaming sparse {G}aussian process approximations},
  author =   {Bui, Thang D. and Nguyen, Cuong V. and Turner, Richard E.},
  booktitle = {Advances in Neural Information Processing Systems 30},
  year =   {2017}
}

@article{BuiYanTur16,
  title={A Unifying Framework for Sparse {G}aussian Process Approximation using {P}ower {E}xpectation {P}ropagation},
  author={Thang D. Bui and Josiah Yan and Richard E. Turner},
  journal={arXiv preprint arXiv:1605.07066},
  year={2016}
}
"""
import os
import logging
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, call
import hydra
import wandb
import gpflow

import sys
sys.path.append("..")

from exp_utils import convert_data_to_online, get_hydra_output_dir


def _setup_wandb(cfg):
    """
    Set up wandb if username is passed.
    """
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    wandb.init(project="UCI", entity=cfg.wandb.username, config=wandb_cfg)

    log.info("wandb initialized!!!")


# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../configs/", config_name="streaming_experiment")
def run_experiment(cfg: DictConfig):
    """
    Initialize and run the experiment.
    """
    output_dir = get_hydra_output_dir()

    all_train_data, all_test_data = call(cfg.dataset.dataloader)

    if len(all_train_data) > 1:
        log.info(f"Cross validation starting with {cfg.dataset.dataloader.n_k_folds} K-folds!!!")

    k_fold_nlpds = []
    k_fold_rmse = []
    k_fold_time = []
    k_fold_id = 0

    log.info(f"---------------------------------------------")
    log.info(f"Dataset : {cfg.dataset}")
    log.info(f"---------------------------------------------")
    for train_data, test_data in zip(all_train_data, all_test_data):

        log.info(f"---------------------------------------------")
        log.info(f"Starting with set {k_fold_id}")
        log.info(f"---------------------------------------------")

        log.info(f"Train data are {train_data[0].shape[0]} and test data are {test_data[0].shape[0]}")
        log.info("Splitting data into sets...")

        online_data = convert_data_to_online(train_data, cfg.n_sets, cfg.sort_data)
        log.info(f"Data splitted successfully into {cfg.n_sets} sets!!!")

        np.savez(os.path.join(output_dir, "splitted_dataset" + str(k_fold_id) + ".npz"), data=online_data)

        if cfg.wandb.username is not None:
            _setup_wandb(cfg)

        # Set up inducing variables
        inducing_variable = train_data[0][:cfg.n_inducing_variable].copy().tolist()
        cfg.model.inducing_variable = inducing_variable

        if "SGPR" in cfg.model._target_:
            model = instantiate(cfg.model)(data=online_data[0])
        else:
            cfg.model.num_data = train_data[0].shape[0]
            model = instantiate(cfg.model)

        # Loading model hyperparam values
        if cfg.load_model_path is None:
            raise Exception("FC model should have a model path from where hyperparams are loaded!")

        model_path = os.path.join(cfg.load_model_path, "model_" + str(k_fold_id) + ".pkl")

        with open(model_path, "rb") as f:
            dict_params = pickle.load(f)

        model.inducing_variable.Z = dict_params['.inducing_variable.Z']
        model.kernel.lengthscales = dict_params['.kernel.lengthscales']
        model.kernel.variance = dict_params['.kernel.variance']

        # not present in classification
        if '.likelihood.variance' in dict_params:
            model.likelihood.variance = dict_params['.likelihood.variance']

        # make then non-trainable
        gpflow.utilities.set_trainable(model.inducing_variable.Z, False)
        gpflow.utilities.set_trainable(model.kernel.lengthscales, False)
        gpflow.utilities.set_trainable(model.kernel.variance, False)

        if isinstance(model.likelihood, gpflow.likelihoods.Gaussian):
            gpflow.utilities.set_trainable(model.likelihood.variance, False)

        log.info("Model initialized; Optimization started!!!")

        optimizer = instantiate(cfg.optimizer)

        start_time = time.time()
        nlpd_vals, rmse_vals, time_vals = call(cfg.optimize)(optimizer=optimizer, model=model, train_data=online_data,
                                                  test_data=test_data, use_old_z=True, fast_conditioning=True)
        end_time = time.time()

        log.info(f"Test NLPD: {nlpd_vals[-1]}")
        log.info(f"Test RMSE: {rmse_vals[-1]}")
        log.info(f"Time (s): {end_time - start_time}")
        log.info("Optimization successfully done!!!")

        if cfg.wandb.username is not None:
            plt.clf()
            plt.plot(nlpd_vals)
            plt.title("NLPD")
            wandb.log({"optim_nlpd_vals": plt})

            plt.clf()
            plt.plot(rmse_vals)
            plt.title("RMSE")
            wandb.log({"optim_rmse_vals": plt})

        np.savez(os.path.join(output_dir, "training_statistics_" + str(k_fold_id) + ".npz"), nlpd=nlpd_vals,
                 rmse=rmse_vals, time_vals=time_vals)

        parameters = gpflow.utilities.parameter_dict(model)
        with open(os.path.join(output_dir, "model_" + str(k_fold_id) + ".pkl"), "wb") as f:
            pickle.dump(parameters, f)

        k_fold_id += 1
        k_fold_nlpds.append(nlpd_vals[-1])
        k_fold_rmse.append(rmse_vals[-1])
        k_fold_time.append(end_time-start_time)
        log.info(f"---------------------------------------------")

    if len(k_fold_nlpds) > 1:
        log.info(f"Mean NLPD over k-folds = {np.mean(k_fold_nlpds)}")
        log.info(f"Std NLPD over k-folds = {np.std(k_fold_nlpds)}")

        log.info(f"Mean RMSE over k-folds = {np.mean(k_fold_rmse)}")
        log.info(f"Std RMSE over k-folds = {np.std(k_fold_rmse)}")

        log.info(f"Mean time over k-folds = {np.mean(k_fold_time)}")
        log.info(f"Std time over k-folds = {np.std(k_fold_time)}")


if __name__ == '__main__':
    run_experiment()
