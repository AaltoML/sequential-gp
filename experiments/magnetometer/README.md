# Magnetic Anomalies Experiment

The experiment to map the local anomalies in the ambient magnetic field. The experiment is based on [Solin et al. (2018)](https://ieeexplore.ieee.org/document/8373720).

## Offline Model
For training an offline model:
```shell
python offline_model.py model/kernel=sum_constant_matern52
```
The configurations and model hyperparameters can be found in the Hydra config file `../configs/magnetometer_offline_experiment.yaml`.
## Fast Conditiong of the proposed model

For fast-conditiong of the model, Figure 5(b), run the following command:
```shell
python online_fc_plots.py -o={offline_model_path}
```

## Proposed model

Experiment when multiple observations paths are observed:
```
python online_model.py model/kernel=sum_constant_matern52
```

The configurations and model hyperparameters can be found in the Hydra config file `../configs/magnetometer_online_experiment.yaml`. 

## For NLPD comparison experiment

**Proposed Model**

```shell
python online_model.py model/kernel=sum_constant_matern52 streaming=True
```
The configurations and model hyperparameters can be found in the Hydra config file `../configs/magnetometer_online_experiment.yaml`.

**Bui et al. (2017):**

```
python streaming_gp_model.py model/kernel=sum_constant_matern52
```

The configurations and model hyperparameters can be found in the Hydra config file `../configs/magnetometer_streaming_experiment.yaml`. 
