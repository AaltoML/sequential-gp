# UCI Experiment

In the UCI experiment we compare three models: the offline model that has access to the whole dataset, our proposed online model, and the online model proposed by Bui *et al.* (2017).

For the paper, for each data set we perform 10-fold cross-validation. In all cases we use a Matérn-5/2 kernel.

## Offline model

The sparse variational GP model (t-SVGP) with access to the whole data set. 

To run the model on a **regression** dataset:
```
python offline_model.py dataset=bike model.likelihood.variance=0.1 dataset.dataloader.n_k_folds=10 dataset.dataloader.random_state=33
```
To run the model on a **classification** dataset:
```
python offline_model.py dataset=adult optimize.lambda_lr=0.5 model/likelihood=bernoulli dataset.dataloader.n_k_folds=10 dataset.dataloader.random_state=33
```

The configurations and model hyperparameters can be found in the Hydra config file `../configs/offline_experiment.yaml`. 

## Proposed model

To run the model on a **regression** dataset:
```
python online_model.py dataset=bike dataset.dataloader.n_k_folds=10 dataset.dataloader.random_state=33 n_sets=50 model.likelihood.variance=0.1 optimize.hyperparams_step=100 online_gp.n_steps=2
```
To run the model on a **classification** dataset:
```
python online_model.py dataset=adult model/likelihood=bernoulli online_gp.lambda_lr=0.5 online_gp.n_steps=4 dataset.dataloader.n_k_folds=10 dataset.dataloader.random_state=33 n_sets=50 optimize.hyperparams_step=100
```
The configurations and model hyperparameters can be found in the Hydra config file `../configs/online_experiment.yaml`.

The memory picker can be changed from BLS (by default) to random by using the command-line argument `online_gp.memory_picker=random`.

## Bui *et al.* (2017)

To run the Bui *et al.* (2017) model on a **regression** dataset:
```
python streaming_sgpr.py dataset=bike dataset.dataloader.n_k_folds=10 dataset.dataloader.random_state=33 n_sets=50 optimize.iterations=100 model.noise_variance=0.1 optimizer=scipy
```
To run the Bui *et al.* (2017) model on a **classification** dataset:
```
python streaming_sgpr.py dataset=adult dataset.dataloader.n_k_folds=10 dataset.dataloader.random_state=33 optimizer=adam optimize.iterations=100 optimize.task="classification" model=svgp model/likelihood=bernoulli n_sets=50
```
The configurations and model hyperparameters can be found in the Hydra config file `../configs/streaming_experiment.yaml`.

## Fast-Conditioning
Fast-conditoning _i.e._ only variational parameters are optimized.
### Proposed model
To run the fast-conditioning of the proposed model:
```
python online_fc_model.py dataset=bike dataset.dataloader.n_k_folds=10 dataset.dataloader.random_state=33 n_sets=50 model.likelihood.variance=0.1 online_gp.num_mem=0 online_gp.n_steps=2 optimize.train_memory=False load_model_path={trained_model_path} optimize.train_hyperparams=False
```

### Bui *et al.* (2017)
To run the fast-conditioning of the Bui _et al._ (2017) model:

```
python streaming_sgpr_fc.py dataset=bike dataset.dataloader.n_k_folds=10 dataset.dataloader.random_state=33 n_sets=50 optimizer=adam load_model_path={trained_model_path} optimize.iterations=0 optimize.task="regression" model.noise_variance=0.1
```
