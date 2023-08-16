# Memory-based dual Gaussian processes for sequential learning

This repository is the official implementation of the methods in the publication:

* P.E. Chang, P. Verma, S.T. John, A. Solin, and M.E. Khan (2023). **Memory-based dual Gaussian processes for sequential learning**. In *International Conference on Machine Learning (ICML)*. [[arXiv]](https://arxiv.org/abs/2306.03566)

Sequential learning with Gaussian processes (GPs) is challenging when access to past data is limited, for example, in continual and active learning. In such cases, errors can accumulate over time due to inaccuracies in the posterior, hyperparameters, and inducing points, making accurate learning challenging. Here, we present a method to keep all such errors in check using the recently proposed dual sparse variational GP. Our method enables accurate inference for generic likelihoods and improves learning by actively building and updating a memory of past data. We demonstrate its effectiveness in several applications involving Bayesian optimization, active learning, and continual learning. 

## Environment

We recommend setting up a [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment for running the experiments. The code base is tested on a machine with a Ubuntu 22.04 distribution, CUDA11.6, and conda 23.1.0.
```shell
conda create -n sequential-gp python==3.8
conda activate sequential-gp
```

Within the virtual environment, install the dependencies by running
```shell
pip install -r requirements.txt
```

(Note that the `hotspots` experiment has its own environment and setup instructions.)

## Experiments

There are a series of experiments which are organized inside the `experiments` folder as separate sub-folders. 
Each experiment sub-folder has their respective readme files with instructions on how to run the particular experiment.

## Data sets

The datasets used for banana, UCI, and magnetometer experiments are available in `experiments/data/` directory.
The original source of the data sets are:

* **Banana:** https://github.com/thangbui/streaming_sparse_gp
* **UCI:** https://archive.ics.uci.edu/datasets
* **Magnetometer:** https://github.com/AaltoML/magnetic-data

## Contributing

For all correspondence, please contact [paul.chang@aalto.fi](mailto:paul.chang@aalto.fi) 
or [prakhar.verma@aalto.fi](mailto:prakhar.verma@aalto.fi).


## License

This software is provided under the [MIT license](LICENSE).
