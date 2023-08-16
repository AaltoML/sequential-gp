# Hotspots experiment

This folder contains the experiments to reproduce our comparison against [Maddox et al. (2021)](https://arxiv.org/abs/2110.15172).

As such, `data/civ_data.csv` is a verbatim copy of [their data set](https://github.com/wjmaddox/online_vargp/blob/main/experiments/hotspots/data/civ_data.csv) and `hotspots.py` only has minimal changes from [their original experiment script](https://github.com/wjmaddox/online_vargp/blob/main/experiments/hotspots/hotspots.py) to integrate our own method.

Our only noteworthy departure from Maddox et al. (2021) is that we remove their tempering (changing `beta=0.1` to `beta=1.0`). This change benefits all methods.

`our_tsvgp.py` contains a GPyTorch-compatible implementation of our proposed method (not feature-complete; it only includes those aspects required to run the Hotspots experiment).

`env.yaml` describes a Conda environment with all required dependencies; it can be instantiated using
```bash
conda env create --file env.yaml
```

## Re-run experiments

The experiments can be reproduced by submitting the following jobs on a SLURM cluster:
```bash
sbatch submit_random.sh
sbatch submit_ovc.sh
sbatch submit_ours.sh
```
Each script will spawn the respective experiment for 50 different seeds using SLURM's [Job Array support](https://slurm.schedmd.com/job_array.html).

After all the runs have finished, run
```bash
python extract_results.py
```
to regenerate `results/hotspot_results.npz` (which should be equivalent to the version stored in this repository).

## Re-create figures

To recreate Figure 4 and the timing results in our paper, run
```bash
python visualize_results.py
```
which regenerates the following three files in the `results/` subdirectory:
- `hotspots-results-acc.tex` is the TikZ/pgfplots figure for Hotspot Accuracy.
- `hotspots-results-mse.tex` is the TikZ/pgfplots figure for Prevalence MSE.
- `timings.dat` gives the average run times per step of each method, including standard deviation.
