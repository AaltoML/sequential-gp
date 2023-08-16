#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=5000M
#SBATCH --output=./hotspots_random_%a.out
#SBATCH --array=1-50

module load miniconda
source activate ovcexperiment

mkdir -p output_dir

python hotspots.py --seed=$SLURM_ARRAY_TASK_ID --n_batch=100 --num_init=100 \
    --beta=1.0 --loss=elbo --dataset=civ --random \
    --output=output_dir/civ_ind_svgp_${SLURM_ARRAY_TASK_ID}_AMD_random.pt
