import torch

import numpy as np

import glob
import re

num_re = re.compile(r".*_([0-9]*)_.*")

def get_all_runs_available(fn_globs):
    all_runs_available = set()
    for fn_glob in fn_globs:
        files = glob.glob(fn_glob)
        nums = [int(num_re.match(fn)[1]) for fn in files]
        print(fn_glob, len(nums))
        if not all_runs_available:
            all_runs_available = set(nums)
        else:
            all_runs_available = all_runs_available.intersection(set(nums))

    all_runs_available = sorted(all_runs_available)
    return all_runs_available

from collections import namedtuple

Run = namedtuple("Run", ["label", "results_pattern", "output_pattern"])

class Data:
    beta = 1.0
    _results_base = "output_dir/civ_ind_svgp_{i}_AMD_"
    runs = [
        Run("Random",         _results_base+"random.pt",  "./hotspots_random_{i}.out"),
        Run("Entropy (OVC)",  _results_base+"entropy.pt", "./hotspots_ovc_{i}.out"),
        Run("Entropy (Ours)", _results_base+"tsvgp.pt",   "./hotspots_ours_{i}.out"),
    ]

class DataTempered:
    beta = 0.1
    _results_base = "beta0.1/output_dir/civ_ind_svgp_{i}_AMD_"
    runs = [
        Run("random",         _results_base+"random.pt",  "./beta0.1/hotspots_random_{i}.out"),
        Run("entropy (OVC)",  _results_base+"entropy.pt", "./beta0.1/hotspots_ovc_{i}.out"),
        Run("entropy (Ours)", _results_base+"tsvgp.pt",   "./beta0.1/hotspots_ours_{i}.out"),
    ]

time_re = re.compile(r"time = ([0-9.]*)\.memory")

def parse_time(fn):
    with open(fn) as f:
        matches = time_re.findall(f.read())
    return np.array(list(map(float, matches)))

def get_timings(fnames):
    return np.array([parse_time(fn) for fn in fnames])

def load_results(fnames):
    props = ("acc", "mse", "sens", "sampled_acc")
    all_res = {prop: [] for prop in props}
    for fn in fnames:
        d = torch.load(fn, map_location=torch.device('cpu'))
        res = {}
        [
            res["acc"], res["mse"], res["sens"], res["sampled_acc"]
        ] = [
            hotspot_acc_list,
            hotspot_mse_list,
            hotspot_sens_list,
            hotspot_sampled_acc_list
        ] = d["results"]
        for prop in props:
            all_res[prop].append(res[prop].numpy())
    for prop in props:
        all_res[prop] = np.array(all_res[prop])
    mean = {prop: all_res[prop].mean(axis=0) for prop in props}
    stderr = {prop: all_res[prop].std(axis=0) / np.sqrt(all_res[prop].shape[0] - 1) for prop in props}
    return all_res, mean, stderr

#Results = namedtuple("Results", ["label", "timings", "all_res", "mean", "stderr"])

def process(dat):
    all_runs_available = get_all_runs_available([run.results_pattern.format(i="*") for run in dat.runs])
    results = []
    for run in dat.runs:
        ts = get_timings([run.output_pattern.format(i=i) for i in all_runs_available])
        all_res, mean, stderr = load_results([run.results_pattern.format(i=i) for i in all_runs_available])
        results.append((run.label, ts, all_res, mean, stderr))
    return results

def main():
    np.savez("results/hotspots_results.npz", [dict(
        results=process(Data()),
        # tempered=process(DataTempered()),
    )])

if __name__ == "__main__":
    main()
