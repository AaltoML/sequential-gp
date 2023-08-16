import numpy as np
import matplotlib.pyplot as plt

[[res]] = np.load("results/hotspots_results.npz", allow_pickle=True).values()

betas = {"results": 1.0, "tempered": 0.1}
keys = [
    "results",
    # "tempered",
]

import sys

def print_timings(res, io=sys.stdout):
    label, ts, _, _, _ = res
    mean = ts.mean()
    stddev = ts.std()
    stderr = ts.std() / np.sqrt(ts.size - 1)
    print(f"{label:29s}: {mean:6.2f} +/- {stderr:.2f} (std.err) +/- {stddev:.2f} (std.dev)", file=io)

with open("results/timings.dat", "w") as f:
    for key in keys:
        # print(f"beta={betas[key]}:", file=f)
        for r in res[key]:
            print_timings(r, f)
        print(file=f)


def plot_mean(res, prop):
    label, ts, all_res, mean, stderr = res
    plt.plot(range(100,200), mean[prop], label=label)
    scale = 1
    plt.fill_between(range(100, 200), mean[prop] - scale*stderr[prop], mean[prop] + scale*stderr[prop], alpha=0.3)


prop_titles = {
    # "sampled_acc": "accuracy (sampled)",
    "acc": "Hotspot Accuracy",
    "mse": "Prevalence MSE",
    # "sens": "[sens]"
}


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


import tikzplotlib

for key in keys:
    for prop in ["acc", "mse"]:
        fig = plt.figure()
        # plt.title(f"$\\beta={betas[key]}$")
        plt.xlabel("Steps")
        plt.ylabel(f"{prop_titles[prop]}")

        if prop == "mse":
            # plt.ylim(0.013, 0.0261)  # limits used in Maddox et al. (2021)
            plt.ylim(0.0115, 0.0265)  # our method is too much better to fit
        elif prop == "acc" or prop == "sampled_acc":
            plt.ylim(0.81, 0.89)
            plt.yticks([0.82,0.84,0.86,0.88])

        for r in res[key][:3]:
            plot_mean(r, prop)
        plt.grid()
        plt.legend()
        tikzplotlib_fix_ncols(fig)
        tikzplotlib.save(f"results/hotspots-{key}-{prop}.tex", figure=fig,
                         axis_width=r"\figurewidth",
                         axis_height=r"\figureheight")
