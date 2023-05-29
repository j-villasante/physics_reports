from json import load
from math import pi, sqrt
from os import path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

current_dir = path.dirname(__file__)
data_dir = path.normpath(f"{current_dir}/../data")
img_dir = path.normpath(f"{current_dir}/../img")

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

f = open(current_dir + "/spec_params.json", encoding="utf8")
spec_params = load(f)


def normal(x, n, sigma, mu):
    return n * np.exp(-(((x - mu) / sigma) ** 2) / 2) / (sigma * sqrt(2 * pi))


def cuad(x, a, b, c):
    return a * x**2 + b * x + c


for detect in spec_params:
    calibration = {
        "channel": [],
        "energy": [],
    }

    for source in detect["sources"]:
        # Load data
        filename = f"{detect['detector']}_{source['source']}"
        filepath = f"{data_dir}/{filename}"
        if not path.isfile(filepath):
            print(f"Missing file at {filepath}")
            continue

        x, y = np.genfromtxt(filepath, delimiter=" ").T

        # Plot spectrum
        fig, ax = plt.subplots()
        ax.plot(x, y, label="Eventos", color="black", linewidth=0.1)

        for b in source["brackets"]:
            # Curve fit
            half_point = b["start"] + ((b["end"] - b["start"]) / 2)
            _x = x[b["start"] : b["end"]]
            _y = y[b["start"] : b["end"]]
            popt = curve_fit(normal, _x, _y, p0=(10**5, 100, half_point))[0]

            # Add mu value for calibration
            calibration["channel"].append(popt[2])
            calibration["energy"].append(b["energy"])

            # Plot curve fit
            ax.plot(
                _x,
                normal(_x, *popt),
                label=f"$\mu={round(popt[2])}, \sigma={round(popt[1])}$",
                color=b["color"],
                linewidth=1.5,
            )

        # Add labels
        ax.set_xlabel("Canal")
        ax.set_ylabel("Eventos")
        ax.legend()

        # Save
        plt.savefig(f"{img_dir}/{filename}.pdf", bbox_inches="tight")

    fig, ax = plt.subplots()
    x = np.array(calibration["channel"])
    y = np.array(calibration["energy"])
    ax.plot(x, y, "o")

    if np.size(y) < 3:
        print(f"Missing channel data for calibration at detector {detect['detector']}")
        continue

    popt = curve_fit(cuad, x, y)[0]
    # Plot curve fit
    _x = np.linspace(np.amin(x), np.amax(x))
    ax.plot(
        _x,
        cuad(_x, *popt),
        label=f"$a={round(popt[0], 5)}, b={round(popt[1], 2)}, c={round(popt[2], 2)}$",
        color="black",
    )
    ax.legend()
    plt.savefig(f"{img_dir}/cal_{detect['detector']}.pdf")
