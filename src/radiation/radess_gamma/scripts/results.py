from json import load
from math import pi, sqrt, ceil
from os import path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tabulate import tabulate

current_dir = path.dirname(__file__)
data_dir = path.normpath(f"{current_dir}/../data")
img_dir = path.normpath(f"{current_dir}/../img")
resolution = False

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rc("font", size=12)

f = open(current_dir + "/spec_params.json", encoding="utf8")
spec_params = load(f)


def normal(x, n, sigma, mu):
    return n * np.exp(-(((x - mu) / sigma) ** 2) / 2) / (sigma * sqrt(2 * pi))

def linear(x, a, b):
    return a * x + b

def cuad(x, a, b, c):
    return a * x**2 + b * x + c


def cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


def format_name(source: str) -> str:
    (n, e) = source.split("_")
    return r"${}^{" + str(e) + r"}" + n.capitalize() + "$"


def efficiency(counts: int, y: int, time: int) -> int:
    return counts / (10**5 * y * time)


for detect in spec_params:
    calibration = {
        "channel": [],
        "energy": [],
        "error": [],
        "source": [],
        "efficiency": [],
        "resolution": [],
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
            (n, sigma, mu) = curve_fit(normal, _x, _y, p0=(10**5, 100, half_point))[0]

            # Add mu value for calibration
            calibration["channel"].append(mu)
            calibration["energy"].append(b["energy"])
            calibration["error"].append(sigma)
            calibration["source"].append(source["source"])
            calibration["efficiency"].append(
                efficiency(
                    np.sum(y[ceil(mu - (2.355 * sigma)) : ceil(mu + (2.355 * sigma))]),
                    b["yield"],
                    b["time"],
                )
            )
            calibration["resolution"].append(
                (2.355 * sigma) / mu,
            )

            # Plot curve fit
            ax.plot(
                _x,
                normal(_x, n, sigma, mu),
                label=f"$\mu={round(mu)}, \sigma={round(sigma)}$",
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
    error = np.array(calibration["error"])
    ax.errorbar(x, y, yerr=error, fmt="o")

    if np.size(y) < 3:
        print(f"Missing channel data for calibration at detector {detect['detector']}")
        continue

    popt = curve_fit(linear, x, y)[0]
    # Plot curve fit
    _x = np.linspace(np.amin(x), np.amax(x))
    ax.plot(
        _x,
        linear(_x, *popt),
        label=f"$a={popt[0]:.03g}, b={popt[1]:.03g}$",
        color="black",
    )

    for i, source in enumerate(calibration["source"]):
        ax.annotate(format_name(source), (x[i] + 20, y[i] + 20))
    ax.legend()
    ax.set_xlabel("Canal")
    ax.set_ylabel("Energía (keV)")
    plt.savefig(f"{img_dir}/cal_{detect['detector']}.pdf")

    # Print expected values
    print(f"Expected values of energy for detector {detect['detector']}")
    for i, source in enumerate(calibration["source"]):
        print(f"{source}: {linear(x[i], *popt)}")
    
    # Unknown source
    if detect["detector"] == "nai_33":
        fig, ax = plt.subplots()
        x, y = np.genfromtxt(f"{data_dir}/unknown_source", delimiter=" ").T
        x = linear(x, *popt)
        ax.plot(x, y, label="Eventos", color="black", linewidth=0.1)
        plt.savefig(f"{img_dir}/unknown_source.pdf")

    # plot efficiency
    fig, ax = plt.subplots()
    x = np.array(calibration["energy"])
    y = np.array(calibration["efficiency"])

    # Plot curve fit
    popt = curve_fit(cubic, x, y)[0]
    _x = np.linspace(np.amin(x), np.amax(x))
    ax.plot(
        _x,
        cubic(_x, *popt),
        label=f"$a={popt[0]:.03g}, b={popt[1]:.03g}, c={popt[2]:.03g}, d={popt[3]:.03g}$",
        color="black",
    )
    ax.plot(x, y, "o")

    for i, source in enumerate(calibration["source"]):
        ax.annotate(format_name(source), (x[i] + 20, y[i] - 0.0025))

    ax.legend()
    ax.set_xlabel("Energía (keV)")
    ax.set_ylabel("Eficiencia")
    plt.savefig(f"{img_dir}/{detect['detector']}_efficiency.pdf")

    if not resolution:
        print(f"Plotted energy calibration for detector {detect['detector']}")
        # plot energy resolution
        fig, ax = plt.subplots()
        x = np.array(calibration["energy"])
        y = np.array(calibration["resolution"])
        # Plot curve fit
        popt = curve_fit(linear, x, y)[0]
        _x = np.linspace(np.amin(x), np.amax(x))
        ax.plot(
            _x,
            linear(_x, *popt),
            label=f"$a={popt[0]:.03g}, b={popt[1]:.03g}$",
            color="black",
        )
        ax.plot(x, y, "o")

        for i, source in enumerate(calibration["source"]):
            ax.annotate(format_name(source), (x[i], y[i]))

        ax.legend()
        ax.set_xlabel("Energía (keV)")
        ax.set_ylabel("Resolución (keV)")
        plt.savefig(f"{img_dir}/resolution_vs_energy.pdf")
        resolution = True

        print(tabulate({
            "Source": calibration["source"],
            "Resolution": calibration["resolution"],
            "Energy": calibration["energy"],
        }, headers="keys"))
