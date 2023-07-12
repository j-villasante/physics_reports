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

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rc('font', size=12)

areas = []

# Plot calibration
def linear(x, a, b):
    return a * x + b

def inverse_linear(y, a, b):
    return round((y - b) / a)

def normal(x, n, sigma, mu):
    return n * np.exp(-(((x - mu) / sigma) ** 2) / 2) / (sigma * sqrt(2 * pi))

fig, ax = plt.subplots()
x = np.array([565.26, 601.77, 636.26])
y = np.array([5155.5, 5486, 5901.7])
popt = curve_fit(linear, x, y)[0]
_x = np.linspace(np.amin(x), np.amax(x))
ax.plot(
    _x,
    linear(_x, *popt),
    label=f"$E = {popt[0]:.03g}c {popt[1]:.03g}$",
)
ax.plot(x, y, "o")
ax.legend()
ax.set_xlabel("Canal")
ax.set_ylabel("Energía (keV)")
plt.savefig(f"{img_dir}/calibration.pdf", bbox_inches="tight")

f = open(current_dir + "/spec_params.json", encoding="utf8")
spec_params = load(f)

table = {
    "name": [],
    "canal": [],
    "energy": [],
    "fwhm": [],
    "net_area": [],
}

for image in spec_params:
    # Plot spectrum
    fig, ax = plt.subplots()

    if "file" in image:
        x, y = np.genfromtxt(f"{data_dir}/{image['file']}", delimiter=",").T
        if image["use_calibration"]:
            x = linear(x, *popt)

            for peak in (image["peaks"] if "peaks" in image else []):
                half_point = peak["start"] + ((peak["end"] - peak["start"]) / 2)
                _x = x[peak["start"] : peak["end"]]
                _y = y[peak["start"] : peak["end"]]
                p0_sigma = peak["sigma"] if "sigma" in peak else 600
                (n, sigma, mu) = curve_fit(normal, _x, _y, p0=(10**5, p0_sigma, half_point))[0]
                delta = 2.355 * sigma

                # ax.plot(
                #     _x,
                #     normal(_x, n, sigma, mu),
                #     label=f"$\mu={round(mu)}, \sigma={round(sigma)}$",
                # )
                # ax.plot(np.array([mu - delta, mu - delta]), np.array([0, 100]))
                # ax.plot(np.array([mu + delta, mu + delta]), np.array([0, 100]))
                
                area = np.sum(y[inverse_linear(mu - delta, *popt) : inverse_linear(mu + delta, *popt)])
                if "area" in peak and peak["area"]:
                    areas.append(area)
                    ax.annotate(f"$A = {area:.3}$", (mu + peak["dx"], peak["y"]))
                else:
                    ax.annotate(f"{mu:.0f}keV", (mu + peak["dx"], peak["y"]))
                
                table["name"].append(image["name"])
                table["canal"].append(inverse_linear(mu, *popt))
                table["energy"].append(mu)
                table["fwhm"].append(delta)
                table["net_area"].append(f"a{area:.3g}")

        ax.plot(x, y, linewidth=0.75)
    else:
        for spectrum in image["files"]:
            x, y = np.genfromtxt(f"{data_dir}/{spectrum['filename']}", delimiter="\t").T
            ax.plot(x, y, label=spectrum["label"], linewidth=0.5)
    
    # Add labels
    ax.set_xlabel("Energía (keV)" if "use_calibration" in image and image["use_calibration"] else "Canal")
    ax.set_ylabel("Eventos")

    if "files" in image:
        ax.legend()

     # Save
    plt.savefig(f"{img_dir}/{image['name']}.pdf", bbox_inches="tight")


print(tabulate(table, headers="keys"))