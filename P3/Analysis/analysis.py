from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "font.size": 12,
        "axes.labelsize": 12,
        "text.latex.preamble": "\n".join(
            [
                r"\usepackage{siunitx}",
                r"\sisetup{separate-uncertainty-units=single,uncertainty-mode=separate}",
            ]
        ),
    }
)

BASE_DIR = Path(__file__).resolve().parent
path = BASE_DIR / "Datos"

figs_dir = Path("figs/")
figs_dir.mkdir(exist_ok=True)

df = pd.read_csv(
    path / "F1hist1100000.csv",
    header=4,
    usecols=["Time", "Ampl"],
)

time = df["Time"].to_numpy()
counts = df["Ampl"].to_numpy()

errors = np.sqrt(counts)
errors[errors == 0.0] = 1
N_counts = counts.sum()
mean = np.sum(counts * time) / N_counts
std = np.sqrt(np.sum(counts * (time - mean) ** 2) / N_counts)
mean_err = std / np.sqrt(N_counts)
mean_ns = mean * 1e9
mean_err_ns = mean_err * 1e9

# Velocidad de la luz
distances = np.array([135, 135, 135.8, 135.3]) / 100
distances_err = np.ones(4) * (0.1 / 2) / 100
distance = distances.mean()
distance_err = np.sqrt(np.sum(distances_err**2)) / distances.size
stat_err = np.std(distances, ddof=1) / np.sqrt(distances.size)
total_err = np.sqrt(distance_err**2 + stat_err**2)

light_speed = distance / mean
light_speed_err = light_speed * np.sqrt(
    (total_err / distance) ** 2 + (mean_err / mean) ** 2
)

print(f"c = {light_speed:.3e} ± {light_speed_err:.3e} m/s")

fig, ax = plt.subplots(figsize=(9, 6))
ax.step(
    time,
    counts,
    where="mid",
    linewidth=0.8,
    color="limegreen",
)
ax.errorbar(
    time,
    counts,
    yerr=errors,
    fmt="none",
    ecolor="black",
    elinewidth=1,
    capsize=2,
)
ax.text(
    0.05,
    0.95,
    rf"$\mu = \qty{{{mean_ns:.4f} \pm {mean_err_ns:.4f}}}{{\ns}}$",
    transform=ax.transAxes,
    va="top",
)
ax.set_xlabel(r"Tiempo ($\unit{\s}$)")
ax.set_ylabel("Coincidencias")
fig.savefig(figs_dir / "histogram.pdf", dpi=400)
plt.show()
