from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import mode
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

N_counts = counts.sum()
mean = np.sum(counts * time) / N_counts  # Tiempo promedio
variance = np.sum(counts * (time - mean) ** 2) / (N_counts - 1)
std = np.sqrt(variance)
mean_err = std / np.sqrt(N_counts)

# Time in ns
mean_ns = mean * 1e9
mean_err_ns = mean_err * 1e9

## Barras de error
errors = np.sqrt(counts)
errors[errors == 0.0] = 1

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

print(f"c = {light_speed:.2e} ± {light_speed_err:.2e} m/s")

##### Cálculo de la distancia con la moda de la distribución ####
idx_max = np.argmax(counts)
time_mode = time[idx_max]
dt = np.mean(np.diff(time))
time_mode_err = dt / 2
time_mode_ns = time_mode / 1e-9
time_mode_err_ns = time_mode_err / 1e-9

# Velocidad de la luz
ls_mode = distance / time_mode
ls_mode_err = ls_mode * np.sqrt(
    (total_err / distance) ** 2 + (time_mode_err / time_mode) ** 2
)
print("Valor de la rapidez de la luz usando la moda")
print(f"c = {ls_mode:.2e} ± {ls_mode_err:.2e} m/s")
print(time_mode, time_mode_ns)
print(time_mode_err, time_mode_err_ns)

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
    (
        rf"$\mu = \qty{{{mean_ns:.2f} \pm {mean_err_ns:.2f}}}{{\s}}$"
        "\n"
        rf"$\textrm{{moda}} = \qty{{{time_mode_ns:.2f} \pm {time_mode_err_ns:.2f}}}{{\s}}$"
    ),
    transform=ax.transAxes,
    va="top",
)
ax.set_xlabel(r"Tiempo ($\unit{\s}$)")
ax.set_ylabel("Coincidencias")
fig.savefig(figs_dir / "histogram.pdf", dpi=400)
plt.show()
