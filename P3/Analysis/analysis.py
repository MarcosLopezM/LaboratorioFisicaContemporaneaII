from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "font.size": 12,
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
N_counts = counts.sum()
mean = np.sum(counts * time) / N_counts
std = np.sqrt(np.sum(counts * (time - mean) ** 2) / N_counts)
mean_err = std / np.sqrt(N_counts)
std_err = std / np.sqrt(2 * N_counts)


print(f"Mean $\\mu$ = {mean * 1e9:.4f} $\\pm$ {mean_err * 1e9:.4f} ns")
print(f"$\\sigma$ = {std * 1e9:.2f} $\\pm$ {std_err * 1e9:.2f} ns")

fig, ax = plt.subplots(figsize=(9, 6))
bin_width = np.diff(df["Time"]).mean()
ax.bar(
    df["Time"],
    df["Ampl"],
    width=bin_width,
    align="center",
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

ax.set_xlabel("Time (ns)")
ax.set_ylabel("Coincidencias")
fig.savefig(figs_dir / "histogram.pdf", dpi=400)
