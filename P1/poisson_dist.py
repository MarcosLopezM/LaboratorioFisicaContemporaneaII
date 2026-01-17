from pathlib import Path
from scipy.stats import poisson
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

## Figs and Data dirs
figs_dir = Path("figs/")
figs_dir.mkdir(exist_ok=True)
path = Path("Datos/")


## Sorting files function
def avg_sort(path):
    folder = path.parent.name
    folder = folder.removeprefix("Equipo4_f").removesuffix("_v")
    return int(folder)


data = sorted(list(path.glob("Equipo4_*/*.csv")), key=avg_sort)

dfs = []
stats = {
    "normal": {"avg": [], "var": []},
    "verde": {"avg": [], "var": []},
}

for exp in data:
    name = ""
    if exp.parent.name.endswith("_v"):
        name = f"Prom. fotones {avg_sort(exp)} Verde"
    else:
        name = f"Prom. fotones {avg_sort(exp)}"

    dfs.append(pd.read_csv(exp, header=0, names=[name], dtype=int))

df = pd.concat(dfs, axis=1)

## Color map
cmap = plt.get_cmap("tab10")
colors = cmap(np.linspace(0, 1, 8))

for clr, col in enumerate(df):
    fotones = df[col].to_numpy()
    values, counts = np.unique(fotones, return_counts=True)

    ## Normalizing data
    pmf_emp = counts / counts.sum()
    pmf_error = np.sqrt(counts) / counts.sum()

    avg = fotones.mean()
    var = fotones.var(ddof=1, dtype=np.float64)

    if col.endswith("Verde"):
        stats["verde"]["avg"].append(avg)
        stats["verde"]["var"].append(var)
    else:
        stats["normal"]["avg"].append(avg)
        stats["normal"]["var"].append(var)

    std_dev = fotones.std()

    ## Theoretical values
    pmf = poisson.pmf(values, avg)

    label = "$\\bar{n} =$ " + f"{avg:.2f} "
    file_name = "_".join(col.split(" "))

    fig_dist, ax = plt.subplots(figsize=(9, 6))
    ax.bar(
        values,
        pmf_emp,
        yerr=pmf_error,
        capsize=4,
        color=colors[clr],
        label="Datos Experimentales",
        zorder=2,
    )
    ax.plot(values, pmf, "o-", color="black", label=label, zorder=3)
    # plt.title(
    #     f"Distribución de Poisson para $\\bar{{n}} = ${avg:.2f} $\\pm$ {std_dev:.2f}"
    # )
    ax.set_xlabel("Número de fotones $n$")
    ax.set_ylabel("Probabilidad")
    ax.legend()
    fig_dist.savefig(f"figs/{file_name}.png", dpi=300)
    plt.close(fig_dist)

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(
    stats["normal"]["avg"],
    stats["normal"]["var"],
    "o--",
    color="violet",
    label="SPDC",
)

ax.plot(
    stats["verde"]["avg"],
    stats["verde"]["var"],
    "s-",
    color="green",
    label="Atenuado",
)

# Recta de referencia
line_spdc = np.linspace(
    0,
    max(stats["normal"]["avg"]),
    200,
)
line_verde = np.linspace(
    0,
    max(stats["verde"]["avg"]),
    200,
)
ax.plot(line_spdc, line_spdc, "--", label="$v=\\bar{n}$ SPDC")
ax.plot(line_verde, line_verde, "-.", label="$v=\\bar{n}$ Atenuado")
ax.set_xlabel("Promedio de número de fotones")
ax.set_ylabel("Variancia")
ax.legend()
fig.savefig("var_vs_mean.png", dpi=300)

for avg, var in zip(stats["normal"]["avg"], stats["normal"]["var"]):
    print(f"{avg:.2f} & {var:.2f}\\\\")
print("----")
for avg, var in zip(stats["verde"]["avg"], stats["verde"]["var"]):
    print(f"{avg:.2f} & {var:.2f}\\\\")
