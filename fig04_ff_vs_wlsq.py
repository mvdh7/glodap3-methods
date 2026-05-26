# %%
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams

from g3m.real_dic import cruises


rcParams["font.size"] = 8

ffexpo = pd.read_csv("data/DIC_KeyInfo_null.csv")
ffexpo["regions"] = ""
for i, row in ffexpo.iterrows():
    ffexpo.loc[i, "regions"] = cruises.regions[
        cruises.EXPOCODE == row.EXPOCODE
    ].values
assert not (ffexpo.regions == "").any()
wlsq = pd.read_csv("data/WLSQ.csv")
with open("data/ff_full_new.pkl", "rb") as f:
    fff = pickle.load(f)
with open("data/ff_glodapv3_new.pkl", "rb") as f:
    ffg = pickle.load(f)
wlsq_adjustments = np.array(
    [wlsq.Adjustments[wlsq.expocode == e] for e in ffexpo.EXPOCODE]
)

# %%
fig, axs = plt.subplots(figsize=(3.3, 5), nrows=2)
ax = axs[0]
ax.text(0, 1.04, "(a)", transform=ax.transAxes)
ax.scatter(
    fff.adjustments,
    wlsq_adjustments,
    s=20,
    c="xkcd:dark",
    alpha=0.7,
    edgecolor="none",
)
ax.set_ylabel("WLSQ adjustment / µmol kg$^{-1}$")
ax = axs[1]
ax.text(0, 1.04, "(b)", transform=ax.transAxes)
ax.scatter(
    fff.adjustments,
    ffg.adjustments,
    s=20,
    c="xkcd:dark",
    alpha=0.7,
    edgecolor="none",
)
ax.set_ylabel("FF$_2$ adjustment / µmol kg$^{-1}$")
for ax in axs:
    ax.axline((0, 0), slope=1, c="k", lw=0.8)
    ax.grid(alpha=0.2)
    ax.set_aspect(1)
    ax.set_xlabel("FF$_1$ adjustment / µmol kg$^{-1}$")
    ax.set_xlim(-10, 20)
    ax.set_ylim(-10, 20)
    ax.set_xticks(range(-10, 25, 5))
    ax.set_yticks(range(-10, 25, 5))
fig.tight_layout()
fig.savefig("figures/fig04_ff_vs_wlsq.pdf")
