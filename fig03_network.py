# %%
import pickle

import colormaps as cm
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from xover import inversion as xinv

from g3m.real_dic import cruises


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

xovers_before = xinv.adjust_xovers(fff.xovers_adjusted, -fff.adjustments)
offsets_final_f = xinv.offsets_weighted(fff.xovers_adjusted, fff.weights)
offsets_final_g = xinv.offsets_weighted(ffg.xovers_adjusted, ffg.weights)
offsets_before = xinv.offsets_weighted(xovers_before, fff.weights)

ub = xinv.offset_uncertainties(
    xovers_before, fff.weights, offsets_before, fff.dof, fff.t_crit
)
uf = xinv.offset_uncertainties(
    fff.xovers_adjusted, fff.weights, offsets_final_f, fff.dof, fff.t_crit
)
ug = xinv.offset_uncertainties(
    ffg.xovers_adjusted, ffg.weights, offsets_final_g, ffg.dof, ffg.t_crit
)

# %% Graphs
fig_opts = [
    dict(
        label="(a) FF$_1$, all cruises adjusted",
        ff=fff,
        node_color=fff.adjustments,
    ),
    dict(
        label="(b) FF$_2$, minimum adjustment cut-off",
        ff=ffg,
        node_color=ffg.adjustments,
    ),
]

fig, axs = plt.subplots(figsize=(12, 14), nrows=2)
for i in range(len(fig_opts)):
    fig_opt = fig_opts[i]
    ax = axs[i]
    label = fig_opt["label"]
    ff = fig_opt["ff"]
    node_color = fig_opt["node_color"]
    vmin = -12
    vmax = 12
    weights = ff.weights
    graph = nx.from_numpy_array(weights > 0)
    edge_weights = {e: np.sqrt(1 + weights[*e]) / 5 for e in graph.edges}
    nx.set_edge_attributes(graph, edge_weights, name="weight")
    cdict = nx.current_flow_betweenness_centrality(graph, weight="weight")
    node_size = [5 + 8 * np.abs(n) for n in node_color]
    pos = nx.nx_agraph.graphviz_layout(graph, prog="neato")
    pos = {k: (v[1], -v[0]) for k, v in pos.items()}
    pos_min = np.min(np.array(list(pos.values())), axis=0)
    pos_max = np.max(np.array(list(pos.values())), axis=0)
    cmap = cm.temps_r
    nodes = nx.draw_networkx_nodes(
        graph,
        ax=ax,
        pos=pos,
        node_size=node_size,
        node_color=node_color,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    nx.draw_networkx_edges(
        graph,
        ax=ax,
        pos=pos,
        node_size=node_size,
        alpha=0.3,
        edge_color="xkcd:dark",
        width=[edge_weights[e] * 1.5 for e in graph.edges],
    )

    cax_pos = [0.78, 0.15, 0.14, 0.05]
    cax = ax.inset_axes(cax_pos)
    cax.set_facecolor("none")
    cx = np.arange(vmin, vmax + 1, 3)
    cy = np.ones_like(cx)
    cax.scatter(
        cx,
        cy,
        c=cx,
        linewidths=0.5,
        s=5 + 8 * np.abs(cx),
        clip_on=False,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    cax.set_xlim(vmin, vmax)
    cax.set_xticks([-12, -6, 0, 6, 12])
    cax.set_xlabel("Adjustment / µmol kg$^{-1}$")
    cax.get_yaxis().set_visible(False)
    for p in ["top", "right", "left"]:
        cax.spines[p].set_visible(False)

    ax.set_xlim(pos_min[0] - 10, pos_max[0] + 10)
    ax.set_ylim(pos_min[1] - 10, pos_max[1] + 10)
    ax.set_axis_off()
    txt = dict(
        transform=ax.transAxes,
        fontsize=15,
        ha="center",
        va="center",
        c="xkcd:dark",
    )
    ax.text(0.06, 0.82, "Pacific", **txt)
    ax.text(0.86, 0.41, "Atlantic", **txt)
    ax.text(0.46, 0.07, "Indian", **txt)
    ax.text(0.53, 0.59, "Southern", **txt)
    ax.text(
        0.012,
        0.92,
        label,
        transform=ax.transAxes,
        fontsize=14,
    )
fig.tight_layout()
fig.savefig("figures/fig03_network.pdf")
