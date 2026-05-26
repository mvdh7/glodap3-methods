# %%
import pickle

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt, rcParams
from scipy import stats
from xover import inversion as xinv

import g3m.simulate as simulate


rcParams["font.size"] = 8

rerun_ff = False  # set to True to redo the FF fitting (slow!)
rng = np.random.default_rng(1)
c1 = 280
c2 = 25
c3 = 55
c4 = 70
c5 = 50
c6 = 50

px, py = rng.uniform(size=(2, c1))
px2, py2 = rng.uniform(low=1.1, high=1.3, size=(2, c2))
px3, py3 = rng.uniform(low=-0.5, high=0, size=(2, c3))
px4, py4 = rng.uniform(low=1.4, high=1.8, size=(2, c4))
px5 = rng.uniform(low=1.2, high=1.5, size=c5)
py5 = rng.uniform(low=-0.5, high=-0.2, size=c5)
px6 = rng.uniform(low=-0.5, high=-0.2, size=c6)
py6 = rng.uniform(low=1.2, high=1.5, size=c6)
px = np.concat((px, px2, px3, px4, px5, px6))
py = np.concat((py, py2, py3, py4, py5, py6))
clusters = np.array(
    [
        *([1] * c1),
        *([2] * c2),
        *([3] * c3),
        *([4] * c4),
        *([5] * c5),
        *([6] * c6),
    ]
)
n_cruises = len(px)  # this is the final number of cruises
# Calculate distances between points in p-space
distances = np.sqrt((px - np.vstack(px)) ** 2 + (py - np.vstack(py)) ** 2)
# Simulate degrees
mn1 = stats.multivariate_normal(
    rng.uniform(low=0.1, high=0.9, size=2), [[0.04, 0], [0, 0.04]]
)
mn2 = stats.multivariate_normal(
    rng.uniform(low=0.1, high=0.9, size=2), [[0.01, 0], [0, 0.01]]
)
degree_shift = 10
degrees = np.ceil(
    mn1.pdf(np.dstack([px, py])) * n_cruises / 30
    + mn2.pdf(np.dstack([px, py])) * n_cruises / 150
    + degree_shift
    + rng.integers(low=-degree_shift, high=degree_shift, size=n_cruises)
).astype(int)
# Turn some extra nodes into degree 1
degrees[rng.integers(low=0, high=n_cruises, size=10)] = 1
# Reduce the density of connections for one section
degrees[px < 0] = np.ceil(degrees[px < 0] / 3).astype(int)
# If any have ended up with degree zero (or less) somehow, set it to 1
degrees[degrees < 1] = 1
# Multiply a few by 1.5 and a few by 2.5
ix = rng.integers(low=0, high=n_cruises, size=200)
degrees[ix] = np.ceil(degrees[ix] * 1.5).astype(int)
ix = rng.integers(low=0, high=n_cruises, size=50)
degrees[ix] = np.ceil(degrees[ix] * 2.5).astype(int)
# Assign cruise names (expocodes)
# https://github.com/dominictarr/random-name/blob/master/first-names.txt
with open("data/first-names.txt", "r") as f:
    names_all = f.read().splitlines()
expocodes = []
for n in range(n_cruises):
    name = names_all.pop(rng.integers(len(names_all)))
    expocodes.append(name)
expocodes = np.array(expocodes)
dg = simulate.Degrees(expocodes, degrees, distances, px, py, n_cruises)

graph = simulate.generate_graph(degrees, distances, expocodes, rng)

# Make everything map to graph.nodes
node_order = [list(expocodes).index(n) for n in graph.nodes]
expocodes = expocodes[node_order]
px = px[node_order]
py = py[node_order]
clusters_raw = clusters.copy()
clusters = clusters[node_order]
degrees = degrees[node_order]

# Reduce between-cluster connections
cluster_links_allowed = [
    [1, 1],
    [2, 2],
    [3, 3],
    [4, 4],
    [5, 5],
    [6, 6],
    [1, 2],
    [1, 3],
    [1, 5],
    [1, 6],
]
edges_to_remove = []
for e in graph.edges:
    e0 = list(expocodes).index(e[0])
    e1 = list(expocodes).index(e[1])
    clusters_here = [clusters[e0], clusters[e1]]
    if clusters_here not in cluster_links_allowed:
        edges_to_remove.append(e)
graph.remove_edges_from(edges_to_remove)

# Make sure all clusters are still connected though
# NOTE CAREFUL---this will break with a different random seed!
c16 = expocodes[(clusters == 1) & (px < 0.1) & (py > 0.95)]
c61 = expocodes[(clusters == 6).nonzero()[0][:3]]
graph.add_edge(c16[0], c61[0])
graph.add_edge(c16[0], c61[1])
graph.add_edge(c16[1], c61[1])
graph.add_edge(c16[1], c61[2])

c24 = expocodes[(clusters == 2).nonzero()[0][:3]]
c42 = expocodes[(clusters == 4).nonzero()[0][:3]]
graph.add_edge(c24[0], c42[0])
graph.add_edge(c24[0], c42[1])
graph.add_edge(c24[0], c42[2])
graph.add_edge(c24[1], c42[2])
graph.add_edge(c24[2], c42[0])
graph.add_edge(c24[2], c42[2])

c35 = expocodes[(clusters == 3).nonzero()[0][0]]
c53 = expocodes[(clusters == 5).nonzero()[0][0]]
graph.add_edge(c35, c53)

# Initialise cruise offsets (i.e., the true errors that we want to find)
n_big_offsets = 30
n_no_offset = 30
scale_offsets = 3
scale_big_offsets = 8
trend_type = "constant"
trend = 0
trend_se_factor = 1
trend_min = np.inf
weights_s = 0.5
weights_loc = 0
weights_scale = 0.25
randomise_weights = True
crossover_noise_scale = 0

offsets_true = rng.normal(
    scale=scale_offsets, size=n_cruises
)  # "true" offsets
# Add a few bigger offsets
offsets_true[rng.integers(low=0, high=n_cruises, size=n_big_offsets)] += (
    rng.normal(scale=scale_big_offsets, size=n_big_offsets)
)
# Add some zero offsets
offsets_true[rng.integers(low=0, high=n_cruises, size=n_no_offset)] = 0

# Set weights
weights = nx.adjacency_matrix(graph, nodelist=dg.expocodes).toarray()
# Check that the orders of rows & columns in the the weights matrix are correct
graph_degree = np.array([graph.degree[e] for e in dg.expocodes])
assert (graph_degree == weights.sum(axis=0)).all()
assert (graph_degree == weights.sum(axis=1)).all()
if randomise_weights:
    weights = weights * stats.lognorm.rvs(
        weights_s,
        loc=weights_loc,
        scale=weights_scale,
        size=weights.shape,
        random_state=rng,
    )
    # Make it symmetrical
    i_lower = np.tril_indices(n_cruises, -1)
    weights[i_lower] = weights.T[i_lower]

# Calculate `trends_true` - true rate of change per cruise per year
# (year starts at zero)
dates = rng.uniform(low=0, high=50, size=n_cruises)
if trend_type.lower() == "constant":
    trends_true = np.full_like(dates, trend)  # constant trend
else:
    trends_true = py * trend  # varying trend across the field
date_diffs = (weights > 0) * (np.vstack(dates) - dates)
trend_diffs = (weights > 0) * (
    np.vstack(dates * trends_true) - dates * trends_true
)
# ^ `trend_diffs` gives *apparent* offsets between cruises due to trends

# Calculate the apparent crossovers
crossovers = np.vstack(offsets_true) - offsets_true + trend_diffs
crossovers = np.where(weights == 0, np.nan, crossovers)

# Add crossover noise (frepresenting variability within a cruise)
crossover_noise = rng.normal(
    scale=crossover_noise_scale / np.sqrt(2),
    size=crossovers.shape,
)
crossover_noise = crossover_noise + crossover_noise.T
lc = np.tril_indices(crossovers.shape[0], k=-1)
crossover_noise[lc] = -crossover_noise[lc]
crossovers = crossovers + crossover_noise

# `trb.slope` is the apparent trends before inversion computed from
# crossovers ("trb" = "trends before")
trb = xinv.get_trends_wls(crossovers, weights, date_diffs, min_xovers=5)
# `trends` is the subset of `trb` that we choose to believe
# NOTE that this already looks quite different from `trends_true`
trends = np.where(
    (np.abs(trb.slope) < trb.slope_se * trend_se_factor)
    | (np.abs(trb.slope) < trend_min),
    0,
    trb.slope,
)

# `crossovers_dt_sym` is trend-corrected and skew-symmetric,
#  ready for inversion
crossovers_dt = crossovers - date_diffs * trends
crossovers_dt_sym = (crossovers_dt - crossovers_dt.T) / 2

offsets_apparent = xinv.offsets_weighted(crossovers, weights)

xo = simulate.Crossovers(
    dg,
    graph,
    offsets_true,
    offsets_apparent,
    weights,
    dates,
    date_diffs,
    trends_true,
    trend_diffs,
    trb,
    trends,
    crossovers,
    crossovers_dt,
    crossovers_dt_sym,
    crossover_noise,
)
if rerun_ff:
    ff = xinv.furthest_first(
        xo.crossovers,
        niter=1_000_000,
        weights=xo.weights,
    )
    with open("data/regional_clusters.pkl", "wb") as f:
        pickle.dump(ff, f)
else:
    with open("data/regional_clusters.pkl", "rb") as f:
        ff = pickle.load(f)


# %% Visualise connections and adjustments
cmap = plt.cm.viridis
pos = {n: (px[i], py[i]) for i, n in enumerate(graph.nodes)}
fig, axs = plt.subplots(nrows=2, figsize=[3.3, 4.5])
ax = axs[0]
nx.draw_networkx_nodes(
    graph,
    pos=pos,
    node_size=5,
    node_color=clusters,
    ax=ax,
    edgecolors="#1c243133",  # based on xkcd:dark
    linewidths=0.2,
    cmap=cmap,
)
nx.draw_networkx_edges(
    graph,
    pos=pos,
    node_size=10,
    alpha=0.2,
    width=0.8,
    edge_color="xkcd:dark",
    ax=ax,
)
ax.text(0, 1.03, "(a)", transform=ax.transAxes)
ax = axs[1]
cluster_colours = {c: cmap((c - 1) / 5) for c in range(7)}
for c in np.unique(ff.steps.index):
    L = ff.steps.index == c
    iterations = np.array([0, *(np.nonzero(L)[0] + 1)])
    nudges = np.array([0, *ff.steps.adjustment[L]])
    adjustment = np.cumsum(nudges)
    iters = []
    adjs = [0]
    for i in range(len(iterations) - 1):
        iters.append(iterations[i])
        iters.append(iterations[i + 1])
        adjs.append(adjustment[i])
        adjs.append(adjustment[i + 1])
    adjs.append(adjustment[-1])
    iters.append(iterations[-1])
    iters.append(len(ff.steps.index))
    iters = np.array(iters)
    isplit = 10_000
    imax = 1_000_000
    iters = np.where(
        iters < isplit,
        iters,
        isplit + isplit * (iters - isplit) / (imax - isplit),
    )
    adjs = np.array(adjs)
    kwargs = dict(
        # c=node_color[c],
        # c="xkcd:dark",
        c=cluster_colours[clusters_raw[c]],
        zorder=-1,
        alpha=0.15,
        lw=0.8,
    )
    if clusters_raw[c] == 1:
        kwargs["alpha"] = 0.5
    ax.plot(
        iters,
        # adjs - adjs[-1],
        adjs + xo.offsets_true[c],
        **kwargs,
    )
ax.set_ylabel("Adjustment + true offset / µmol kg$^{-1}$")
ax.set_ylim(-1.5, 1.5)
ax.set_xlim(0, isplit * 2)
ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000, 12410, 14949, 17410, 20000])
ax.set_xticklabels([0, 2, 4, 6, 8, 10, 250, 500, 750, 1000])
ax.axhline(0, c="k", lw=0.8)
ax.axvline(isplit, c="k", lw=0.8)
ax.set_xlabel("Number of FF iterations / 1000")
ax.text(0, 1.03, "(b)", transform=ax.transAxes)
fig.tight_layout()
fig.savefig("figures/fig02_regional_clusters.pdf")
