# %% This script is for easy importing of the real DIC dataset
import networkx as nx
import numpy as np
import pandas as pd
import xover.inversion as xinv


# Select which parameter to analyse
xvar = "DIC"
which_subgraph = 0  # this needs to be run across every subgraph separately

# Import datasets
# ---------------
#   xovers: crossover difference matrix without trend correction
#    dates: matrix of date differences for the crossovers
#  weights: weights matrix for the crossovers
#  cruises: table of summary info of all cruises
# xcruises: `cruises`, subset to rows that have crossovers for `xvar`
xovers = np.loadtxt(f"data/read/{xvar}_xovers.txt")
dates = np.loadtxt(f"data/read/{xvar}_dates.txt")
weights = np.loadtxt(f"data/read/{xvar}_weights.txt")
cruises = pd.read_parquet("data/read/cruises.parquet")
xcruises = cruises[cruises[f"ix_{xvar}"].notnull()].copy()
xcruises[f"ix_{xvar}"] = xcruises[f"ix_{xvar}"].astype(int)
xcruises = xcruises.reset_index().set_index(f"ix_{xvar}")
assert (xcruises.index.values == range(xcruises.shape[0])).all()

# Identify connected subgraphs
# ----------------------------
graph = nx.from_numpy_array(weights > 0)
subgraphs = list(nx.connected_components(graph))
print(f"Parameter {xvar} has {len(subgraphs)} connected subgraphs.")
print("Their sizes are: ", [len(sg) for sg in subgraphs])
xcruises["subgraph"] = -999
for g, subgraph in enumerate(subgraphs):
    xcruises.loc[xcruises.index.isin(subgraph), "subgraph"] = g

# Subset to one subgraph
# ----------------------
# xcruises_sg: `xcruises`, subset to the connected subgraph being analysed
#         dof: degrees of freedom per cruise
#      t_crit: t-distribution value per cruise
G = xcruises.subgraph == which_subgraph
xovers = xovers[G][:, G]
dates = dates[G][:, G]
weights = weights[G][:, G]
xcruises_sg = xcruises.loc[G].reset_index()
dof = xinv.dof_kish(weights)
t_crit = xinv.t_critical(dof)

# Get trends and adjust `xovers` for them
# ---------------------------------------
# offsets_raw: calculated without trend correction
#      trends: trends in `xvar` for each cruise
#   xovers_dt: crossover difference matrix with trend correction
offsets_raw = xinv.offsets_weighted(xovers, weights)
# Recalculate trends here
trends_wls = xinv.get_trends_wls(xovers, weights, dates)
xcruises_sg[f"intercept_{xvar}"] = trends_wls.intercept
xcruises_sg[f"slope_{xvar}"] = trends_wls.slope
xcruises_sg[f"slope_{xvar}_stderr"] = trends_wls.slope_se
trends = xcruises_sg[f"slope_{xvar}"].values.copy()
trends_se = xcruises_sg[f"slope_{xvar}_stderr"].values.copy()
trends = np.where(np.abs(trends) < trends_se, 0, trends)
# Adjust `xovers` with their trends
xovers_dt = xovers - dates * trends
offsets_pre = xinv.offsets_weighted(xovers_dt, weights)
offsets_pre_u = xinv.sem_weighted_t(
    xovers_dt, weights, offsets_pre, dof, t_crit
)
offsets_pre_u = np.where(
    np.isnan(offsets_pre_u) | np.isinf(offsets_pre_u),
    2 * np.std(offsets_pre, ddof=1),
    offsets_pre_u,
)

# Determine where adjustments are allowed
xovers_dt_sym = (xovers_dt - xovers_dt.T) / 2

# Get connected graph
graph_sg = graph.subgraph(subgraphs[which_subgraph])
graph_sg = nx.relabel_nodes(
    graph_sg, {v: i for i, v in xcruises_sg.ix_DIC.items()}
)

# Add region labels
for c in ["SO", "ATL", "PAC", "IND"]:
    cruises[c] = cruises[c] == 1
cruises["regions"] = ""
for i, row in cruises.iterrows():
    if row.ATL:
        cruises.loc[i, "regions"] += "ATL-"
    if row.IND:
        cruises.loc[i, "regions"] += "IND-"
    if row.PAC:
        cruises.loc[i, "regions"] += "PAC-"
    if row.SO:
        cruises.loc[i, "regions"] += "SO-"
for i, row in cruises.iterrows():
    if row.regions.endswith("-"):
        cruises.loc[i, "regions"] = row.regions[:-1]
