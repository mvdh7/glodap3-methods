from collections import namedtuple

import networkx as nx
import numpy as np
import xover.inversion as xinv
from scipy import stats


Degrees = namedtuple(
    "Degrees",
    (
        "expocodes",
        "degrees",
        "distances",
        "px",
        "py",
        "n_cruises",
    ),
)
Crossovers = namedtuple(
    "Crossovers",
    (
        "dg",
        "graph",
        "offsets_true",
        "offsets_apparent",
        "weights",
        "dates",
        "date_diffs",
        "trends_true",
        "trend_diffs",
        "trb",
        "trends",
        "crossovers",
        "crossovers_dt",
        "crossovers_dt_sym",
        "crossover_noise",
    ),
)


def generate_degrees(rng, c1=360, c2=25, c3=65, c4=80):
    """Simulate degree of (number of edges connected to) each cruise."""
    px, py = rng.uniform(size=(2, c1))
    px2, py2 = rng.uniform(low=1.1, high=1.3, size=(2, c2))
    px3, py3 = rng.uniform(low=-0.5, high=0, size=(2, c3))
    px4, py4 = rng.uniform(low=1.4, high=1.8, size=(2, c4))
    px = np.concat((px, px2, px3, px4))
    py = np.concat((py, py2, py3, py4))
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
    return Degrees(expocodes, degrees, distances, px, py, n_cruises)


def generate_graph(degrees, distances, expocodes, rng):
    """Generate a GLODAP-like random crossover graph.

    Parameters
    ----------
    degrees : np.array of int
        A size (n,) array of the degree of each node, where n is the total
        number of nodes.  Generated randomly internally if not provided,
        in which case n_nodes must be provided.
    distances : np.array of float
        A size (n, n) array of the "distances" between all pairs of nodes,
        used to determine which nodes are connected (closest first).
        Should be symmetric with main diagonal all zeroes.
    expocodes : np.array of str
        A size (n,) array of the expocode (name) of each cruise, in the same
        order as degrees.  These names are assigned to the nodes of the graph.
    rng : optional
        A Numpy random number Generator object.
        If not provided, `np.default_rng()` is used.

    Returns
    -------
    nx.Graph
        A randomly simulated graph.
    """
    degrees = degrees.copy()
    edges = []
    connected = {n: [] for n in range(len(degrees))}
    for _ in range(np.floor(degrees.sum() / 2).astype(int)):
        has_open = degrees > 0
        if has_open.sum() == 0:
            break
        # Pick a point `i` that needs more edges
        L = has_open & (degrees == degrees[degrees > 0].min())
        i = np.flatnonzero(L)[rng.integers(L.sum())]
        # Find `j`, the nearest point to `i` that also needs more edges...
        distances_i = np.where(
            (distances[i] == 0) | ~has_open,
            np.max(distances[i]) + 1,
            distances[i],
        )
        # ... and that isn't already connected to `i`
        distances_i[connected[i]] = np.max(distances[i]) + 1
        j = np.argmin(distances_i)
        if i == j:
            degrees[i] = 0
        elif degrees[j] == 0:
            # This happens if `i` is already connected to all nodes
            # that still need more edges
            degrees[i] = 0
        else:
            # Connect `i` to `j`
            connected[i].append(j)
            connected[j].append(i)
            degrees[i] -= 1
            degrees[j] -= 1
            edges.append((i, j))
    graph = nx.Graph()
    graph.add_edges_from(edges)
    # Merge subgraphs together, if there are any
    subgraphs = list(nx.connected_components(graph))
    for i, sg in enumerate(subgraphs):
        if i > 0:
            sg0 = subgraphs[0]
            graph.add_edge(
                list(sg0)[rng.integers(len(sg0))],
                list(sg)[rng.integers(len(sg))],
            )
    graph = nx.relabel_nodes(
        graph, {i: str(e) for i, e in enumerate(expocodes)}
    )
    return graph


def generate_crossovers(
    rng,
    n_big_offsets=30,
    n_no_offset=30,
    scale_offsets=3,
    scale_big_offsets=8,
    trend_type="constant",
    trend=0,
    trend_se_factor=1,
    trend_min=np.inf,
    weights_s=0.5,
    weights_loc=0,
    weights_scale=0.25,
    randomise_weights=True,
    crossover_noise_scale=0,
):
    # Simulate degree of (number of edges connected to) each cruise
    dg = generate_degrees(rng)
    expocodes, degrees, distances, px, py, n_cruises = dg
    graph = generate_graph(degrees, distances, expocodes, rng)

    # Initialise cruise offsets (i.e., the true errors that we want to find)
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

    return Crossovers(
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
