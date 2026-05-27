# %%
import numpy as np
from matplotlib import pyplot as plt, rcParams
from scipy.stats import linregress
from xover import inversion as xinv

import g3m.simulate as sim


rcParams["font.size"] = 8

# Initialise random number generator
rng = np.random.default_rng(1)
# Simulate crossovers - no noise
xo_nn = sim.generate_crossovers(
    rng,
    n_big_offsets=30,
    n_no_offset=30,
    scale_offsets=3,
    scale_big_offsets=8,
    # trend_type="constant",
    trend_type="grad",
    trend=0.2,
    trend_se_factor=1,
    trend_min=0,
    weights_s=0.6,
    weights_loc=0,
    weights_scale=0.1,
    randomise_weights=False,
    crossover_noise_scale=0,
)

# Initialise random number generator
rng = np.random.default_rng(1)
# Simulate crossovers
xo = sim.generate_crossovers(
    rng,
    n_big_offsets=30,
    n_no_offset=30,
    scale_offsets=3,
    scale_big_offsets=8,
    # trend_type="constant",
    trend_type="grad",
    trend=0.2,
    trend_se_factor=1,
    trend_min=0,
    weights_s=0.6,
    weights_loc=0,
    weights_scale=0.1,
    randomise_weights=True,
    crossover_noise_scale=1,
)


ff = xinv.furthest_first(
    xo.crossovers,
    niter=50_000,
    weights=xo.weights,
)
offsets_final = xo.offsets_true + ff.adjustments
xovers_final = np.vstack(offsets_final) - offsets_final + xo.trend_diffs
xovers_final = np.where(xo.weights == 0, np.nan, xovers_final)
# ^ the above is the same as:
# xovers_final = ff.xovers_adjusted + date_diffs * trends
# Calculate `tra`, apparent trends after inversion ("tra" = "trends after")
# for comparison with `xo.trb` ("trb" = "trends before")
tra = xinv.get_trends_wls(
    xovers_final, xo.weights, xo.date_diffs, min_xovers=5
)

ff_sym = xinv.furthest_first(
    xo.crossovers_dt_sym,
    niter=50_000,
    weights=xo.weights,
)
offsets_final = xo.offsets_true + ff_sym.adjustments
xovers_final = np.vstack(offsets_final) - offsets_final + xo.trend_diffs
xovers_final = np.where(xo.weights == 0, np.nan, xovers_final)
# ^ the above is the same as:
# xovers_final = ff.xovers_adjusted + date_diffs * trends
# Calculate `tra`, apparent trends after inversion ("tra" = "trends after")
# for comparison with `xo.trb` ("trb" = "trends before")
tra_sym = xinv.get_trends_wls(
    xovers_final, xo.weights, xo.date_diffs, min_xovers=5
)

# %% Now, repeat the above but with FF2
# Without trend correction
step1_ff = np.argwhere(np.abs(ff.steps.adjustment) < 0.1)[0][0]
adj1_ff = xinv.adjustments_at_step(ff, step=step1_ff)
allowed_ff = np.abs(adj1_ff) >= 2
ff2 = xinv.furthest_first(
    xo.crossovers,
    niter=50_000,
    weights=xo.weights,
    allowed=allowed_ff,
)
offsets_final_ff2 = xo.offsets_true + ff2.adjustments
xovers_final_ff2 = (
    np.vstack(offsets_final_ff2) - offsets_final_ff2 + xo.trend_diffs
)
xovers_final_ff2 = np.where(xo.weights == 0, np.nan, xovers_final_ff2)
tra_ff2 = xinv.get_trends_wls(
    xovers_final_ff2, xo.weights, xo.date_diffs, min_xovers=5
)

# With trend correction
step1_ff_sym = np.argwhere(np.abs(ff_sym.steps.adjustment) < 0.1)[0][0]
adj1_ff_sym = xinv.adjustments_at_step(ff_sym, step=step1_ff_sym)
allowed_ff_sym = np.abs(adj1_ff_sym) >= 2
ff2_sym = xinv.furthest_first(
    xo.crossovers_dt_sym,
    niter=50_000,
    weights=xo.weights,
    allowed=allowed_ff_sym,
)
offsets_final_ff2_sym = xo.offsets_true + ff2_sym.adjustments
xovers_final_ff2_sym = (
    np.vstack(offsets_final_ff2_sym) - offsets_final_ff2_sym + xo.trend_diffs
)
xovers_final_ff2_sym = np.where(xo.weights == 0, np.nan, xovers_final_ff2_sym)
tra_ff2_sym = xinv.get_trends_wls(
    xovers_final_ff2_sym, xo.weights, xo.date_diffs, min_xovers=5
)

# %% Make the figure
fig, axs = plt.subplots(nrows=3, figsize=(3.3, 8))
ax = axs[0]
ax.text(0, 1.03, "(a)", transform=ax.transAxes)
ax.scatter(
    xo.trb.slope,
    tra.slope,
    alpha=0.8,
    s=8,
    label="Standard FF$_1$",
    c="xkcd:cool blue",
    marker="x",
)
ax.scatter(
    xo.trb.slope,
    tra_sym.slope,
    alpha=0.8,
    s=10,
    label="Trend-preserving FF$_1$",
    c="xkcd:seaweed green",
    marker="+",
)
ax.set_xlabel("Apparent trend before / µmol kg$^{-1}$ yr$^{-1}$")
ax.set_ylabel("Apparent trend after / µmol kg$^{-1}$ yr$^{-1}$")
ax = axs[1]
ax.text(0, 1.03, "(b)", transform=ax.transAxes)
ax.scatter(
    tra.slope,
    tra_ff2.slope,
    s=8,
    c="xkcd:cool blue",
    alpha=0.8,
    label="Standard",
    marker="x",
    zorder=2,
)
ax.scatter(
    tra_sym.slope,
    tra_ff2_sym.slope,
    s=10,
    c="xkcd:seaweed green",
    alpha=0.8,
    label="Trend-preserving",
    marker="+",
)
ax.set_xlabel("Trend after FF$_1$ / µmol kg$^{-1}$ yr$^{-1}$")
ax.set_ylabel("Trend after FF$_2$ / µmol kg$^{-1}$ yr$^{-1}$")
ax = axs[2]
ax.text(0, 1.03, "(c)", transform=ax.transAxes)
ax.scatter(
    xo.trends_true,
    xo.trb.slope,
    s=10,
    edgecolor="none",
    c="xkcd:black",
    alpha=0.8,
    label="Before",
    marker="o",
)
ax.scatter(
    xo.trends_true,
    tra.slope,
    s=8,
    c="xkcd:cool blue",
    alpha=0.8,
    label="After (st. FF$_1$)",
    marker="x",
)
ax.scatter(
    xo.trends_true,
    tra_sym.slope,
    s=10,
    c="xkcd:seaweed green",
    alpha=0.8,
    label="After (tp. FF$_1$)",
    marker="+",
)
ax.set_xlabel("True trend / µmol kg$^{-1}$ yr$^{-1}$")
ax.set_ylabel("Apparent trend / µmol kg$^{-1}$ yr$^{-1}$")
for ax in axs:
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.4, 0.6)
    ax.set_ylim(-0.4, 0.6)
    ax.set_aspect(1)
    ax.axline((0, 0), slope=1, c="k", lw=0.8)
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6])
    ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6])
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("figures/fig01_trends.pdf")

# %% Calculate statistics
lr_a = linregress(xo.trb.slope, tra_sym.slope)
print(f"Panel (a) r-square = {lr_a.rvalue**2:.2f}")
lr_b = linregress(tra_sym.slope, tra_ff2_sym.slope)
print(f"Panel (b) r-square = {lr_b.rvalue**2:.2f}")
rmsd_ff1 = np.sqrt(np.mean((tra_sym.slope - xo.trb.slope) ** 2))
print(f"RMSD FF1 vs assigned trends = {rmsd_ff1:.3f} µmol / (kg * yr)")
rmsd_ff2 = np.sqrt(np.mean((tra_ff2_sym.slope - xo.trb.slope) ** 2))
print(f"RMSD FF2 vs assigned trends = {rmsd_ff2:.3f} µmol / (kg * yr)")
