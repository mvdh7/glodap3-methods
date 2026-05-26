from collections import namedtuple


FurthestFirstSteps = namedtuple(
    "FurthestFirstSteps",
    ("index", "adjustment", "uncertainty"),
)
FurthestFirstResult = namedtuple(
    "FurthestFirstResult",
    (
        "adjustments",
        "uncertainties",
        "steps",
        "xovers_adjusted",
        "weights",
        "allowed",
        "dof",
        "t_crit",
        "niter",
    ),
)
