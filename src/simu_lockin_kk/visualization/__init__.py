"""Module de visualisation graphique pour le projet simu_lockin_kk.
"""

from simu_lockin_kk.visualization.plots import (
    plot_bode,
    plot_extrapolation_comparison,
    plot_kramers_kronig_comparison,
    plot_extrapolated_points,
    plot_time_domain,
)

__all__ = [
    "plot_bode",
    "plot_kramers_kronig_comparison",
    "plot_time_domain",
    "plot_extrapolated_points",
    "plot_extrapolation_comparison"
]