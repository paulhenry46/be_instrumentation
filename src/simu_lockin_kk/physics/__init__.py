"""Module de modélisation physique du circuit RC.

Ce module regroupe les fonctions et classes pour calculer la fonction de transfert
complexe, générer le spectre théorique et simuler la réponse temporelle du circuit RC.
"""

from simu_lockin_kk.physics.rc_circuit import (
    RCCircuit,
    compute_spectrum,
    compute_transfer_function,
    generate_time_response,
)

__all__ = [
    "RCCircuit",
    "compute_transfer_function",
    "compute_spectrum",
    "generate_time_response",
]