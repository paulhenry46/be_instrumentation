"""Package principal simu_lockin_kk.

Simulation Python d'un amplificateur synchrone (Lock-in Amplifier)
et vérification des relations de Kramers-Kronig (VPC).
"""

from simu_lockin_kk.main import main, run_pipeline

__version__ = "0.1.0"

__all__ = [
    "run_pipeline",
    "main",
]