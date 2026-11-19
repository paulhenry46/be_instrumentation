"""Package de traitement du signal et détection synchrone.

Contient la génération de bruits parasites (bruit blanc, bruit 1/f)
et l'amplificateur synchrone numérique (Lock-in Amplifier).
"""

from simu_lockin_kk.signal.lockin import (
    LockInAmplifier,
    demodulate_signal,
    demodulate_sweep,
    demodulate_time_data,
)
from simu_lockin_kk.signal.noise import (
    apply_noise,
    generate_white_noise,
)

__all__ = [
    "generate_white_noise",
    "apply_noise",
    "LockInAmplifier",
    "demodulate_signal",
    "demodulate_time_data",
    "demodulate_sweep",
]