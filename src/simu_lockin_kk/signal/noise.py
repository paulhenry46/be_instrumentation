"""Module de génération de bruits parasites pour la simulation de signaux.

Fournit des fonctions pour générer du bruit blanc gaussien,
du bruit en 1/f (flicker noise) et leur combinaison.
"""

from typing import Literal
import numpy as np

from simu_lockin_kk.config import NoiseConfig

def generate_white_noise(
    n_samples: int,
    std: float = 0.05,
    seed: int | None = None,
) -> np.ndarray:
    """Génère un vecteur de bruit blanc gaussien d'espérance nulle.

    Parameters
    ----------
    n_samples : int
        Nombre d'échantillons.
    std : float, optional
        Écart-type du bruit en Volts (défaut : 0.05 V).
    seed : int | None, optional
        Graine du générateur pseudo-aléatoire.

    Returns
    -------
    np.ndarray
        Vecteur de bruit blanc de taille (n_samples,).
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples doit être strictement positif (obtenu : {n_samples}).")
    if std < 0:
        raise ValueError(f"std doit être >= 0 (obtenu : {std}).")

    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=std, size=n_samples)

def generate_flicker_noise(
    n_samples: int,
    fs: float,
    std: float = 0.05,
    alpha: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Génère du bruit rose / scintillement (1/f^alpha) par mise en forme spectrale (FFT).

    Parameters
    ----------
    n_samples : int
        Nombre d'échantillons.
    fs : float
        Fréquence d'échantillonnage en Hz.
    std : float, optional
        Écart-type cible du bruit généré (défaut : 0.05 V).
    alpha : float, optional
        Exposant spectral (alpha = 1.0 pour bruit 1/f, alpha = 2.0 pour bruit brun).
    seed : int | None, optional
        Graine aléatoire.

    Returns
    -------
    np.ndarray
        Vecteur de bruit 1/f de taille (n_samples,).
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples doit être strictement positif (obtenu : {n_samples}).")
    if std == 0.0:
        return np.zeros(n_samples, dtype=np.float64)

    rng = np.random.default_rng(seed)
    # 1. Bruit blanc dans le domaine temporel
    white = rng.normal(loc=0.0, scale=1.0, size=n_samples)

    # 2. Passage dans le domaine fréquentiel
    fft_white = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

    # 3. Filtrage en 1 / f^(alpha/2)
    gain = np.ones_like(freqs)
    gain[1:] = 1.0 / (freqs[1:] ** (alpha / 2.0))
    gain = gain  # Composante continue constante

    fft_flicker = fft_white * gain

    # 4. Retour dans le domaine temporel
    flicker = np.fft.irfft(fft_flicker, n=n_samples)

    # 5. Normalisation à l'écart-type souhaité
    current_std = np.std(flicker)
    if current_std > 0:
        flicker = (flicker / current_std) * std

    return flicker

def apply_noise(
    signal: np.ndarray,
    fs: float,
    noise_cfg: NoiseConfig,
) -> np.ndarray:
    """Applique la configuration de bruit spécifiée sur un signal temporel propre.

    Parameters
    ----------
    signal : np.ndarray
        Signal temporel propre.
    fs : float
        Fréquence d'échantillonnage en Hz.
    noise_cfg : NoiseConfig
        Configuration du bruit.

    Returns
    -------
    np.ndarray
        Signal temporel bruité.
    """
    if noise_cfg.noise_type == "none":
        return np.copy(signal)

    n_samples = len(signal)
    noisy_signal = np.copy(signal)

    if noise_cfg.noise_type in ("white", "combined"):
        white = generate_white_noise(
            n_samples=n_samples,
            std=noise_cfg.std_white,
            seed=noise_cfg.seed,
        )
        noisy_signal += white

    if noise_cfg.noise_type in ("flicker", "combined"):
        flicker_seed = noise_cfg.seed + 1 if noise_cfg.seed is not None else None
        flicker = generate_flicker_noise(
            n_samples=n_samples,
            fs=fs,
            std=noise_cfg.std_white,
            alpha=noise_cfg.alpha_flicker,
            seed=flicker_seed,
        )
        noisy_signal += flicker

    return noisy_signal