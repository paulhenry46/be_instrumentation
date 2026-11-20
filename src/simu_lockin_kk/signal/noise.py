"""Module de génération de bruits parasites pour la simulation de signaux.

Fournit des fonctions pour générer du bruit blanc gaussien,
du bruit en 1/f (flicker noise) et leur combinaison.
"""

import numpy as np  # Bibliothèque de calcul numérique et vectoriel sur tableaux

from simu_lockin_kk.config import NoiseConfig  # Structure de configuration des paramètres de bruit

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
    # Un signal ne peut pas avoir un nombre d'échantillons nul ou négatif.
    if n_samples <= 0:
        raise ValueError(f"n_samples doit être strictement positif (obtenu : {n_samples}).")

    # L'écart-type d'une distribution gaussienne est nécessairement positif ou nul (std == 0 -> constant nul).
    if std < 0:
        raise ValueError(f"std doit être >= 0 (obtenu : {std}).")

    # Instancie un générateur moderne  avec graine optionnelle pour
    # debugage.
    rng = np.random.default_rng(seed)

    # Tire n_samples échantillons indépendants selon une loi normale centrée (loc=0.0) d'écart-type
    # scale=std : densité spectrale de puissance uniforme (bruit "blanc") et distribution gaussienne.
    return rng.normal(loc=0.0, scale=std, size=n_samples)


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
    # Si aucun bruit n'est demandé ("none"), retourne une copie du signal propre sans le modifier.
    if noise_cfg.noise_type == "none":
        return np.copy(signal)

    # Récupère la longueur temporelle du signal pour générer des vecteurs de bruit de même taille.
    n_samples = len(signal)

    # Copie du signal pour éviter toute mutation de l'objet original.
    noisy_signal = np.copy(signal)

    # Ajoute la composante de bruit blanc si le mode est "white"
    if noise_cfg.noise_type in ("white", "combined"):
        white = generate_white_noise(
            n_samples=n_samples,
            std=noise_cfg.std_white,
            seed=noise_cfg.seed,
        )
        noisy_signal += white

    return noisy_signal