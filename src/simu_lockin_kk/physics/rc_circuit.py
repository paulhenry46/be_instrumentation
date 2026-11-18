"""Modélisation physique et simulation temporelle d'un circuit RC passe-bas.

Ce module fournit les équations de réponse en fréquence complexe :
    H(w) = 1 / (1 + jwRC)
ainsi que la simulation temporelle exacte pour une excitation sinusoïdale ou arbitraire,
compatible avec l'injection de bruit.
"""

import numpy as np

from simu_lockin_kk.config import (
    NoiseConfig,
    RCCircuitConfig,
    SpectrumData,
    TimeDomainData,
)

def compute_transfer_function(omega: np.ndarray | float, R: float, C: float) -> np.ndarray:
    r"""Calcule la réponse fréquentielle complexe H(w) d'un circuit RC passe-bas. Fonction vectorielle.

    Parameters
    ----------
    omega : np.ndarray | float
        Pulsation(s) d'excitation en rad/s.
    R : float
        Résistance en Ohms (R > 0).
    C : float
        Capacité en Farads (C > 0).

    Returns
    -------
    np.ndarray
        Fonction de transfert complexe H(w).
    """
    if R <= 0 or C <= 0:
        raise ValueError(f"R ({R}) et C ({C}) doivent être strictement positifs.")

    w_arr = np.asarray(omega, dtype=np.float64)
    tau = R * C
    denom = 1.0 + (w_arr * tau) ** 2
    real_part = 1.0 / denom
    imag_part = -(w_arr * tau) / denom
    return real_part + 1j * imag_part

def compute_spectrum(config: RCCircuitConfig, freqs: np.ndarray | None = None) -> SpectrumData:
    """Génère le spectre de fréquence théorique (SpectrumData) pour une configuration RC donnée.

    Parameters
    ----------
    config : RCCircuitConfig
        Paramètres du circuit RC (R, C, f_min, f_max, num_freqs, amplitude).
    freqs : np.ndarray | None, optional
        Vecteur personnalisé de fréquences en Hz. Si None, une échelle logarithmique
        entre config.f_min et config.f_max est générée.

    Returns
    -------
    SpectrumData
        Structure de données contenant freqs, omega, X, Y, R, phase et H_complex.
    """
    if freqs is None:
        freqs_arr = np.logspace(
            np.log10(config.f_min),
            np.log10(config.f_max),
            num=config.num_freqs,
            dtype=np.float64,
        )
    else:
        freqs_arr = np.asarray(freqs, dtype=np.float64)
        if len(freqs_arr) < 2:
            raise ValueError("Le vecteur de fréquences doit comporter au moins 2 points.")

    omega = 2.0 * np.pi * freqs_arr
    H_complex = compute_transfer_function(omega, config.R, config.C) * config.amplitude
    return SpectrumData.from_complex(freqs_arr, H_complex)

def generate_time_response(
    config: RCCircuitConfig,
    f_ref: float,
    duration: float = 0.1,
    fs: float = 1e5,
    noise_cfg: NoiseConfig | None = None,
) -> TimeDomainData:
    """Génère la réponse temporelle du circuit RC à une excitation sinusoïdale pure.

    Utilise la réponse en régime permanent sinusoïdal (gain d'amplitude et déphasage analytiques)
    pour garantir une simulation exacte sans biais transitoire.

    Parameters
    ----------
    config : RCCircuitConfig
        Configuration du circuit RC.
    f_ref : float
        Fréquence de l'excitation sinusoïdale en Hz.
    duration : float, optional
        Durée du signal en secondes (défaut : 0.1 s).
    fs : float, optional
        Fréquence d'échantillonnage en Hz (défaut : 100 kHz).
    noise_cfg : NoiseConfig | None, optional
        Configuration du bruit à injecter. Si None, aucun bruit n'est ajouté.

    Returns
    -------
    TimeDomainData
        Objet contenant t, v_in, v_out_clean, v_out_noisy et f_ref.
    """
    if f_ref <= 0:
        raise ValueError(f"f_ref doit être strictement positive (obtenu : {f_ref}).")
    if duration <= 0:
        raise ValueError(f"La durée doit être strictement positive (obtenu : {duration}).")
    if fs <= 2.0 * f_ref:
        raise ValueError(
            f"La fréquence d'échantillonnage fs ({fs} Hz) doit respecter le critère de Shannon "
            f"(> 2*f_ref = {2.0 * f_ref} Hz)."
        )

    num_samples = int(np.ceil(duration * fs))
    t = np.linspace(0.0, duration, num_samples, endpoint=False, dtype=np.float64)

    omega_ref = 2.0 * np.pi * f_ref
    v_in = config.amplitude * np.sin(omega_ref * t)

    # Réponse harmonique complexe H(w_ref)
    H_ref = compute_transfer_function(omega_ref, config.R, config.C)
    gain = np.abs(H_ref)
    phase_shift = np.angle(H_ref)

    # Signal de sortie propre en régime permanent
    v_out_clean = config.amplitude * gain * np.sin(omega_ref * t + phase_shift)

    # Injection du bruit si configuré
    v_out_noisy = np.copy(v_out_clean)
    if noise_cfg is not None and noise_cfg.noise_type != "none":
        from simu_lockin_kk.signal.noise import apply_noise

        v_out_noisy = apply_noise(v_out_clean, fs=fs, noise_cfg=noise_cfg)

    return TimeDomainData(
        t=t,
        v_in=v_in,
        v_out_clean=v_out_clean,
        v_out_noisy=v_out_noisy,
        f_ref=f_ref,
    )

class RCCircuit:
    """Modélisation orientée objet représentative d'un composant physique RC."""

    def __init__(self, config: RCCircuitConfig) -> None:
        """Initialise le circuit RC avec sa configuration physique.

        Parameters
        ----------
        config : RCCircuitConfig
            Configuration physique (R, C, amplitude, etc.).
        """
        self.config = config

    @property
    def R(self) -> float:
        """Résistance en Ohms."""
        return self.config.R

    @property
    def C(self) -> float:
        """Capacité en Farads."""
        return self.config.C

    @property
    def tau(self) -> float:
        """Constante de temps tau = R * C (s)."""
        return self.config.tau

    @property
    def fc(self) -> float:
        """Fréquence de coupure à -3 dB en Hz."""
        return self.config.fc

    def H(self, omega: np.ndarray | float) -> np.ndarray:
        """Calcule H(w) pour une ou plusieurs pulsations."""
        return compute_transfer_function(omega, self.config.R, self.config.C)

    def get_spectrum(self, freqs: np.ndarray | None = None) -> SpectrumData:
        """Génère la structure de données spectrales SpectrumData."""
        return compute_spectrum(self.config, freqs=freqs)

    def simulate_signal(
        self,
        f_ref: float,
        duration: float = 0.1,
        fs: float = 1e5,
        noise_cfg: NoiseConfig | None = None,
    ) -> TimeDomainData:
        """Simule la réponse temporelle du circuit RC pour une fréquence donnée."""
        return generate_time_response(
            config=self.config,
            f_ref=f_ref,
            duration=duration,
            fs=fs,
            noise_cfg=noise_cfg,
        )