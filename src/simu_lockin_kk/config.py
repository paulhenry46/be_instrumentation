"""Module de configuration et structures de données (contrats d'interfaces).

Ce module définit l'ensemble des structures de données typées (`dataclass`)
partagées entre les différents maillons de la chaîne de traitement.
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np

@dataclass(frozen=True)
class RCCircuitConfig:
    """Configuration physique d'un circuit RC passe-bas et du balayage fréquentiel.

    Attributes
    ----------
    R : float
        Résistance en Ohms (R > 0).
    C : float
        Capacité en Farads (C > 0).
    amplitude : float
        Amplitude du signal de commande en Volts (défaut : 1.0 V).
    f_min : float
        Fréquence minimale de balayage en Hz (défaut : 1.0 Hz).
    f_max : float
        Fréquence maximale de balayage en Hz (défaut : 1.0 MHz).
    num_freqs : int
        Nombre de points de fréquence (défaut : 100).
    """

    R: float
    C: float
    amplitude: float = 1.0
    f_min: float = 1.0
    f_max: float = 1e6
    num_freqs: int = 100

    def __post_init__(self) -> None:
        """Validation des paramètres physiques."""
        if self.R <= 0:
            raise ValueError(f"La résistance R doit être strictement positive (obtenu : {self.R})")
        if self.C <= 0:
            raise ValueError(f"La capacité C doit être strictement positive (obtenu : {self.C})")
        if self.amplitude <= 0:
            raise ValueError(f"L'amplitude doit être strictement positive (obtenu : {self.amplitude})")
        if self.f_min <= 0:
            raise ValueError(f"f_min doit être strictement positive (obtenu : {self.f_min})")
        if self.f_max <= self.f_min:
            raise ValueError(f"f_max ({self.f_max}) doit être supérieure à f_min ({self.f_min})")
        if self.num_freqs < 2:
            raise ValueError(f"num_freqs doit être au moins égal à 2 (obtenu : {self.num_freqs})")

    @property
    def tau(self) -> float:
        """Constante de temps du circuit RC (tau = R * C) en secondes."""
        return self.R * self.C

    @property
    def fc(self) -> float:
        """Fréquence de coupure à -3 dB en Hz (fc = 1 / (2 * pi * R * C))."""
        return 1.0 / (2.0 * np.pi * self.tau)

    @property
    def wc(self) -> float:
        """Pulsation de coupure en rad/s (wc = 1 / (R * C))."""
        return 1.0 / self.tau

@dataclass(frozen=True)
class NoiseConfig:
    """Configuration des bruits parasites injectés dans le signal temporel.

    Attributes
    ----------
    noise_type : Literal["none", "white", "flicker", "combined"]
        Type de bruit à simuler.
    std_white : float
        Écart-type du bruit blanc gaussien en Volts.
    alpha_flicker : float
        Exposant spectral pour le bruit en 1/f^alpha (défaut : 1.0).
    seed : int | None
        Graine aléatoire pour la répétabilité des simulations.
    """

    noise_type: Literal["none", "white", "flicker", "combined"] = "white"
    std_white: float = 0.05
    alpha_flicker: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validation de la configuration de bruit."""
        if self.std_white < 0:
            raise ValueError(f"std_white doit être >= 0 (obtenu : {self.std_white})")
        if self.alpha_flicker < 0:
            raise ValueError(f"alpha_flicker doit être >= 0 (obtenu : {self.alpha_flicker})")

@dataclass(frozen=True)
class LockInConfig:
    """Configuration de l'amplificateur synchrone (Lock-in Amplifier).

    Attributes
    ----------
    f_ref : float
        Fréquence de la référence de détection en Hz.
    harmonic : int
        Harmonique de détection (1 pour la fondamentale, 2 pour 2f, etc.).
    lpf_cutoff : float
        Fréquence de coupure du filtre passe-bas en Hz.
    lpf_order : int
        Ordre du filtre passe-bas (ex: ordre 2 = 12 dB/octave).
    lpf_type : Literal["butterworth", "moving_average"]
        Type de filtre passe-bas appliqué après le multiplicateur PSD.
    """

    f_ref: float
    harmonic: int = 1
    lpf_cutoff: float = 10.0
    lpf_order: int = 2
    lpf_type: Literal["butterworth", "moving_average"] = "butterworth"

    def __post_init__(self) -> None:
        """Validation des paramètres du Lock-in."""
        if self.f_ref <= 0:
            raise ValueError(f"f_ref doit être strictement positive (obtenu : {self.f_ref})")
        if self.harmonic < 1:
            raise ValueError(f"harmonic doit être >= 1 (obtenu : {self.harmonic})")
        if self.lpf_cutoff <= 0:
            raise ValueError(f"lpf_cutoff doit être strictement positive (obtenu : {self.lpf_cutoff})")
        if self.lpf_order < 1:
            raise ValueError(f"lpf_order doit être >= 1 (obtenu : {self.lpf_order})")

    @property
    def tau_lpf(self) -> float:
        """Constante de temps équivalente du filtre passe-bas (s)."""
        return 1.0 / (2.0 * np.pi * self.lpf_cutoff)

@dataclass
class TimeDomainData:
    """Structure de données temporelles échangées entre le circuit RC et le Lock-in."""

    t: np.ndarray
    v_in: np.ndarray
    v_out_clean: np.ndarray
    v_out_noisy: np.ndarray
    f_ref: float

    def __post_init__(self) -> None:
        """Validation des dimensions et cohérence des tableaux."""
        n = len(self.t)
        if n < 2:
            raise ValueError("Le vecteur temps 't' doit contenir au moins 2 points.")
        for name, arr in [("v_in", self.v_in), ("v_out_clean", self.v_out_clean), ("v_out_noisy", self.v_out_noisy)]:
            if len(arr) != n:
                raise ValueError(f"Le tableau '{name}' a une taille ({len(arr)}) incohérente avec 't' ({n}).")

@dataclass
class SpectrumData:
    """Structure de données spectrales complexes (entrée/sortie du Lock-in et de Kramers-Kronig)."""

    freqs: np.ndarray
    omega: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    R: np.ndarray
    phase: np.ndarray
    H_complex: np.ndarray

    def __post_init__(self) -> None:
        """Vérification de l'intégrité spectrale."""
        n = len(self.freqs)
        if n < 2:
            raise ValueError("Le spectre doit contenir au moins 2 points de fréquence.")
        for name, arr in [
            ("omega", self.omega),
            ("X", self.X),
            ("Y", self.Y),
            ("R", self.R),
            ("phase", self.phase),
            ("H_complex", self.H_complex),
        ]:
            if len(arr) != n:
                raise ValueError(f"Le tableau '{name}' (taille {len(arr)}) ne correspond pas à 'freqs' ({n}).")

    @classmethod
    def from_complex(cls, freqs: np.ndarray, H_complex: np.ndarray) -> "SpectrumData":
        """Construit un objet `SpectrumData` directement à partir d'un vecteur complexe H(w)."""
        omega = 2.0 * np.pi * freqs
        X = np.real(H_complex)
        Y = np.imag(H_complex)
        R = np.abs(H_complex)
        phase = np.angle(H_complex)
        return cls(
            freqs=np.asarray(freqs, dtype=np.float64),
            omega=np.asarray(omega, dtype=np.float64),
            X=np.asarray(X, dtype=np.float64),
            Y=np.asarray(Y, dtype=np.float64),
            R=np.asarray(R, dtype=np.float64),
            phase=np.asarray(phase, dtype=np.float64),
            H_complex=np.asarray(H_complex, dtype=np.complex128),
        )

@dataclass(frozen=True)
class KKConfig:
    """Configuration de l'intégration Kramers-Kronig par Valeur Principale de Cauchy."""

    integration_method: Literal["cauchy_maclaurin", "hilbert", "trapezoidal_pv"] = "cauchy_maclaurin"
    singularity_eps: float = 1e-6
    extrapolation: Literal["none", "asymptotic", "constant"] = "asymptotic"
    omega_min_extrap: float | None = None
    omega_max_extrap: float | None = None

@dataclass
class KKResult:
    """Résultat de la vérification des relations de Kramers-Kronig."""

    omega: np.ndarray
    chi_original: np.ndarray
    chi_reconstructed: np.ndarray
    residuals: np.ndarray
    rms_error: float
    r2_score: float
    target_component: Literal["real", "imag"]