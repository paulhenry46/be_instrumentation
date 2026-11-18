"""Module d'amplification synchrone (Lock-in Amplifier).

Implémente la détection sensible à la phase (PSD - Phase Sensitive Detection)
et le filtrage passe-bas (LPF) pour extraire l'amplitude et la phase
ou les composantes complexes (X, Y) d'un signal temporel.
"""

from typing import Literal
import numpy as np
from scipy.signal import butter, sosfiltfilt

from simu_lockin_kk.config import (
    LockInConfig,
    NoiseConfig,
    RCCircuitConfig,
    SpectrumData,
    TimeDomainData,
)

def _apply_lpf(
    signal: np.ndarray,
    fs: float,
    f_ref: float,
    cutoff: float,
    order: int = 2,
    lpf_type: Literal["butterworth", "moving_average"] = "butterworth",
) -> float:
    """Applique le filtre passe-bas et extrait la composante continue (DC). Voir docs pour plus de détails sur les méthodes disponibles.

    Parameters
    ----------
    signal : np.ndarray
        Signal démodulé après multiplication par la référence.
    fs : float
        Fréquence d'échantillonnage en Hz.
    f_ref : float
        Fréquence de la référence (Hz).
    cutoff : float
        Fréquence de coupure du filtre passe-bas en Hz.
    order : int, optional
        Ordre du filtre Butterworth (défaut : 2).
    lpf_type : Literal["butterworth", "moving_average"], optional
        Type de filtrage (défaut : "butterworth").

    Returns
    -------
    float
        Valeur continue moyenne mesurée (composante X ou Y).
    """
    n_total = len(signal)
    period_samples = int(np.round(fs / f_ref))

    if period_samples >= 1 and n_total >= period_samples:
        num_periods = n_total // period_samples
        valid_len = num_periods * period_samples
        sig = signal[:valid_len]
    else:
        sig = signal

    if lpf_type == "butterworth":
        nyquist = 0.5 * fs
        normal_cutoff = min(cutoff / nyquist, 0.49)

        # Implémentation numérique stable via SOS (Second-Order Sections)
        sos = butter(order, normal_cutoff, btype="low", output="sos")

        # Extension périodique pour éliminer les résonances transitoires sur les signaux courts
        min_samples_needed = int(np.ceil(3.0 * fs / cutoff))
        if len(sig) < min_samples_needed:
            reps = int(np.ceil(min_samples_needed / len(sig)))
            sig_ext = np.tile(sig, reps)
        else:
            sig_ext = sig

        filtered = sosfiltfilt(sos, sig_ext)

        # Moyenne sur le segment stabilisé
        n_eval = max(1, len(sig))
        return float(np.mean(filtered[-n_eval:]))

    elif lpf_type == "moving_average":
        return float(np.mean(sig))

    else:
        raise ValueError(f"Type de LPF inconnu : {lpf_type}")

def demodulate_signal(
    t: np.ndarray,
    v_measured: np.ndarray,
    f_ref: float,
    lockin_cfg: LockInConfig | None = None,
    normalization: Literal["peak", "rms"] = "peak",
) -> tuple[float, float, float, float]:
    """Démodule un signal temporel v_measured(t) pour une fréquence de référence donnée.

    PSD (Phase Sensitive Detection) :
        s_X(t) = v_measured(t) * C * sin(h * w0 * t)
        s_Y(t) = v_measured(t) * C * cos(h * w0 * t)

    où C = 2.0 pour la normalisation d'amplitude crête ('peak')
    ou C = sqrt(2) pour la normalisation efficace ('rms').

    Parameters
    ----------
    t : np.ndarray
        Vecteur temps d'échantillonnage en secondes.
    v_measured : np.ndarray
        Signal temporel mesuré (Volts).
    f_ref : float
        Fréquence de la référence en Hz.
    lockin_cfg : LockInConfig | None, optional
        Configuration du Lock-in.
    normalization : Literal["peak", "rms"], optional
        Normalisation d'amplitude ("peak" => R = V_crête, "rms" => R = V_rms).

    Returns
    -------
    tuple[float, float, float, float]
        Composantes (X, Y, R, phase) démodulées.
    """
    if len(t) != len(v_measured):
        raise ValueError("Les vecteurs t et v_measured doivent avoir la même longueur.")
    if len(t) < 2:
        raise ValueError("Au moins 2 points temporels sont requis pour la démodulation.")

    if lockin_cfg is None:
        lockin_cfg = LockInConfig(f_ref=f_ref)

    dt = float(t[1] - t[0])
    fs = 1.0 / dt

    w0 = 2.0 * np.pi * f_ref * lockin_cfg.harmonic
    factor = 2.0 if normalization == "peak" else np.sqrt(2.0)

    # VOIE RÉFÉRENCE (Références sinusoïdales  0 et pi/2)
    ref_X = factor * np.sin(w0 * t)
    ref_Y = factor * np.cos(w0 * t)

    #  MULTIPLICATEUR
    s_X = v_measured * ref_X # Transpose en DC + composante à 2 f_ref
    s_Y = v_measured * ref_Y

    # FILTRAGE PASSE-BAS
    X = _apply_lpf(
        s_X,
        fs=fs,
        f_ref=f_ref,
        cutoff=lockin_cfg.lpf_cutoff,
        order=lockin_cfg.lpf_order,
        lpf_type=lockin_cfg.lpf_type,
    )
    Y = _apply_lpf(
        s_Y,
        fs=fs,
        f_ref=f_ref,
        cutoff=lockin_cfg.lpf_cutoff,
        order=lockin_cfg.lpf_order,
        lpf_type=lockin_cfg.lpf_type,
    )

    R = float(np.sqrt(X**2 + Y**2))
    phase = float(np.arctan2(Y, X))

    return X, Y, R, phase

def demodulate_time_data(
    time_data: TimeDomainData,
    lockin_cfg: LockInConfig | None = None,
    use_noisy: bool = True,
    normalization: Literal["peak", "rms"] = "peak",
) -> tuple[float, float, float, float]:
    """Démodule une structure TimeDomainData issue de la simulation temporelle du circuit RC."""
    signal = time_data.v_out_noisy if use_noisy else time_data.v_out_clean
    return demodulate_signal(
        t=time_data.t,
        v_measured=signal,
        f_ref=time_data.f_ref,
        lockin_cfg=lockin_cfg,
        normalization=normalization,
    )

class LockInAmplifier:
    """Modélisation orientée objet d'un amplificateur synchrone numérique."""

    def __init__(self, config: LockInConfig) -> None:
        self.config = config

    def measure(
        self,
        t: np.ndarray,
        v_measured: np.ndarray,
        normalization: Literal["peak", "rms"] = "peak",
    ) -> tuple[float, float, float, float]:
        """Effectue une mesure synchrone sur un signal temporel."""
        return demodulate_signal(
            t=t,
            v_measured=v_measured,
            f_ref=self.config.f_ref,
            lockin_cfg=self.config,
            normalization=normalization,
        )

    def measure_time_data(
        self,
        time_data: TimeDomainData,
        use_noisy: bool = True,
        normalization: Literal["peak", "rms"] = "peak",
    ) -> tuple[float, float, float, float]:
        """Effectue une mesure synchrone à partir d'un objet TimeDomainData."""
        return demodulate_time_data(
            time_data=time_data,
            lockin_cfg=self.config,
            use_noisy=use_noisy,
            normalization=normalization,
        )

def demodulate_sweep(
    rc_config: RCCircuitConfig,
    freqs: np.ndarray | None = None,
    duration: float = 0.1,
    fs: float = 1e5,
    noise_cfg: NoiseConfig | None = None,
    lpf_cutoff: float = 10.0,
    lpf_order: int = 2,
    lpf_type: Literal["butterworth", "moving_average"] = "butterworth",
) -> SpectrumData:
    """Effectue un balayage en fréquence complet simulé et extrait le spectre complexe (SpectrumData)."""
    from simu_lockin_kk.physics.rc_circuit import generate_time_response

    if freqs is None:
        freqs_arr = np.logspace(
            np.log10(rc_config.f_min),
            np.log10(rc_config.f_max),
            num=rc_config.num_freqs,
            dtype=np.float64,
        )
    else:
        freqs_arr = np.asarray(freqs, dtype=np.float64)

    X_list = []
    Y_list = []

    for f in freqs_arr:
        fs_point = max(fs, 10.0 * f)
        duration_point = max(duration, 5.0 / f)

        time_data = generate_time_response(
            config=rc_config,
            f_ref=f,
            duration=duration_point,
            fs=fs_point,
            noise_cfg=noise_cfg,
        )

        lockin_cfg = LockInConfig(
            f_ref=f,
            lpf_cutoff=lpf_cutoff,
            lpf_order=lpf_order,
            lpf_type=lpf_type,
        )

        X, Y, _, _ = demodulate_time_data(time_data, lockin_cfg=lockin_cfg, use_noisy=True)
        X_list.append(X)
        Y_list.append(Y)

    X_arr = np.array(X_list, dtype=np.float64)
    Y_arr = np.array(Y_list, dtype=np.float64)
    H_complex = X_arr + 1j * Y_arr

    return SpectrumData.from_complex(freqs_arr, H_complex)