"""Module d'amplification synchrone (Lock-in Amplifier).

Implémente la détection sensible à la phase (PSD - Phase Sensitive Detection)
et le filtrage passe-bas (LPF) pour extraire l'amplitude et la phase
ou les composantes complexes (X, Y) d'un signal temporel.
"""

from typing import Literal  # Restreint certains paramètres à des chaînes littérales valides
import numpy as np  # Calcul numérique vectorisé sur tableaux
from scipy.signal import butter, sosfiltfilt  # Conception Butterworth et filtrage sans déphasage

from simu_lockin_kk.config import (  # Structures de configuration et conteneurs de données du projet
    LockInConfig,
    NoiseConfig,
    RCCircuitConfig,
    SpectrumData,
    TimeDomainData,
)


# Rôle : isole la composante continue (DC) d'un signal déjà démodulé.
# Retour : float (valeur moyenne X ou Y).
# Pourquoi : éliminer la composante à 2 * f_ref
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
    # Plafonne la coupure à 0.1 * f_ref afin d'atténuer fortement la composante à 2 * f_ref
    # générée par le mélange synchrone et de garder une bande passante étroite autour du DC.
    cutoff = min(cutoff, 0.1 * f_ref)

    # Nombre total d'échantillons du signal démodulé.
    n_total = len(signal)

    # Nombre d'échantillons contenus dans une période de référence T_ref = 1 / f_ref.
    period_samples = int(np.round(fs / f_ref))

    # Tronque à un nombre entier de périodes de référence quand possible : moyenner sur un nombre
    # entier de périodes annule exactement les termes périodiques à f_ref et 2 * f_ref.
    if period_samples >= 1 and n_total >= period_samples:
        num_periods = n_total // period_samples
        valid_len = num_periods * period_samples
        sig = signal[:valid_len]  # [:valid_len] sélectionne les indices de 0 à valid_len - 1
    else:
        # Repli si le signal est plus court qu'une période (ou fs < f_ref) : garde tout le signal.
        sig = signal

    if lpf_type == "butterworth":
        # Fréquence de Nyquist (fs / 2), limite supérieure représentable sans repliement à fs.
        nyquist = 0.5 * fs

        # Normalise cutoff par rapport à nyquist (0.5 = fs/2) et plafonne à 0.49 pour rester
        # strictement sous la limite d'échantillonnage et éviter toute instabilité numérique.
        normal_cutoff = min(cutoff / nyquist, 0.49)

        # Calcule les coefficients du filtre Butterworth passe-bas en sections d'ordre 2 (SOS),
        # plus stables numériquement qu'une forme polynomiale (b, a) d'ordre élevé.
        sos = butter(order, normal_cutoff, btype="low", output="sos")

        # Estime la longueur minimale (~3 constantes de temps liées à cutoff) pour que la
        # réponse du filtre s'établisse correctement.
        min_samples_needed = int(np.ceil(3.0 * fs / cutoff))

        # Si le signal est plus court que min_samples_needed, le répète périodiquement pour
        # allonger la fenêtre et limiter les effets de bord lors du filtrage.
        if len(sig) < min_samples_needed:
            reps = int(np.ceil(min_samples_needed / len(sig)))
            sig_ext = np.tile(sig, reps)
        else:
            sig_ext = sig

        # Applique un filtrage avant-arrière (zero-phase) : annule le déphasage introduit par
        # le filtre tout en doublant la pente effective (ordre 2 * order).
        filtered = sosfiltfilt(sos, sig_ext)

        # Calcule la moyenne sur les len(sig) derniers échantillons (fin du signal étendu,
        # où le régime filtré est le plus stabilisé) pour obtenir la composante continue.
        n_eval = max(1, len(sig))
        return float(np.mean(filtered[-n_eval:]))

    elif lpf_type == "moving_average":
        # Moyenne sur le signal (tronqué à un nombre entier de périodes si possible),
        # équivalente à une intégration/moyenne sur la fenêtre temporelle.
        return float(np.mean(sig))

    else:
        # Rejette explicitement toute méthode de filtrage non reconnue.
        raise ValueError(f"Type de LPF inconnu : {lpf_type}")


# Rôle : exécute la détection synchrone (PSD) sur un signal temporel et un vecteur temps.
# Retour : (X, Y, R, phase).
# Pourquoi : extraire l'amplitude et la phase à la fréquence de référence même en présence de bruit.
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
    # Vérifie que t et v_measured ont exactement le même nombre d'échantillons.
    if len(t) != len(v_measured):
        raise ValueError("Les vecteurs t et v_measured doivent avoir la même longueur.")

    # Exige au moins 2 points pour pouvoir déduire le pas temporel dt.
    if len(t) < 2:
        raise ValueError("Au moins 2 points temporels sont requis pour la démodulation.")

    # Instancie une configuration minimale par défaut si aucune n'est fournie.
    if lockin_cfg is None:
        lockin_cfg = LockInConfig(f_ref=f_ref)

    # Pas d'échantillonnage temporel dt (en secondes), déduit des deux premiers instants.
    dt = float(t[1] - t[0])

    # Fréquence d'échantillonnage fs = 1 / dt (en Hz).
    fs = 1.0 / dt

    # Pulsation de démodulation w0 = 2 * pi * f_ref * harmonic
    w0 = 2.0 * np.pi * f_ref * lockin_cfg.harmonic

    # Facteur de normalisation : C = 2.0 restitue l'amplitude crête A après filtrage passe-bas,
    # tandis que C = sqrt(2) restitue la valeur efficace A / sqrt(2).
    factor = 2.0 if normalization == "peak" else np.sqrt(2.0)

    # Références orthogonales en phase (sinus) et en quadrature (cosinus, déphasée de +pi/2).
    ref_X = factor * np.sin(w0 * t)
    ref_Y = factor * np.cos(w0 * t)

    # Multiplie le signal mesuré par chaque référence : la composante à w0 est transposée en
    # composante continue (DC) tout en créant un terme autour de 2 * w0.
    s_X = v_measured * ref_X  # Transpose en DC + composante à 2 f_ref
    s_Y = v_measured * ref_Y

    # Filtre passe-bas sur la voie en phase pour extraire la composante X.
    X = _apply_lpf(
        s_X,
        fs=fs,
        f_ref=f_ref,
        cutoff=lockin_cfg.lpf_cutoff,
        order=lockin_cfg.lpf_order,
        lpf_type=lockin_cfg.lpf_type,
    )

    # Filtre passe-bas sur la voie en quadrature pour extraire la composante Y.
    Y = _apply_lpf(
        s_Y,
        fs=fs,
        f_ref=f_ref,
        cutoff=lockin_cfg.lpf_cutoff,
        order=lockin_cfg.lpf_order,
        lpf_type=lockin_cfg.lpf_type,
    )

    # Amplitude polaire R = sqrt(X^2 + Y^2).
    R = float(np.sqrt(X**2 + Y**2))

    # Phase en radians via arctan2(Y, X), gérant tous les quadrants et le cas X == 0.
    phase = float(np.arctan2(Y, X))

    return X, Y, R, phase


# Rôle : adapte demodulate_signal à l'objet TimeDomainData produit par la simulation temporelle.
# Retour : tuple (X, Y, R, phase).
# Pourquoi : simplifier l'appel sur les résultats de simulation avec ou sans bruit.
def demodulate_time_data(
    time_data: TimeDomainData,
    lockin_cfg: LockInConfig | None = None,
    use_noisy: bool = True,
    normalization: Literal["peak", "rms"] = "peak",
) -> tuple[float, float, float, float]:
    """Démodule une structure TimeDomainData issue de la simulation temporelle du circuit RC."""
    # Choisit le signal bruité (cas réaliste) ou le signal propre (validation sans bruit).
    if use_noisy:
        signal = time_data.v_out_noisy
    else:
        signal = time_data.v_out_clean

    # Délègue la démodulation à demodulate_signal avec l'axe temps et f_ref de TimeDomainData.
    return demodulate_signal(
        t=time_data.t,
        v_measured=signal,
        f_ref=time_data.f_ref,
        lockin_cfg=lockin_cfg,
        normalization=normalization,
    )


# Rôle : encapsule une configuration LockInConfig dans une classe orientée objet.
# Retour : méthodes .measure() et .measure_time_data() retournant (X, Y, R, phase).
# Pourquoi : conserver une configuration fixe sur plusieurs mesures successives.
class LockInAmplifier:
    """Modélisation orientée objet d'un amplificateur synchrone numérique."""

    def __init__(self, config: LockInConfig) -> None:
        # Enregistre la config
        self.config = config

    def measure(
        self,
        t: np.ndarray,
        v_measured: np.ndarray,
        normalization: Literal["peak", "rms"] = "peak",
    ) -> tuple[float, float, float, float]:
        """Effectue une mesure synchrone sur un signal temporel."""
        # Appelle demodulate_signal en réutilisant la fréquence et la configuration de l'instance.
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
        # Appelle demodulate_time_data avec la configuration enregistrée dans l'objet.
        return demodulate_time_data(
            time_data=time_data,
            lockin_cfg=self.config,
            use_noisy=use_noisy,
            normalization=normalization,
        )


# Rôle : simule et démodule un balayage complet en fréquence sur le circuit RC.
# Retour : SpectrumData contenant la réponse complexe mesurée H_complex = X + j*Y.
# Pourquoi : obtenir un spectre fréquentiel complet comparable à la théorie ou à Kramers-Kronig.
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
    # Import local pour éviter une dépendance circulaire avec rc_circuit.py.
    from simu_lockin_kk.physics.rc_circuit import generate_time_response

    if freqs is None:
        # Grille logarithmique entre f_min et f_max pour couvrir plusieurs décades à résolution
        # fréquentielle relative constante.
        freqs_arr = np.logspace(
            np.log10(rc_config.f_min),
            np.log10(rc_config.f_max),
            num=rc_config.num_freqs,
            dtype=np.float64,
        )
    else:
        # Convertit le vecteur de fréquences fourni en tableau float64.
        freqs_arr = np.asarray(freqs, dtype=np.float64)

    # Accumule les composantes X et Y obtenues pour chaque fréquence du balayage.
    X_list = []
    Y_list = []

    for f in freqs_arr:
        # Garantit au moins 10 échantillons par période aux hautes fréquences pour limiter l'erreur
        # numérique de discrétisation.
        fs_point = max(fs, 10.0 * f)

        # Garantit au moins 5 périodes complètes aux basses fréquences pour disposer d'une fenêtre
        # temporelle suffisamment longue.
        duration_point = max(duration, 5.0 / f)

        # Simule le signal temporel du circuit RC à la fréquence f. TODO Why ?
        time_data = generate_time_response(
            config=rc_config,
            f_ref=f,
            duration=duration_point,
            fs=fs_point,
            noise_cfg=noise_cfg,
        )

        # Configure le Lock-in à la fréquence courante f du balayage.
        lockin_cfg = LockInConfig(
            f_ref=f,
            lpf_cutoff=lpf_cutoff,
            lpf_order=lpf_order,
            lpf_type=lpf_type,
        )

        # Démodule le signal bruité et ne conserve que X et Y (R et phase sont recalculés dans SpectrumData).
        X, Y, _, _ = demodulate_time_data(time_data, lockin_cfg=lockin_cfg, use_noisy=True)
        X_list.append(X)
        Y_list.append(Y)

    # Convertit les résultats en tableaux float64 pour construire la réponse complexe.
    X_arr = np.array(X_list, dtype=np.float64)
    Y_arr = np.array(Y_list, dtype=np.float64)

    # Reconstruit le spectre complexe expérimental/simulé H(f) = X(f) + j * Y(f).
    H_complex = X_arr + 1j * Y_arr

    # Retourne l'objet SpectrumData complet à partir des fréquences et de H_complex.
    return SpectrumData.from_complex(freqs_arr, H_complex)