"""Modélisation physique et simulation temporelle d'un circuit RC passe-bas.

Ce module fournit les équations de réponse en fréquence complexe :
    H(w) = 1 / (1 + jwRC)
ainsi que la simulation temporelle exacte pour une excitation sinusoïdale,
compatible avec l'injection de bruit.
"""

import numpy as np  # Bibliothèque de calcul numérique et vectoriel sur tableaux

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
    # Vérifie que R et C sont strictement positifs : des valeurs nulles ou négatives n'ont pas de sens
    # physique ici et provoqueraient des divisions par zéro ou des constantes de temps invalides.
    if R <= 0 or C <= 0:
        raise ValueError(f"R ({R}) et C ({C}) doivent être strictement positifs.")

    # Convertit omega (scalaire ou np.ndarray) en ndarray float64 pour uniformiser le calcul
    # vectorisé et éviter les conversions implicites ou pertes de précision.
    w_arr = np.asarray(omega, dtype=np.float64)

    # Constante de temps tau = R * C (en secondes), qui fixe la dynamique et la coupure à -3 dB.
    tau = R * C

    # Dénominateur réel 1 + (w * tau)^2 obtenu après rationalisation de 1 / (1 + j * w * tau)
    # en multipliant haut et bas par le conjugué complexe (1 - j * w * tau).
    denom = 1.0 + (w_arr * tau) ** 2

    # Partie réelle Re{H(w)} = 1 / (1 + (w * tau)^2)
    real_part = 1.0 / denom

    # Partie imaginaire Im{H(w)} = -(w * tau) / (1 + (w * tau)^2)
    imag_part = -(w_arr * tau) / denom

    # Reconstruit le nombre/tableau complexe H(w) = Re{H(w)} + j * Im{H(w)}.
    return real_part + 1j * imag_part


def compute_spectrum(config: RCCircuitConfig, freqs: np.ndarray | None = None) -> SpectrumData: #TODO Why ?
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
        # Échantillonnage logarithmique entre f_min et f_max : indispensable sur plusieurs décades
        # pour conserver une résolution fréquentielle relative constante.
        freqs_arr = np.logspace(
            np.log10(config.f_min),
            np.log10(config.f_max),
            num=config.num_freqs,
            dtype=np.float64,
        )
    else:
        # Conversion du vecteur de fréquences fourni en tableau float64.
        freqs_arr = np.asarray(freqs, dtype=np.float64)
        # Il faut au moins 2 points pour définir un intervalle ou un balayage fréquentiel exploitable.
        if len(freqs_arr) < 2:
            raise ValueError("Le vecteur de fréquences doit comporter au moins 2 points.")

    # Conversion des fréquences en Hz vers les pulsations omega = 2 * pi * f en rad/s, requises par H(w).
    omega = 2.0 * np.pi * freqs_arr

    # Évalue H(w) puis multiplie par l'amplitude d'entrée : par linéarité en régime harmonique,
    # l'amplitude complexe de sortie vaut H(w) * V_in.
    H_complex = compute_transfer_function(omega, config.R, config.C) * config.amplitude

    # Encapsule fréquences et réponse complexe dans SpectrumData (qui calcule X, Y, module, phase...).
    return SpectrumData.from_complex(freqs_arr, H_complex)


def generate_time_response( #TODO Why ?
    config: RCCircuitConfig,
    f_ref: float,
    duration: float = 0.1,
    fs: float = 1e5,
    noise_cfg: NoiseConfig | None = None,
) -> TimeDomainData:
    """Génère la réponse temporelle du circuit RC à une excitation sinusoïdale pure.

    On se place en régime permanent

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
        Configuration du bruit à injecter. Si None = aucun bruit

    Returns
    -------
    TimeDomainData
        Objet contenant t, v_in, v_out_clean, v_out_noisy et f_ref.
    """
    # f_ref doit être strictement positive tant qu'à faire.
    if f_ref <= 0:
        raise ValueError(f"f_ref doit être strictement positive (obtenu : {f_ref}).")

    # duration doit être positive également.
    if duration <= 0:
        raise ValueError(f"La durée doit être strictement positive (obtenu : {duration}).")

    # Critère de shanon (fs > 2 * f_ref) pour éviter alisaing
    if fs <= 2.0 * f_ref:
        raise ValueError(
            f"La fréquence d'échantillonnage fs ({fs} Hz) doit respecter le critère de Shannon "
            f"(> 2*f_ref = {2.0 * f_ref} Hz)."
        )

    # Nombre total d'échantillons, arrondi à l'entier supérieur pour couvrir toute la durée demandée.
    num_samples = int(np.ceil(duration * fs))

    # Vecteur temps de num_samples points sur [0, duration[ : le paramètre endpoint à False évite de
    # dupliquer l'instant t = duration qui équivaut au début de la période suivante.
    t = np.linspace(0.0, duration, num_samples, endpoint=False, dtype=np.float64)

    # Pulsation angulaire de référence omega_ref = 2 * pi * f_ref (en rad/s).
    omega_ref = 2.0 * np.pi * f_ref

    # Signal d'entrée sinusoïdal pur : v_in(t) = A * sin(omega_ref * t).
    v_in = config.amplitude * np.sin(omega_ref * t)

    # Fonction de transfert complexe évaluée exactement à la pulsation d'excitation omega_ref.
    H_ref = compute_transfer_function(omega_ref, config.R, config.C)

    # Gain en amplitude |H(omega_ref)| = 1 / sqrt(1 + (omega_ref * tau)^2).
    gain = np.abs(H_ref)

    # Déphasage en radians arg(H(omega_ref)) = -arctan(omega_ref * tau).
    phase_shift = np.angle(H_ref)

    # Réponse de sortie propre en régime permanent sinusoïdal : v_out(t) = A * |H| * sin(w*t + phi).
    # On suppose que l'on est en régime permanent pour simplifier
    v_out_clean = config.amplitude * gain * np.sin(omega_ref * t + phase_shift)

    # Copie explicite de v_out_clean pour sauver le signal propre intact même si apply_noise
    # modifie son entrée ou crée un nouveau tableau.
    v_out_noisy = np.copy(v_out_clean)

    # applique le bruit que si une configuration non nulle et active ("none" exclu) est fournie.
    if noise_cfg is not None and noise_cfg.noise_type != "none":
        # On importe que maintenant pour éviter un import inutile.
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
        # Expose R en lecture directe depuis config.
        return self.config.R

    @property
    def C(self) -> float:
        """Capacité en Farads."""
        # Expose C en lecture directe depuis config.
        return self.config.C

    @property
    def tau(self) -> float:
        """Constante de temps tau = R * C (s)."""
        # Délègue le calcul de tau à config pour etre cohérence.
        return self.config.tau

    @property
    def fc(self) -> float:
        """Fréquence de coupure à -3 dB en Hz."""
        # Délègue le calcul de fc = 1 / (2 * pi * R * C) à config.
        return self.config.fc

    def H(self, omega: np.ndarray | float) -> np.ndarray:
        """Calcule H(w) pour une ou plusieurs pulsations."""
        # Délègue à la fonction pure compute_transfer_function pour éviter la duplication de logique.
        return compute_transfer_function(omega, self.config.R, self.config.C)

    def get_spectrum(self, freqs: np.ndarray | None = None) -> SpectrumData:
        """Génère la structure de données spectrales SpectrumData."""
        # Délègue à compute_spectrum avec la configuration du circuit.
        return compute_spectrum(self.config, freqs=freqs)

    def simulate_signal(
        self,
        f_ref: float,
        duration: float = 0.1,
        fs: float = 1e5,
        noise_cfg: NoiseConfig | None = None,
    ) -> TimeDomainData:
        """Simule la réponse temporelle du circuit RC pour une fréquence donnée."""
        # Délègue à generate_time_response en transmettant les paramètres de simulation.
        return generate_time_response(
            config=self.config,
            f_ref=f_ref,
            duration=duration,
            fs=fs,
            noise_cfg=noise_cfg,
        )