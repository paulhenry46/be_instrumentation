"""Implémentation de la vérification des relations de Kramers-Kronig (KK).

Utilise l'intégration en Valeur Principale de Cauchy (VPC) avec soustraction
de la singularité (méthode de Maclaurin), la transformée de Hilbert ou l'exclusion de fenêtre.
Gère l'extrapolation asymptotique pour limiter l'erreur de troncature sur borne finie.
"""

from typing import Literal
import numpy as np
from scipy.integrate import trapezoid 
from scipy.interpolate import CubicSpline  # Interpolation spline cubique C2 pour grille uniforme
from scipy.signal import hilbert 

from simu_lockin_kk.config import KKConfig, KKResult, SpectrumData  # Paramètres et résultats KK


# Rôle : interpole (omega, chi) sur une grille linéaire uniforme de N = 2^gamma points.
# Retour : (omega_grid, chi_interp) avec pas constant dw = w_max / N.
# Pourquoi : l'algo FFT de Bruzzoni exige un pas fréquentiel constant et N = 2^gamma.
def _interpolate_uniform_fft_grid(
    omega: np.ndarray,
    chi: np.ndarray,
    gamma: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpole un spectre sur une grille linéaire uniforme de taille N = 2**gamma."""
    # Vérifie que gamma est strictement positif pour éviter une taille de grille nulle ou négative.
    if gamma < 1:
        raise ValueError(f"gamma doit être >= 1 (obtenu : {gamma}).")

    # Convertit les entrées en tableaux float64 pour éviter des comportements bizarre.
    w_src = np.asarray(omega, dtype=np.float64)
    chi_src = np.asarray(chi, dtype=np.float64)

    # Trie les pulsations par ordre croissant afin de garantir la stricte croissance exigée par CubicSpline.
    order = np.argsort(w_src)
    w_src = w_src[order]
    chi_src = chi_src[order]

    # Calcule N = 2^gamma par décalage de bits (1 << gamma), conformément à l'article.
    n_points = 2 ** gamma # on aurait pu également utiliser 1 << gamma si on voulait un code plus performant ^^

    # Récupère la pulsation maximale et définit le pas régulier dw = w_max / N.
    w_max = float(w_src[-1])
    dw = w_max / n_points

    # Construit la grille linéaire uniforme [0, dw, 2*dw, ..., (N-1)*dw].
    omega_grid = np.arange(n_points, dtype=np.float64) * dw

    # Ajuste une spline cubique sur les données connues sans extrapoler hors bornes avec extrapolate=False.
    cs = CubicSpline(w_src, chi_src, extrapolate=False)

    # Initialise le tableau interpolé à zéro et n'évalue la spline que dans [w_src[0], w_src[-1]].
    chi_interp = np.zeros(n_points, dtype=np.float64)
    in_bounds = (omega_grid >= w_src[0]) & (omega_grid <= w_src[-1])
    chi_interp[in_bounds] = cs(omega_grid[in_bounds])

    return omega_grid, chi_interp


# Rôle : exécute l'algorithme de  de Bruzoni
# Retour : composante reconstruite (réelle ou imaginaire) évaluée sur la grille d'origine `omega`.
# Pourquoi : calculer toutes les fréquences via deux FFT au lieu d'une double boucle.
def _reconstruct_bruzzoni_fft(
    omega: np.ndarray,
    chi: np.ndarray,
    target: Literal["real", "imag"],
    gamma: int = 10,
) -> np.ndarray:
    """Reconstruit la composante conjuguée ('real' ou 'imag') selon Bruzzoni et al. (2002)."""
    # Convertit les entrées en float64.
    w_eval = np.asarray(omega, dtype=np.float64)
    chi_eval = np.asarray(chi, dtype=np.float64)

    # Étape i : interpolation par spline cubique sur N = 2^gamma points équidistants.
    omega_grid, chi_interp = _interpolate_uniform_fft_grid(w_eval, chi_eval, gamma=gamma)
    n_points = len(omega_grid)
    half_n = n_points // 2  # Sépare la première moitié (tau > 0) et la seconde moitié (tau < 0).

    # Étape ii : première FFT sur le signal interpolé.
    fft_vals = np.fft.fft(chi_interp)# ouf, on n'a pas à le faire nous meme...
    re_part = np.real(fft_vals).copy()# bonne pratique
    im_part = np.imag(fft_vals).copy()

    # Facteur de signe +2/pi pour la première moitié (tau > 0) et -2/pi pour la seconde (tau < 0).
    sign_factor = np.ones(n_points, dtype=np.float64) * (2.0 / np.pi)
    sign_factor[half_n:] = -(2.0 / np.pi)

    # Étape iii : pondération, échange des parties réelle/imaginaire et mise à zéro.
    if target == "real":
        # Reconstruction de Re[H] à partir de Im[H] (eq. 9) : on pondère la partie imaginaire,
        # on la place dans la partie réelle et on annule la partie imaginaire.
        new_re = sign_factor * im_part
        new_im = np.zeros(n_points, dtype=np.float64)
    elif target == "imag":
        # Reconstruction de Im[H] à partir de Re[H] (eq. 10) : on pondère la partie réelle,
        # on la place dans la partie imaginaire et on annule la partie réelle.
        new_re = np.zeros(n_points, dtype=np.float64)
        new_im = sign_factor * re_part
    else:
        raise ValueError(f"Cible inconnue pour Bruzzoni FFT : {target}")

    # Recompose le spectre modifié dans l'espace conjugué.
    modified_fft = new_re + 1j * new_im

    # Étape iv : FFT inverse (`np.fft.ifft` divise déjà par N, donc on multiplie par
    # pi * N = (2 * pi) * (N / 2) pour retrouver le facteur d'échelle 2*pi*(N/2) de l'article).
    # conv_result = np.real(np.fft.ifft(modified_fft)) * (np.pi * n_points)
    conv_result = np.real(np.fft.ifft(modified_fft)) * (np.pi)
    # Ré-interpole sur les pulsations `w_eval` d'origine (en restant dans la plage de la grille).
    reconstructed = np.interp(w_eval, omega_grid, conv_result)
    return reconstructed


# Rôle : prolonge la courbe (omega, chi) avant w_min et après w_max.
# Retour : (w_full, chi_full) étendus.
# Pourquoi : éviter l'effet de coupure brut d'une intégrale définie sur un domaine plus large.
def _extrapolate_spectrum(
    omega: np.ndarray,
    chi: np.ndarray,
    extrap_type: Literal["none", "asymptotic", "constant"] = "asymptotic",
) -> tuple[np.ndarray, np.ndarray]:
    """Extrapole le spectre aux basses et hautes fréquences pour minimiser les erreurs de bord."""
    # Pas d'extrapolation demandée ou pas assez de points pour estimer une tendance fiable.
    if extrap_type == "none" or len(omega) < 4:
        return omega, chi

    # Bornes inférieure et supérieure de la bande de pulsations observée.
    w_min, w_max = omega[0], omega[-1]

    # Étend sur 2 décades sous w_min et 2 décades au-dessus de w_max en échelle logarithmique.
    low_w = np.logspace(np.log10(w_min / 100.0), np.log10(w_min), num=30, endpoint=False)
    high_w = np.logspace(np.log10(w_max), np.log10(w_max * 100.0), num=30)[1:]

    if extrap_type == "asymptotic":
        # Pente linéaire locale aux plus basses pulsations pour prolonger doucement vers low_w.
        slope_low = (chi[1] - chi[0]) / (omega[1] - omega[0])
        low_chi = chi[0] + slope_low * (low_w - w_min)

        # Décroissance haute fréquence : ~ 1/w^2 pour un comportement positif type partie réelle RC,
        # ou ~ 1/w pour un comportement négatif/dissipatif type partie imaginaire RC.
        decay_power = 2.0 if np.all(chi >= 0) else 1.0
        high_chi = chi[-1] * (w_max / high_w) ** decay_power
    elif extrap_type == "constant":
        # Prolonge de façon constante à partir de chi[0] en bas et chi[-1] en haut.
        low_chi = np.full_like(low_w, chi[0])
        high_chi = np.full_like(high_w, chi[-1])
    else:
        return omega, chi

    # Assemble les trois zones (basse, mesurée, haute) en un seul couple
    w_full = np.concatenate([low_w, omega, high_w])
    chi_full = np.concatenate([low_chi, chi, high_chi])
    return w_full, chi_full

# Rôle : reconstruit la partie réelle chi'(w) à partir de la partie imaginaire chi''(w).
# Retour : tableau ndarray de la partie réelle reconstruite aux pulsations omega.
# Pourquoi : vérifier si chi'(w) est bien déterminée par chi''(w) selon Kramers-Kronig.
def reconstruct_real_from_imag(
    omega: np.ndarray,
    chi_imag: np.ndarray,
    kk_cfg: KKConfig | None = None,
) -> np.ndarray:
    r"""Reconstruit la partie réelle \chi'(\omega) à partir de la partie imaginaire \chi''(\omega).

    Équation VPC avec soustraction de singularité :
    .. math:: \chi'(\omega_0) = \frac{2}{\pi} P.V. \int_{0}^{\infty} \frac{\omega \chi''(\omega) - \omega_0 \chi''(\omega_0)}{\omega^2 - \omega_0^2} d\omega

    Parameters
    ----------
    omega : np.ndarray
        Pulsations en rad/s (triées de façon strictement croissante).
    chi_imag : np.ndarray
        Partie imaginaire mesurée \chi''(\omega).
    kk_cfg : KKConfig | None, optional
        Configuration des options KK.

    Returns
    -------
    np.ndarray
        Partie réelle reconstituée \chi'_{KK}(\omega).
    """
    # Vérifie que les deux vecteurs ont la même longueur et contiennent assez de points.
    if len(omega) != len(chi_imag):
        raise ValueError("omega et chi_imag doivent avoir la même taille.")
    if len(omega) < 3:
        raise ValueError("Au moins 3 points sont nécessaires pour l'intégration KK.")

    # Charge la configuration KK par défaut si aucune n'est fournie.
    if kk_cfg is None:
        kk_cfg = KKConfig()

    # Branche dédiée à l'algorithme de convolution rapide par FFT de Bruzzoni et al. (2002).
    if kk_cfg.integration_method == "bruzzoni_fft":
        gamma = int(getattr(kk_cfg, "fft_gamma", 10))
        return _reconstruct_bruzzoni_fft(omega, chi_imag, target="real", gamma=gamma)

    # Prépare la grille d'évaluation et le spectre extrapolé pour les méthodes d'intégration directe.
    w_eval = np.asarray(omega, dtype=np.float64)
    w_int, chi_img_int = _extrapolate_spectrum(w_eval, chi_imag, extrap_type=kk_cfg.extrapolation)

    # Cas 1 : transformée de Hilbert directe via FFT (rapide, mais besoin d' un pas uniforme).
    if kk_cfg.integration_method == "hilbert":
        analytic_sig = hilbert(chi_imag)
        return np.real(analytic_sig)

    chi_real_rec = np.zeros_like(w_eval)

    # Calcule la valeur reconstruite pour chaque pulsation cible w0.
    for i, w0 in enumerate(w_eval):
        if kk_cfg.integration_method == "cauchy_maclaurin":
            # Dénominateur w^2 - w0^2 qui s'annule lorsque w == w0 (singularité).
            diff_w2 = w_int**2 - w0**2
            # Masque des points très proches de la singularité w0.
            idx_singular = np.abs(w_int - w0) < kk_cfg.singularity_eps

            # Soustrait w0 * chi''(w0) au numérateur pour lever la singularité en w = w0.
            integrand = (w_int * chi_img_int - w0 * chi_imag[i]) / np.where(diff_w2 == 0, 1e-15, diff_w2)

            # Remplace la zone singulière par la limite continue estimée par dérivée locale.
            if np.any(idx_singular):
                dchi_dw = np.gradient(chi_imag, w_eval)[i]
                limit_val = (chi_imag[i] + w0 * dchi_dw) / (2.0 * w0) if w0 > 0 else 0.0
                integrand[idx_singular] = limit_val

            # Intègre par trapèzes avec le facteur -(2/pi)
            chi_real_rec[i] = -(2.0 / np.pi) * trapezoid(integrand, w_int)

        elif kk_cfg.integration_method == "trapezoidal_pv":
            # Exclut l'intervalle [w0 - eps, w0 + eps] autour de la singularité puis intègre par trapèzes.
            mask = np.abs(w_int - w0) >= kk_cfg.singularity_eps
            integrand = (w_int[mask] * chi_img_int[mask]) / (w_int[mask] ** 2 - w0**2)
            chi_real_rec[i] = (2.0 / np.pi) * trapezoid(integrand, w_int[mask])

    return chi_real_rec


# Rôle : reconstruit la partie imaginaire chi''(w) à partir de la partie réelle chi'(w).
# Retour : tableau ndarray de la partie imaginaire reconstruite aux pulsations omega.
# Pourquoi : vérifier la relation réciproque de Kramers-Kronig.
def reconstruct_imag_from_real(
    omega: np.ndarray,
    chi_real: np.ndarray,
    kk_cfg: KKConfig | None = None,
) -> np.ndarray:
    r"""Reconstruit la partie imaginaire \chi''(\omega) à partir de la partie réelle \chi'(\omega).

    Équation VPC :
    .. math:: \chi''(\omega_0) = -\frac{2 \omega_0}{\pi} P.V. \int_{0}^{\infty} \frac{\chi'(\omega) - \chi'(\omega_0)}{\omega^2 - \omega_0^2} d\omega

    Parameters
    ----------
    omega : np.ndarray
        Pulsations en rad/s.
    chi_real : np.ndarray
        Partie réelle mesurée \chi'(\omega).
    kk_cfg : KKConfig | None, optional
        Configuration des options KK.

    Returns
    -------
    np.ndarray
        Partie imaginaire reconstituée \chi''_{KK}(\omega).
    """
    # Vérifie la cohérence des dimensions et le nombre minimal de points.
    if len(omega) != len(chi_real):
        raise ValueError("omega et chi_real doivent avoir la même taille.")
    if len(omega) < 3:
        raise ValueError("Au moins 3 points sont nécessaires pour l'intégration KK.")

    if kk_cfg is None:
        kk_cfg = KKConfig()

    # Branche dédiée à l'algorithme de convolution rapide par FFT de Bruzzoni et al. (2002).
    if kk_cfg.integration_method == "bruzzoni_fft":
        gamma = int(getattr(kk_cfg, "fft_gamma", 10))
        return _reconstruct_bruzzoni_fft(omega, chi_real, target="imag", gamma=gamma)

    w_eval = np.asarray(omega, dtype=np.float64)
    w_int, chi_rl_int = _extrapolate_spectrum(w_eval, chi_real, extrap_type=kk_cfg.extrapolation)

    if kk_cfg.integration_method == "hilbert":
        analytic_sig = hilbert(chi_real)
        return -np.imag(analytic_sig)

    chi_imag_rec = np.zeros_like(w_eval)

    for i, w0 in enumerate(w_eval):
        if kk_cfg.integration_method == "cauchy_maclaurin":
            diff_w2 = w_int**2 - w0**2
            idx_singular = np.abs(w_int - w0) < kk_cfg.singularity_eps

            # Soustrait chi'(w0) au numérateur pour régulariser la singularité en w = w0.
            integrand = (chi_rl_int - chi_real[i]) / np.where(diff_w2 == 0, 1e-15, diff_w2)

            # Estime la limite en w = w0 à partir de la dérivée locale.
            if np.any(idx_singular):
                dchi_dw = np.gradient(chi_real, w_eval)[i]
                limit_val = dchi_dw / (2.0 * w0) if w0 > 0 else 0.0
                integrand[idx_singular] = limit_val

            # Intègre par trapèzes et multiplie par (2 * w0 / pi).
            chi_imag_rec[i] = (2.0 * w0 / np.pi) * trapezoid(integrand, w_int)

        elif kk_cfg.integration_method == "trapezoidal_pv":
            # Exclut la fenêtre [-eps, +eps] autour de w0 puis intègre par trapèzes.
            mask = np.abs(w_int - w0) >= kk_cfg.singularity_eps
            integrand = chi_rl_int[mask] / (w_int[mask] ** 2 - w0**2)
            chi_imag_rec[i] = -(2.0 * w0 / np.pi) * trapezoid(integrand, w_int[mask])

    return chi_imag_rec


# Rôle : orchestre la reconstruction KK et calcule les indicateurs d'écart (résidus, RMS, R²).
# Retour : objet KKResult.
def analyze_kramers_kronig(
    data: SpectrumData | tuple[np.ndarray, np.ndarray, np.ndarray],
    target: Literal["real", "imag"] = "real",
    kk_cfg: KKConfig | None = None,
) -> KKResult:
    """Effectue l'analyse complète de Kramers-Kronig et calcule les métriques d'erreur.

    Parameters
    ----------
    data : SpectrumData | tuple[np.ndarray, np.ndarray, np.ndarray]
        Données spectrale (soit un objet SpectrumData, soit un n-uplet (omega, X, Y)).
    target : Literal["real", "imag"], optional
        Composante à reconstruire ("real" => reconstruire X à partir de Y, "imag" => Y à partir de X).
    kk_cfg : KKConfig | None, optional
        Configuration d'intégration.

    Returns
    -------
    KKResult
        Résultat typé contenant omega, orig, recon, résidus, RMS error et R2 score.
    """
    # Accepte soit un objet SpectrumData, soit un tuple brut (omega, X, Y).
    if isinstance(data, SpectrumData):
        omega = data.omega
        chi_real = data.X
        chi_imag = data.Y
    else:
        omega, chi_real, chi_imag = data

    # Lance la reconstruction selon la composante cible choisie ("real" ou "imag").
    if target == "real":
        chi_orig = chi_real
        chi_recon = reconstruct_real_from_imag(omega, chi_imag, kk_cfg=kk_cfg)
    elif target == "imag":
        chi_orig = chi_imag
        chi_recon = reconstruct_imag_from_real(omega, chi_real, kk_cfg=kk_cfg)
    else:
        raise ValueError(f"Cible inconnue : {target}")

    # Écart point par point entre la valeur originale et la valeur reconstruite par KK.
    residuals = chi_orig - chi_recon

    # Erreur quadratique moyenne (RMS) : mesure la magnitude globale des écarts.
    rms_error = float(np.sqrt(np.mean(residuals**2)))

    # Coefficient de détermination R^2 = 1 - (SS_res / SS_tot) : compare l'erreur de reconstruction
    # à la variance totale des données (R^2 proche de 1 => très bon accord).
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((chi_orig - np.mean(chi_orig)) ** 2)
    r2_score = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0

    return KKResult(
        omega=omega,
        chi_original=chi_orig,
        chi_reconstructed=chi_recon,
        residuals=residuals,
        rms_error=rms_error,
        r2_score=r2_score,
        target_component=target,
    )


# Rôle : encapsule KKConfig dans une classe.
# Retour : méthodes verify_real(), verify_imag() et analyze().
# Pourquoi : éviter de repasser kk_cfg à chaque appel...
class KramersKronigAnalyzer:
    """Classe de service orientée objet pour la validation des relations KK."""

    def __init__(self, config: KKConfig | None = None) -> None:
        # Stocke la configuration KK fournie ou utilise la configuration par défaut.
        self.config = config or KKConfig()

    def verify_real(self, omega: np.ndarray, chi_imag: np.ndarray) -> np.ndarray:
        """Reconstruit la partie réelle à partir de la partie imaginaire."""
        # Délègue à reconstruct_real_from_imag avec la configuration de l'instance.
        return reconstruct_real_from_imag(omega, chi_imag, kk_cfg=self.config)

    def verify_imag(self, omega: np.ndarray, chi_real: np.ndarray) -> np.ndarray:
        """Reconstruit la partie imaginaire à partir de la partie réelle."""
        # Délègue à reconstruct_imag_from_real avec la configuration de l'instance.
        return reconstruct_imag_from_real(omega, chi_real, kk_cfg=self.config)

    def analyze(
        self,
        data: SpectrumData | tuple[np.ndarray, np.ndarray, np.ndarray],
        target: Literal["real", "imag"] = "real",
    ) -> KKResult:
        """Analyse complète restituant un objet KKResult."""
        # Délègue l'analyse complète à analyze_kramers_kronig avec la configuration de l'instance.
        return analyze_kramers_kronig(data, target=target, kk_cfg=self.config)