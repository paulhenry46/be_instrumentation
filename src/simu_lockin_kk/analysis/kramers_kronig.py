"""Implémentation de la vérification des relations de Kramers-Kronig (KK).

Utilise l'intégration en Valeur Principale de Cauchy (VPC) avec soustraction
de la singularité (méthode de Maclaurin), la transformée de Hilbert ou l'exclusion de fenêtre.
Gère l'extrapolation asymptotique pour limiter l'erreur de troncature sur borne finie.
"""

from typing import Literal
import numpy as np
from scipy.integrate import trapezoid 
from scipy.signal import hilbert 

from simu_lockin_kk.config import KKConfig, KKResult, SpectrumData  # Paramètres et résultats KK


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

    # Crée une extension logarithmique sur 2 décades sous w_min (sans répéter w_min à la fin)
    # et 2 décades au-dessus de w_max (sans répéter w_max au début).
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
        # Prolonge de façon constante : attention ici à vérifier la forme de low_chi si utilisé
        # (np.full_like(low_w, chi[0]) maintient une valeur scalaire constante aux basses pulsations).
        low_chi = np.full_like(low_w, chi)
        high_chi = np.full_like(high_w, chi[-1])
    else:
        return omega, chi

    # Concatène les segments basse pulsation, mesure centrale et haute pulsation dans l'ordre croissant.
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

    # Convertit omega d'évaluation en float64 et étend le domaine d'intégration si configuré.
    w_eval = np.asarray(omega, dtype=np.float64)
    w_int, chi_img_int = _extrapolate_spectrum(w_eval, chi_imag, extrap_type=kk_cfg.extrapolation)

    # Cas 1 : transformée de Hilbert directe via FFT (rapide, mais besoin d' un pas uniforme).
    if kk_cfg.integration_method == "hilbert":
        analytic_sig = hilbert(chi_imag)
        return np.real(analytic_sig)

    # Initialise le tableau de sortie de même forme que w_eval.
    chi_real_rec = np.zeros_like(w_eval)

    # Calcule la valeur reconstruite pour chaque pulsation cible w0.
    for i, w0 in enumerate(w_eval):
        if kk_cfg.integration_method == "cauchy_maclaurin":
            # Dénominateur w^2 - w0^2 qui s'annule lorsque w == w0 (singularité).
            diff_w2 = w_int**2 - w0**2

            # Repère les points très proches de w0 (dans un rayon singularity_eps).
            idx_singular = np.abs(w_int - w0) < kk_cfg.singularity_eps

            # Soustrait w0 * chi''(w0) au numérateur pour régulariser la singularité en w = w0 ;
            # np.where évite une division par zéro exacte avant remplacement sur idx_singular.
            integrand = (w_int * chi_img_int - w0 * chi_imag[i]) / np.where(diff_w2 == 0, 1e-15, diff_w2)

            # Remplace la zone singulière par la limite continue estimée via la dérivée locale.
            if np.any(idx_singular):
                dchi_dw = np.gradient(chi_imag, w_eval)[i]
                limit_val = (chi_imag[i] + w0 * dchi_dw) / (2.0 * w0) if w0 > 0 else 0.0
                integrand[idx_singular] = limit_val

            # Intègre numériquement par la méthode des trapèzes sur w_int, avec la formule de KK.
            chi_real_rec[i] = -(2.0 / np.pi) * trapezoid(integrand, w_int)

        elif kk_cfg.integration_method == "trapezoidal_pv":
            # Exclut les points trop proches de w0 pour éviter les grandes valeurs près du pôle,
            # puis intègre sur les points restants par trapèzes.
            mask = np.abs(w_int - w0) >= kk_cfg.singularity_eps
            integrand = (w_int[mask] * chi_img_int[mask]) / (w_int[mask] ** 2 - w0**2)
            chi_real_rec[i] = (2.0 / np.pi) * trapezoid(integrand, w_int[mask])

    return chi_real_rec


# Rôle : reconstruit la partie imaginaire chi''(w) à partir de la partie réelle chi'(w).
# Retour : tableau ndarray de la partie imaginaire reconstruite aux pulsations omega.
# Pourquoi : vérifier la relation réciproque de Kramers-Kronig (absorption depuis dispersion).
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

    # Prépare les vecteurs de pulsation et d'extrapolation.
    w_eval = np.asarray(omega, dtype=np.float64)
    w_int, chi_rl_int = _extrapolate_spectrum(w_eval, chi_real, extrap_type=kk_cfg.extrapolation)

    # Méthode spectrale par transformée de Hilbert (partie imaginaire du signal analytique).
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

            # Valeur limite autour de w0 estimée par dérivée numérique.
            if np.any(idx_singular):
                dchi_dw = np.gradient(chi_real, w_eval)[i]
                limit_val = dchi_dw / (2.0 * w0) if w0 > 0 else 0.0
                integrand[idx_singular] = limit_val

            # Intègre par trapèzes et multiplie par le préfacteur d'intégration.
            chi_imag_rec[i] = (2.0 * w0 / np.pi) * trapezoid(integrand, w_int)

        elif kk_cfg.integration_method == "trapezoidal_pv":
            # Intègre hors de la fenêtre [-eps, +eps] autour de w0 par méthode des trapèzes.
            mask = np.abs(w_int - w0) >= kk_cfg.singularity_eps
            integrand = chi_rl_int[mask] / (w_int[mask] ** 2 - w0**2)
            chi_imag_rec[i] = -(2.0 * w0 / np.pi) * trapezoid(integrand, w_int[mask])

    return chi_imag_rec


# Rôle : orchestre la reconstruction KK et calcule les indicateurs d'écart (résidus, RMS, R²).
# Retour : objet KKResult ok
# Pourquoi : résumer en un seul objet si la reconstruction KK est fidèle aux données.
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

    # Retourne tous les résultats structurés dans KKResult.
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