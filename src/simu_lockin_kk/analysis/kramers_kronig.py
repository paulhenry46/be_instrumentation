"""Implémentation de la vérification des relations de Kramers-Kronig (KK).

Utilise l'intégration en Valeur Principale de Cauchy (VPC) avec soustraction
de la singularité (méthode de Maclaurin), la transformée de Hilbert ou l'exclusion de fenêtre.
Gère l'extrapolation asymptotique pour limiter l'erreur de troncature sur borne finie.
"""

from typing import Literal
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import hilbert

from simu_lockin_kk.config import KKConfig, KKResult, SpectrumData

def _extrapolate_spectrum(
    omega: np.ndarray,
    chi: np.ndarray,
    extrap_type: Literal["none", "asymptotic", "constant"] = "asymptotic",
) -> tuple[np.ndarray, np.ndarray]:
    """Extrapole le spectre aux basses et hautes fréquences pour minimiser les erreurs de bord."""
    if extrap_type == "none" or len(omega) < 4:
        return omega, chi

    w_min, w_max = omega[0], omega[-1]

    # Extension logarithmique
    low_w = np.logspace(np.log10(w_min / 100.0), np.log10(w_min), num=30, endpoint=False)
    high_w = np.logspace(np.log10(w_max), np.log10(w_max * 100.0), num=30)[1:]

    if extrap_type == "asymptotic":
        # Pour circuit RC low-pass : chi' ~ 1/w^2 à haute frq, chi'' ~ -1/w à haute frq
        # Raccordement asymptotique en 1/w et 1/w^2
        slope_low = (chi[1] - chi[0]) / (omega[1] - omega[0])
        low_chi = chi[0] + slope_low * (low_w - w_min)

        decay_power = 2.0 if np.all(chi >= 0) else 1.0
        high_chi = chi[-1] * (w_max / high_w) ** decay_power
    elif extrap_type == "constant":
        low_chi = np.full_like(low_w, chi)
        high_chi = np.full_like(high_w, chi[-1])
    else:
        return omega, chi

    w_full = np.concatenate([low_w, omega, high_w])
    chi_full = np.concatenate([low_chi, chi, high_chi])
    return w_full, chi_full

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
    if len(omega) != len(chi_imag):
        raise ValueError("omega et chi_imag doivent avoir la même taille.")
    if len(omega) < 3:
        raise ValueError("Au moins 3 points sont nécessaires pour l'intégration KK.")

    if kk_cfg is None:
        kk_cfg = KKConfig()

    w_eval = np.asarray(omega, dtype=np.float64)
    w_int, chi_img_int = _extrapolate_spectrum(w_eval, chi_imag, extrap_type=kk_cfg.extrapolation)

    if kk_cfg.integration_method == "hilbert":
        # Transformée de Hilbert via FFT
        analytic_sig = hilbert(chi_imag)
        return np.real(analytic_sig)

    chi_real_rec = np.zeros_like(w_eval)

    for i, w0 in enumerate(w_eval):
        if kk_cfg.integration_method == "cauchy_maclaurin":
            # Soustraction analytique du pôle (Maclaurin)
            diff_w2 = w_int**2 - w0**2
            idx_singular = np.abs(w_int - w0) < kk_cfg.singularity_eps

            # Valeur régulière hors pôle
            integrand = (w_int * chi_img_int - w0 * chi_imag[i]) / np.where(diff_w2 == 0, 1e-15, diff_w2)

            # Traitement de la limite continue en w = w0 via dérivée numérique locale
            if np.any(idx_singular):
                dchi_dw = np.gradient(chi_imag, w_eval)[i]
                limit_val = (chi_imag[i] + w0 * dchi_dw) / (2.0 * w0) if w0 > 0 else 0.0
                integrand[idx_singular] = limit_val

            chi_real_rec[i] = -(2.0 / np.pi) * trapezoid(integrand, w_int)

        elif kk_cfg.integration_method == "trapezoidal_pv":
            # Fenêtre d'exclusion autour du pôle [w0 - eps, w0 + eps]
            mask = np.abs(w_int - w0) >= kk_cfg.singularity_eps
            integrand = (w_int[mask] * chi_img_int[mask]) / (w_int[mask] ** 2 - w0**2)
            chi_real_rec[i] = (2.0 / np.pi) * trapezoid(integrand, w_int[mask])

    return chi_real_rec

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
    if len(omega) != len(chi_real):
        raise ValueError("omega et chi_real doivent avoir la même taille.")
    if len(omega) < 3:
        raise ValueError("Au moins 3 points sont nécessaires pour l'intégration KK.")

    if kk_cfg is None:
        kk_cfg = KKConfig()

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

            integrand = (chi_rl_int - chi_real[i]) / np.where(diff_w2 == 0, 1e-15, diff_w2)

            if np.any(idx_singular):
                dchi_dw = np.gradient(chi_real, w_eval)[i]
                limit_val = dchi_dw / (2.0 * w0) if w0 > 0 else 0.0
                integrand[idx_singular] = limit_val

            chi_imag_rec[i] = (2.0 * w0 / np.pi) * trapezoid(integrand, w_int)

        elif kk_cfg.integration_method == "trapezoidal_pv":
            mask = np.abs(w_int - w0) >= kk_cfg.singularity_eps
            integrand = chi_rl_int[mask] / (w_int[mask] ** 2 - w0**2)
            chi_imag_rec[i] = -(2.0 * w0 / np.pi) * trapezoid(integrand, w_int[mask])

    return chi_imag_rec

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
    if isinstance(data, SpectrumData):
        omega = data.omega
        chi_real = data.X
        chi_imag = data.Y
    else:
        omega, chi_real, chi_imag = data

    if target == "real":
        chi_orig = chi_real
        chi_recon = reconstruct_real_from_imag(omega, chi_imag, kk_cfg=kk_cfg)
    elif target == "imag":
        chi_orig = chi_imag
        chi_recon = reconstruct_imag_from_real(omega, chi_real, kk_cfg=kk_cfg)
    else:
        raise ValueError(f"Cible inconnue : {target}")

    residuals = chi_orig - chi_recon
    rms_error = float(np.sqrt(np.mean(residuals**2)))

    # Calcul du R2 score (coefficient de détermination)
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

class KramersKronigAnalyzer:
    """Classe de service orientée objet pour la validation des relations KK."""

    def __init__(self, config: KKConfig | None = None) -> None:
        self.config = config or KKConfig()

    def verify_real(self, omega: np.ndarray, chi_imag: np.ndarray) -> np.ndarray:
        """Reconstruit la partie réelle à partir de la partie imaginaire."""
        return reconstruct_real_from_imag(omega, chi_imag, kk_cfg=self.config)

    def verify_imag(self, omega: np.ndarray, chi_real: np.ndarray) -> np.ndarray:
        """Reconstruit la partie imaginaire à partir de la partie réelle."""
        return reconstruct_imag_from_real(omega, chi_real, kk_cfg=self.config)

    def analyze(
        self,
        data: SpectrumData | tuple[np.ndarray, np.ndarray, np.ndarray],
        target: Literal["real", "imag"] = "real",
    ) -> KKResult:
        """Analyse complète restituant un objet KKResult."""
        return analyze_kramers_kronig(data, target=target, kk_cfg=self.config)