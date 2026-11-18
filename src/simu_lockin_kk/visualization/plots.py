"""Fonctions de visualisation graphique pour le projet simu_lockin_kk.

Ce module contient des générateurs de figures matplotlib pour :
- Les signaux temporels (entrée, sortie propre, sortie bruitée).
- Les diagrammes de Bode (Amplitude en dB et Phase en degrés).
- Les comparaisons Kramers-Kronig (superposition originale/reconstruite et résidus).
"""

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from simu_lockin_kk.config import KKResult, SpectrumData, TimeDomainData

# Backend non-interactif par sécurité
#matplotlib.use("Agg")

def plot_time_domain(
    time_data: TimeDomainData,
    num_periods: float | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Génère le graphique temporel de l'excitation et de la réponse du circuit RC.

    Parameters
    ----------
    time_data : TimeDomainData
        Données temporelles (t, v_in, v_out_clean, v_out_noisy).
    num_periods : float | None, optional
        Si spécifié, restreint l'affichage aux num_periods premières périodes du signal.
    save_path : str | Path | None, optional
        Chemin où sauvegarder l'image (.png, .pdf, etc.).
    show : bool, optional
        Si True, affiche la figure (défaut : False).

    Returns
    -------
    Figure
        Objet Figure matplotlib généré.
    """
    t = time_data.t
    v_in = time_data.v_in
    v_clean = time_data.v_out_clean
    v_noisy = time_data.v_out_noisy
    f_ref = time_data.f_ref

    if num_periods is not None and num_periods > 0:
        period = 1.0 / f_ref
        t_max = num_periods * period
        mask = t <= t_max
        t = t[mask]
        v_in = v_in[mask]
        v_clean = v_clean[mask]
        v_noisy = v_noisy[mask]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    # Signal bruité en arrière-plan transparent
    ax.plot(t * 1e3, v_noisy, color="#d62728", alpha=0.4, linewidth=1.0, label="v_out (bruité)")
    ax.plot(t * 1e3, v_in, color="#1f77b4", linewidth=1.8, label="v_in (excitation)")
    ax.plot(t * 1e3, v_clean, color="#2ca02c", linewidth=2.0, linestyle="--", label="v_out (théorique/propre)")

    ax.set_xlabel("Temps t (ms)", fontsize=11)
    ax.set_ylabel("Tension (V)", fontsize=11)
    ax.set_title(f"Réponse temporelle du circuit RC (f_ref = {f_ref:.1f} Hz)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig

def plot_bode(
    spectrum: SpectrumData,
    use_omega: bool = False,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Génère le diagramme de Bode (Gain en dB et Phase en degrés).

    Parameters
    ----------
    spectrum : SpectrumData
        Données spectrale complexes.
    use_omega : bool, optional
        Si True, affiche l'axe des X en pulsation w (rad/s), sinon en fréquence f (Hz).
    save_path : str | Path | None, optional
        Chemin de sauvegarde du fichier image.
    show : bool, optional
        Si True, affiche la figure.

    Returns
    -------
    Figure
        Objet Figure matplotlib généré.
    """
    x_data = spectrum.omega if use_omega else spectrum.freqs
    x_label = "Pulsation ω (rad/s)" if use_omega else "Fréquence f (Hz)"

    gain_db = 20.0 * np.log10(np.maximum(spectrum.R, 1e-12))
    phase_deg = np.degrees(spectrum.phase)

    fig, (ax_gain, ax_phase) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, dpi=150)

    # 1. Sous-graphique du Gain
    ax_gain.semilogx(x_data, gain_db, color="#1f77b4", linewidth=2.0, label="Gain |H|")
    ax_gain.axhline(-3.01, color="r", linestyle="--", alpha=0.7, label="-3 dB")
    ax_gain.set_ylabel("Gain (dB)", fontsize=11)
    ax_gain.set_title("Diagramme de Bode - Réponse fréquentielle", fontsize=12, fontweight="bold")
    ax_gain.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_gain.legend(loc="lower left")

    # 2. Sous-graphique de la Phase
    ax_phase.semilogx(x_data, phase_deg, color="#ff7f0e", linewidth=2.0, label="Phase θ")
    ax_phase.axhline(-45.0, color="r", linestyle="--", alpha=0.7, label="-45° (Coupure)")
    ax_phase.set_xlabel(x_label, fontsize=11)
    ax_phase.set_ylabel("Phase (°)", fontsize=11)
    ax_phase.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_phase.legend(loc="lower left")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig

def plot_kramers_kronig_comparison(
    result: KKResult,
    use_omega: bool = True,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Génère le graphique de comparaison entre la mesure et la reconstruction Kramers-Kronig.

    Parameters
    ----------
    result : KKResult
        Résultat de l'analyse KK (omega, chi_original, chi_reconstructed, residuals, rms_error, r2_score).
    use_omega : bool, optional
        Si True, axe x en rad/s, sinon en Hz.
    save_path : str | Path | None, optional
        Chemin de sauvegarde du fichier image.
    show : bool, optional
        Si True, affiche la figure.

    Returns
    -------
    Figure
        Objet Figure matplotlib.
    """
    x_data = result.omega if use_omega else result.omega / (2.0 * np.pi)
    x_label = "Pulsation ω (rad/s)" if use_omega else "Fréquence f (Hz)"

    target_name = "Partie Réelle χ' (X)" if result.target_component == "real" else "Partie Imaginaire χ'' (Y)"

    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}, dpi=150
    )

    # 1. Graphique principal : Superposition
    ax_main.semilogx(x_data, result.chi_original, "o", color="#1f77b4", markersize=4, label="Mesuré / Original")
    ax_main.semilogx(
        x_data,
        result.chi_reconstructed,
        "-",
        color="#d62728",
        linewidth=2.0,
        label=f"Reconstruit KK (RMS={result.rms_error:.3e}, R²={result.r2_score:.4f})",
    )
    ax_main.set_ylabel(target_name, fontsize=11)
    ax_main.set_title(f"Vérification des Relations de Kramers-Kronig - {target_name}", fontsize=12, fontweight="bold")
    ax_main.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_main.legend(loc="best", framealpha=0.9)

    # 2. Graphique des résidus
    ax_res.semilogx(x_data, result.residuals, color="#2ca02c", linewidth=1.5, label="Résidus Δ = Orig - KK")
    ax_res.axhline(0.0, color="black", linestyle="--", alpha=0.7)
    ax_res.set_xlabel(x_label, fontsize=11)
    ax_res.set_ylabel("Résidus Δ", fontsize=11)
    ax_res.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_res.legend(loc="best")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig