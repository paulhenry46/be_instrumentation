"""Fonctions de visualisation graphique pour le projet simu_lockin_kk.

Ce module contient des générateurs de figures matplotlib pour :
- Les signaux temporels (entrée, sortie propre, sortie bruitée).
- Les diagrammes de Bode (Amplitude en dB et Phase en degrés).
- Les comparaisons Kramers-Kronig (superposition originale/reconstruite et résidus).
"""

from pathlib import Path  
import matplotlib.pyplot as plt 
from matplotlib.figure import Figure 
import numpy as np 

from simu_lockin_kk.config import KKResult, SpectrumData, TimeDomainData  # Conteneurs de données du projet

# Backend non-interactif
#matplotlib.use("Agg")

# Rôle : trace les signaux temporels d'entrée et de sortie (propre et bruitée).
# Retour : matplotlib.figure.Figure.
# Pourquoi : vérifier visuellement l'amplitude, le déphasage et le bruit sur une fenêtre lisible.
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
    # Extrait les séries temporelles et la fréquence de référence depuis l'objet TimeDomainData.
    t = time_data.t
    v_in = time_data.v_in
    v_clean = time_data.v_out_clean
    v_noisy = time_data.v_out_noisy
    f_ref = time_data.f_ref

    # Restreint la fenêtre affichée aux num_periods premières périodes pour rendre les oscillations
    # lisibles si la simu totale beaucoup plus longtemps.
    if num_periods is not None and num_periods > 0:
        period = 1.0 / f_ref  # Durée d'une période T = 1 / f_ref en secondes
        t_max = num_periods * period  # Borne temporelle supérieure de la fenêtre affichée
        mask = t <= t_max  # Masque booléen sélectionnant les échantillons dans [0, t_max]
        t = t[mask]
        v_in = v_in[mask]
        v_clean = v_clean[mask]
        v_noisy = v_noisy[mask]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    # Trace d'abord le signal bruité en semi-transparence (alpha=0.4) pour qu'il n'écrase pas
    # visuellement les courbes d'entrée et de sortie théorique ; t * 1e3 convertit les s en ms.
    ax.plot(t * 1e3, v_noisy, color="#d62728", alpha=0.4, linewidth=1.0, label="v_out (bruité)")
    ax.plot(t * 1e3, v_in, color="#1f77b4", linewidth=1.8, label="v_in")
    ax.plot(t * 1e3, v_clean, color="#2ca02c", linewidth=2.0, linestyle="--", label="v_out (théo)")

    ax.set_xlabel("Temps t (ms)", fontsize=11)
    ax.set_ylabel("Tension (V)", fontsize=11)
    ax.set_title(f"Réponse temporelle du circuit RC (f_ref = {f_ref:.1f} Hz)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="upper right", framealpha=0.9)

    # Ajuste automatiquement les marges internes pour éviter tout chevauchement de texte.
    plt.tight_layout()

    # Sauvegarde optionnelle sur disque (200 dpi et bbox_inches="tight" pour ne rien rogner).
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    # Affichage interactif optionnel (désactivé par défaut pour les exécutions automatisées/tests).
    if show:
        plt.show()

    return fig


# Rôle : trace le gain en dB et la phase en degrés sur une échelle fréquentielle logarithmique.
# Retour : matplotlib.figure.Figure.
# Pourquoi : analyser la réponse fréquentielle et repérer la coupure à -3 dB / -45°.
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
    # Sélectionne l'abscisse (pulsation omega en rad/s ou fréquence f en Hz) et son étiquette.
    x_data = spectrum.omega if use_omega else spectrum.freqs
    x_label = "Pulsation w (rad/s)" if use_omega else "Fréquence f (Hz)"

    # Convertit l'amplitude R en décibels via 20 * log10(R) ; le plancher 1e-12 évite log10(0)
    # ou des valeurs négatives infinies (-inf) en cas d'amplitude nulle ou bruitée négative.
    gain_db = 20.0 * np.log10(np.maximum(spectrum.R, 1e-12))

    # Convertit la phase de rad en degrés...
    phase_deg = np.degrees(spectrum.phase)

    # Crée 2 sous-graphiques empilés avec même axe X
    fig, (ax_gain, ax_phase) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, dpi=150)

    #Gain : échelle semi-logarithmique pour décades,
    # avec repère horizontal à -3.01 dB (atténuation en puissance de 1/2, gain en tension 1/sqrt(2)).
    ax_gain.semilogx(x_data, gain_db, color="#1f77b4", linewidth=2.0, label="Gain |H|")
    ax_gain.axhline(-3.01, color="r", linestyle="--", alpha=0.7, label="-3 dB")
    ax_gain.set_ylabel("Gain (dB)", fontsize=11)
    ax_gain.set_title("Diagramme de Bode", fontsize=12, fontweight="bold")
    ax_gain.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_gain.legend(loc="lower left")

    # 2. Sous-graphique de la Phase : échelle semi-logarithmique avec repère à -45°, déphasage
    # caractéristique d'un filtre RC passe-bas du premier ordre à sa fréquence de coupure.
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

# Rôle : compare la courbe originale et la reconstruction KK, puis affiche les résidus.
# Retour : matplotlib.figure.Figure.
# Pourquoi : contrôler visuellement et numériquement (RMS, R²) la validité Kramers-Kronig.
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
    # KKResult stocke l'axe fréquentiel en pulsation omega (rad/s) ; convertit en Hz via omega / (2*pi)
    # si use_omega est False.
    x_data = result.omega if use_omega else result.omega / (2.0 * np.pi)
    x_label = "Pulsation ω (rad/s)" if use_omega else "Fréquence f (Hz)"

    # Nomme explicitement la composante analysée selon que KK a reconstruit la partie réelle ("real")
    # ou la partie imaginaire ("imag").
    target_name = "Partie Réelle χ' (X)" if result.target_component == "real" else "Partie Imaginaire χ'' (Y)"

    # Deux sous-graphiques en écart 3:1 (height_ratios=[3, 1]) : la superposition occupe 75% de la
    # hauteur et les résidus 25%, disposition classique pour l'analyse d'ajustement/reconstruction.
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}, dpi=150
    )

    # 1. Graphique principal : points discrets ("o") pour l'original et trait continu ("-")
    # pour la reconstruction KK
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

    # Résidus, comme sur regressi
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


from typing import Literal
from simu_lockin_kk.config import KKConfig, KKResult, SpectrumData
from simu_lockin_kk.analysis.kramers_kronig import analyze_kramers_kronig


# Rôle : trace la reconstruction KK avec extrapolation vs sans extrapolation face à l'original.
# Retour : objet Figure matplotlib.
# Pourquoi : visualiser l'impact de l'extrapolation sur les effets de bord et la précision (RMS, R²).
def plot_extrapolation_comparison(
    source: SpectrumData | tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[KKResult, KKResult],
    target: Literal["real", "imag"] = "real",
    kk_cfg: KKConfig | None = None,
    use_omega: bool = True,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Compare la reconstruction Kramers-Kronig avec et sans extrapolation.

    Parameters
    ----------
    source : SpectrumData | tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[KKResult, KKResult]
        Données spectrales (SpectrumData ou (omega, X, Y)) ou couple (res_avec_extrap, res_sans_extrap).
    target : Literal["real", "imag"], optional
        Composante reconstruite ("real" ou "imag"), utilisé si source fournit les données brutes.
    kk_cfg : KKConfig | None, optional
        Configuration KK (la méthode d'extrapolation choisie sera comparée à "none").
    use_omega : bool, optional
        Si True, axe X en rad/s ; sinon en Hz.
    save_path : str | Path | None, optional
        Chemin de sauvegarde optionnel.
    show : bool, optional
        Si True, affiche la figure.

    Returns
    -------
    Figure
        Objet Figure matplotlib généré.
    """
    # Cas 1 : l'utilisateur fournit directement deux objets KKResult (avec extrap, sans extrap).
    if (
        isinstance(source, tuple)
        and len(source) == 2
        and isinstance(source[0], KKResult)
        and isinstance(source[1], KKResult)
    ):
        res_extrap, res_no_extrap = source
        omega = res_extrap.omega
        chi_orig = res_extrap.chi_original
        chi_extrap = res_extrap.chi_reconstructed
        chi_no_extrap = res_no_extrap.chi_reconstructed
        target_name = (
            "Partie Réelle χ' (X)" if res_extrap.target_component == "real" else "Partie Imaginaire χ'' (Y)"
        )
    else:
        # Cas 2 : on calcule les deux analyses à partir de SpectrumData ou (omega, X, Y).
        base_cfg = kk_cfg or KKConfig()
        extrap_mode = base_cfg.extrapolation if base_cfg.extrapolation != "none" else "asymptotic"

        cfg_extrap = KKConfig(
            integration_method=base_cfg.integration_method,
            extrapolation=extrap_mode,
            singularity_eps=base_cfg.singularity_eps,
        )
        cfg_no_extrap = KKConfig(
            integration_method=base_cfg.integration_method,
            extrapolation="none",
            singularity_eps=base_cfg.singularity_eps,
        )

        res_extrap = analyze_kramers_kronig(source, target=target, kk_cfg=cfg_extrap)
        res_no_extrap = analyze_kramers_kronig(source, target=target, kk_cfg=cfg_no_extrap)

        omega = res_extrap.omega
        chi_orig = res_extrap.chi_original
        chi_extrap = res_extrap.chi_reconstructed
        chi_no_extrap = res_no_extrap.chi_reconstructed
        target_name = "Partie Réelle χ' (X)" if target == "real" else "Partie Imaginaire χ'' (Y)"

    # Convertit l'axe des abscisses en rad/s ou en Hz selon use_omega.
    x_data = omega if use_omega else omega / (2.0 * np.pi)
    x_label = "Pulsation ω (rad/s)" if use_omega else "Fréquence f (Hz)"

    # Crée deux sous-graphiques alignés sur l'axe X : courbes en haut, écarts à l'original en bas.
    fig, (ax_main, ax_err) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}, dpi=150
    )

    # Courbe de référence (données originales) en points discrets.
    ax_main.semilogx(x_data, chi_orig, "o", color="#1f77b4", markersize=4, label="Original")

    # Reconstruction avec extrapolation.
    ax_main.semilogx(
        x_data,
        chi_extrap,
        "-",
        color="#2ca02c",
        linewidth=2.0,
        label=f"Avec extrapolation (RMS={res_extrap.rms_error:.3e}, R²={res_extrap.r2_score:.4f})",
    )

    # Reconstruction sans extrapolation (en pointillés pour bien distinguer les deux courbes).
    ax_main.semilogx(
        x_data,
        chi_no_extrap,
        "--",
        color="#d62728",
        linewidth=1.8,
        label=f"Sans extrapolation (RMS={res_no_extrap.rms_error:.3e}, R²={res_no_extrap.r2_score:.4f})",
    )

    ax_main.set_ylabel(target_name, fontsize=11)
    ax_main.set_title(
        f"Effet de l'extrapolation sur Kramers-Kronig - {target_name}",
        fontsize=12,
        fontweight="bold",
    )
    ax_main.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_main.legend(loc="best", framealpha=0.9)

    # Sous-graphique inférieur : compare les résidus (Original - Reconstruit) des deux approches.
    ax_err.semilogx(x_data, res_extrap.residuals, "-", color="#2ca02c", linewidth=1.5, label="Résidus (avec extrap)")
    ax_err.semilogx(
        x_data, res_no_extrap.residuals, "--", color="#d62728", linewidth=1.5, label="Résidus (sans extrap)"
    )
    ax_err.axhline(0.0, color="black", linestyle=":", alpha=0.7)
    ax_err.set_xlabel(x_label, fontsize=11)
    ax_err.set_ylabel("Résidus Δ", fontsize=11)
    ax_err.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_err.legend(loc="best")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig




from simu_lockin_kk.config import KKConfig, SpectrumData
from simu_lockin_kk.analysis.kramers_kronig import _extrapolate_spectrum


# Rôle : trace les points originaux et les points ajoutés par _extrapolate_spectrum.
# Retour : objet Figure matplotlib.
# Pourquoi : vérifier visuellement le raccordement des extrapolations basse et haute fréquence.
def plot_extrapolated_points(
    data: SpectrumData | tuple[np.ndarray, np.ndarray, np.ndarray],
    component: str = "imag",
    kk_cfg: KKConfig | None = None,
    use_omega: bool = True,
    save_path: str | None = None,
    show: bool = False,
) -> Figure:
    """Affiche les points spectraux d'origine et les points ajoutés par extrapolation."""
    cfg = kk_cfg or KKConfig()

    # Récupère omega et la composante choisie ("real" => X, "imag" => Y).
    if isinstance(data, SpectrumData):
        omega = np.asarray(data.omega, dtype=np.float64)
        chi = np.asarray(data.X if component == "real" else data.Y, dtype=np.float64)
    else:
        omega, chi_real, chi_imag = data
        omega = np.asarray(omega, dtype=np.float64)
        chi = np.asarray(chi_real if component == "real" else chi_imag, dtype=np.float64)

    # Calcule le spectre complet (points originaux + points extrapolés).
    w_full, chi_full = _extrapolate_spectrum(omega, chi, extrap_type=cfg.extrapolation)

    # Masques pour séparer les points extrapolés des points d'origine.
    n_orig = len(omega)
    n_added_low = int(np.sum(w_full < omega[0]))
    n_added_high = len(w_full) - n_added_low - n_orig

    low_w, low_chi = w_full[:n_added_low], chi_full[:n_added_low]
    high_w, high_chi = w_full[n_added_low + n_orig :], chi_full[n_added_low + n_orig :]

    # Conversion optionnelle de la pulsation omega (rad/s) en fréquence (Hz).
    scale_x = 1.0 if use_omega else (1.0 / (2.0 * np.pi))
    x_label = "Pulsation ω (rad/s)" if use_omega else "Fréquence f (Hz)"
    comp_label = "Partie réelle χ' (X)" if component == "real" else "Partie imaginaire χ'' (Y)"

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    # Courbe continue globale pour voir la continuité du raccordement.
    ax.semilogx(w_full * scale_x, chi_full, "-", color="#888888", alpha=0.5, linewidth=1.2, label="Profil extrapolé total")

    # Points extrapolés basse et haute fréquence.
    if len(low_w) > 0:
        ax.semilogx(low_w * scale_x, low_chi, "x", color="#ff7f0e", markersize=5, label=f"Ajout bas ({cfg.extrapolation})")
    if len(high_w) > 0:
        ax.semilogx(high_w * scale_x, high_chi, "^", color="#2ca02c", markersize=5, label=f"Ajout haut ({cfg.extrapolation})")

    # Points d'origine mis en évidence par-dessus.
    ax.semilogx(omega * scale_x, chi, "o", color="#1f77b4", markersize=5, label="Points originaux")

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(comp_label, fontsize=11)
    ax.set_title(f"Points originaux vs extrapolation ({cfg.extrapolation}) - {comp_label}", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", alpha=0.7)
    ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig