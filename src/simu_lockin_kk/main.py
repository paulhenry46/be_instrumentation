"""Point d'entrée principal (CLI) pour la simulation Lock-in et Kramers-Kronig.

Exécute la chaîne complète de traitement :
1. Modélisation physique du circuit RC.
2. Démodulation synchrone par Lock-in (sweep fréquentiel).
3. Reconstruction et vérification des relations de Kramers-Kronig (VPC).
4. Affichage et sauvegarde des graphiques de synthèse.
"""

import argparse
from pathlib import Path

from simu_lockin_kk.analysis.kramers_kronig import analyze_kramers_kronig
from simu_lockin_kk.config import KKConfig, NoiseConfig, RCCircuitConfig
from simu_lockin_kk.physics.rc_circuit import generate_time_response
from simu_lockin_kk.signal.lockin import demodulate_sweep
from simu_lockin_kk.visualization.plots import (
    plot_bode,
    plot_kramers_kronig_comparison,
    plot_time_domain,
)

def run_pipeline(
    R: float = 1000.0,
    C: float = 1e-6,
    std_white: float = 0.02,
    num_freqs: int = 100,
    output_dir: str | Path = "outputs",
    save_plots: bool = True,
) -> None:
    """Exécute la simulation complète de bout en bout."""
    out_path = Path(output_dir)
    if save_plots:
        out_path.mkdir(parents=True, exist_ok=True)

    print("=========================================================")
    print("  SIMULATION AMPLIFICATEUR SYNCHRONE & KRAMERS-KRONIG   ")
    print("=========================================================\n")

    # 1. Configuration du circuit RC
    rc_cfg = RCCircuitConfig(R=R, C=C, num_freqs=num_freqs, f_min=0.1, f_max=1e5)
    print(f"[1/4] Circuit RC configuré : R = {rc_cfg.R:.1f} Ω, C = {rc_cfg.C * 1e6:.2f} µF")
    print(f"      -> Constante de temps τ = {rc_cfg.tau * 1e3:.3f} ms")
    print(f"      -> Fréquence de coupure fc = {rc_cfg.fc:.2f} Hz\n")

    # 2. Simulation d'un signal temporel au voisinage de fc
    f_test = rc_cfg.fc
    noise_cfg = NoiseConfig(noise_type="white", std_white=std_white, seed=42)
    time_data = generate_time_response(rc_cfg, f_ref=f_test, duration=0.05, fs=1e5, noise_cfg=noise_cfg)
    print(f"[2/4] Signal temporel simulé à f = {f_test:.1f} Hz avec bruit σ = {std_white} V.")

    if save_plots:
        plot_time_domain(time_data, num_periods=4.0, save_path=out_path / "01_time_domain.png")
        print(f"      -> Figure sauvegardée : {out_path / '01_time_domain.png'}")

    # 3. Balayage fréquentiel et démodulation synchrone (Lock-in)
    print("\n[3/4] Exécution du balayage fréquentiel par démodulation synchrone (Lock-in)...")
    spectrum = demodulate_sweep(
        rc_config=rc_cfg,
        duration=0.05,
        fs=1e5,
        noise_cfg=noise_cfg,
        lpf_cutoff=10.0,
        lpf_order=2,
    )
    print(f"      -> Spectre mesuré sur {len(spectrum.freqs)} fréquences (de {spectrum.freqs[0]:.2f} Hz à {spectrum.freqs[-1]:.1f} Hz).")

    if save_plots:
        plot_bode(spectrum, use_omega=False, save_path=out_path / "02_bode_diagram.png")
        print(f"      -> Figure sauvegardée : {out_path / '02_bode_diagram.png'}")

    # 4. Vérification de Kramers-Kronig (VPC)
    print("\n[4/4] Vérification des relations de Kramers-Kronig (Valeur Principale de Cauchy)...")
    kk_cfg = KKConfig(integration_method="cauchy_maclaurin", extrapolation="asymptotic")
    kk_result = analyze_kramers_kronig(spectrum, target="real", kk_cfg=kk_cfg)

    print("      -> Reconstruction de Re{H(ω)} à partir de Im{H(ω)} complétée.")
    print(f"      -> Erreur RMS relative : {kk_result.rms_error:.4e}")
    print(f"      -> Coefficient R²      : {kk_result.r2_score:.5f}")

    if save_plots:
        plot_kramers_kronig_comparison(kk_result, use_omega=True, save_path=out_path / "03_kramers_kronig_verification.png")
        print(f"      -> Figure sauvegardée : {out_path / '03_kramers_kronig_verification.png'}")

    print("\n=========================================================")
    print("  SIMULATION TERMINÉE AVEC SUCCÈS")
    print("=========================================================\n")

def main() -> None:
    """Point d'entrée CLI avec gestion des arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Simulation Python d'un amplificateur synchrone et Kramers-Kronig"
    )
    parser.add_argument("-R", "--resistance", type=float, default=1000.0, help="Résistance R en Ohms (défaut: 1000.0)")
    parser.add_argument("-C", "--capacitance", type=float, default=1e-6, help="Capacité C en Farads (défaut: 1e-6)")
    parser.add_argument("--noise", type=float, default=0.02, help="Écart-type du bruit blanc en Volts (défaut: 0.02)")
    parser.add_argument("-n", "--num-freqs", type=int, default=100, help="Nombre de points de fréquence (défaut: 100)")
    parser.add_argument("-o", "--output-dir", type=str, default="outputs", help="Dossier de sortie pour les graphiques")

    args = parser.parse_args()

    run_pipeline(
        R=args.resistance,
        C=args.capacitance,
        std_white=args.noise,
        num_freqs=args.num_freqs,
        output_dir=args.output_dir,
        save_plots=True,
    )

if __name__ == "__main__":
    main()