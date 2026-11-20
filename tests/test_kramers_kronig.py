"""Tests unitaires pour le module analysis.kramers_kronig."""

import unittest
import numpy as np

from simu_lockin_kk.config import KKConfig, RCCircuitConfig
from simu_lockin_kk.physics.rc_circuit import compute_spectrum
from simu_lockin_kk.analysis.kramers_kronig import (
    KramersKronigAnalyzer,
    analyze_kramers_kronig,
    reconstruct_imag_from_real,
    reconstruct_real_from_imag,
)

class TestKramersKronig(unittest.TestCase):
    """Tests unitaires pour la vérification des relations de Kramers-Kronig (VPC)."""

    def setUp(self) -> None:
        """Génère un spectre RC théorique parfait sur 6 décades de fréquence."""
        self.rc_cfg = RCCircuitConfig(R=1000.0, C=1e-6, num_freqs=200, f_min=0.1, f_max=1e5)
        self.spectrum = compute_spectrum(self.rc_cfg)

    def test_cauchy_maclaurin_real_from_imag(self) -> None:
        """Vérifie la reconstruction de chi' (X) à partir de chi'' (Y) (Cauchy-Maclaurin)."""
        kk_cfg = KKConfig(integration_method="cauchy_maclaurin", extrapolation="asymptotic")
        chi_real_rec = reconstruct_real_from_imag(self.spectrum.omega, self.spectrum.Y, kk_cfg=kk_cfg)

        # Évaluation sur la bande centrale (pour isoler des effets de bord extrêmes)
        mid_idx = slice(20, -20)
        chi_orig_mid = self.spectrum.X[mid_idx]
        chi_rec_mid = chi_real_rec[mid_idx]

        rms_err = float(np.sqrt(np.mean((chi_orig_mid - chi_rec_mid) ** 2)))
        self.assertLess(rms_err, 0.03)  # Erreur RMS < 3 %

    def test_reconstruct_imag_from_real(self) -> None:
        """Vérifie la reconstruction de chi'' (Y) à partir de chi' (X) (Cauchy-Maclaurin)."""
        kk_cfg = KKConfig(integration_method="cauchy_maclaurin", extrapolation="asymptotic")
        chi_imag_rec = reconstruct_imag_from_real(self.spectrum.omega, self.spectrum.X, kk_cfg=kk_cfg)

        mid_idx = slice(20, -20)
        chi_orig_mid = self.spectrum.Y[mid_idx]
        chi_rec_mid = chi_imag_rec[mid_idx]

        rms_err = float(np.sqrt(np.mean((chi_orig_mid - chi_rec_mid) ** 2)))
        self.assertLess(rms_err, 0.03)  # Erreur RMS < 3 %


    def test_trapezoidal_pv(self) -> None:
        """Vérifie l'intégration VPC par exclusion de la fenêtre de singularité."""
        kk_cfg = KKConfig(integration_method="trapezoidal_pv", singularity_eps=1e-3)
        chi_real_rec = reconstruct_real_from_imag(self.spectrum.omega, self.spectrum.Y, kk_cfg=kk_cfg)
        self.assertEqual(len(chi_real_rec), len(self.spectrum.omega))
        self.assertTrue(np.all(np.isfinite(chi_real_rec)))

    def test_asymptotic_extrapolation(self) -> None:
        """Vérifie que les modes d'extrapolation génèrent des spectres valides et finis."""
        cfg_no_extrap = KKConfig(extrapolation="none")
        cfg_with_extrap = KKConfig(extrapolation="asymptotic")

        rec_no = reconstruct_real_from_imag(self.spectrum.omega, self.spectrum.Y, kk_cfg=cfg_no_extrap)
        rec_with = reconstruct_real_from_imag(self.spectrum.omega, self.spectrum.Y, kk_cfg=cfg_with_extrap)

        self.assertEqual(len(rec_no), len(self.spectrum.omega))
        self.assertEqual(len(rec_with), len(self.spectrum.omega))
        self.assertTrue(np.all(np.isfinite(rec_no)))
        self.assertTrue(np.all(np.isfinite(rec_with)))

    def test_kk_analyzer_class(self) -> None:
        """Vérifie le fonctionnement global de la classe KramersKronigAnalyzer et KKResult."""
        analyzer = KramersKronigAnalyzer()
        res = analyzer.analyze(self.spectrum, target="real")

        self.assertEqual(res.target_component, "real")
        self.assertEqual(len(res.chi_reconstructed), len(self.spectrum.omega))
        self.assertEqual(len(res.residuals), len(self.spectrum.omega))
        self.assertGreater(res.r2_score, 0.95)  # Coefficient R² > 0.95
        self.assertLess(res.rms_error, 0.05)    # Erreur RMS global < 5 %

if __name__ == "__main__":
    unittest.main()