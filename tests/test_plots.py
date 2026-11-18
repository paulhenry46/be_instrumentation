"""Tests unitaires pour le module de visualisation graphique."""

import unittest
from pathlib import Path
import matplotlib.pyplot as plt

from simu_lockin_kk.config import RCCircuitConfig
from simu_lockin_kk.analysis.kramers_kronig import analyze_kramers_kronig
from simu_lockin_kk.physics.rc_circuit import compute_spectrum, generate_time_response
from simu_lockin_kk.visualization.plots import (
    plot_bode,
    plot_kramers_kronig_comparison,
    plot_time_domain,
)

class TestPlotsModule(unittest.TestCase):
    """Suite de tests unitaires pour le module de génération des figures."""

    def setUp(self) -> None:
        self.rc_cfg = RCCircuitConfig(R=1000.0, C=1e-6, num_freqs=50)
        self.spectrum = compute_spectrum(self.rc_cfg)
        self.time_data = generate_time_response(self.rc_cfg, f_ref=100.0, duration=0.02, fs=50000.0)
        self.kk_result = analyze_kramers_kronig(self.spectrum, target="real")
        self.tmp_dir = Path("./")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        plt.close("all")

    def test_plot_time_domain(self) -> None:
        fig = plot_time_domain(self.time_data, num_periods=3.0)
        self.assertIsNotNone(fig)
        out_file = self.tmp_dir / "time_domain.png"
        fig.savefig(out_file)
        self.assertTrue(out_file.exists())
        self.assertGreater(out_file.stat().st_size, 0)

    def test_plot_bode(self) -> None:
        fig = plot_bode(self.spectrum, use_omega=False)
        self.assertIsNotNone(fig)
        out_file = self.tmp_dir / "bode.png"
        fig.savefig(out_file)
        self.assertTrue(out_file.exists())
        self.assertGreater(out_file.stat().st_size, 0)

    def test_plot_kramers_kronig_comparison(self) -> None:
        fig = plot_kramers_kronig_comparison(self.kk_result, use_omega=True)
        self.assertIsNotNone(fig)
        out_file = self.tmp_dir / "kk_comp.png"
        fig.savefig(out_file)
        self.assertTrue(out_file.exists())
        self.assertGreater(out_file.stat().st_size, 0)

if __name__ == "__main__":
    unittest.main()