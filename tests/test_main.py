"""Tests d'intégration pour la simulation globale main.py."""

import unittest
from pathlib import Path
from simu_lockin_kk.main import run_pipeline

class TestMainPipeline(unittest.TestCase):
    def test_run_pipeline_execution(self) -> None:
        out_dir = Path("./")
        run_pipeline(
            R=1000.0,
            C=1e-6,
            std_white=0.02,
            num_freqs=50,
            output_dir=out_dir,
            save_plots=True,
        )

        self.assertTrue((out_dir / "01_time_domain.png").exists())
        self.assertTrue((out_dir / "02_bode_diagram.png").exists())
        self.assertTrue((out_dir / "03_kramers_kronig_verification.png").exists())

if __name__ == "__main__":
    unittest.main()