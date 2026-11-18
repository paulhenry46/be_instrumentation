"""Tests unitaires pour le package signal (bruit et lock-in)."""

import unittest
import numpy as np

from simu_lockin_kk.config import LockInConfig, NoiseConfig, RCCircuitConfig
from simu_lockin_kk.physics.rc_circuit import compute_transfer_function
from simu_lockin_kk.signal.lockin import (
    LockInAmplifier,
    demodulate_signal,
    demodulate_sweep,
)
from simu_lockin_kk.signal.noise import (
    apply_noise,
    generate_flicker_noise,
    generate_white_noise,
)

class TestNoiseModule(unittest.TestCase):
    """Tests pour les générateurs de bruit."""

    def test_white_noise_statistics(self) -> None:
        n_samples = 100000
        std_target = 0.1
        noise = generate_white_noise(n_samples=n_samples, std=std_target, seed=42)

        self.assertEqual(len(noise), n_samples)
        self.assertAlmostEqual(float(np.mean(noise)), 0.0, places=2)
        self.assertAlmostEqual(float(np.std(noise)), std_target, places=2)

    def test_flicker_noise_statistics(self) -> None:
        n_samples = 50000
        fs = 1e4
        std_target = 0.2
        flicker = generate_flicker_noise(n_samples=n_samples, fs=fs, std=std_target, seed=123)

        self.assertEqual(len(flicker), n_samples)
        self.assertAlmostEqual(float(np.std(flicker)), std_target, places=2)

    def test_apply_noise_types(self) -> None:
        signal = np.ones(1000, dtype=np.float64)
        fs = 1e4

        # None
        cfg_none = NoiseConfig(noise_type="none")
        res_none = apply_noise(signal, fs=fs, noise_cfg=cfg_none)
        np.testing.assert_array_equal(res_none, signal)

        # White
        cfg_white = NoiseConfig(noise_type="white", std_white=0.05, seed=1)
        res_white = apply_noise(signal, fs=fs, noise_cfg=cfg_white)
        self.assertNotAlmostEqual(float(np.std(res_white)), 0.0)

class TestLockInAmplifier(unittest.TestCase):
    """Tests pour la démodulation synchrone (Lock-in)."""

    def setUp(self) -> None:
        self.f_ref = 1000.0
        self.fs = 100000.0
        self.duration = 0.1
        self.t = np.linspace(0.0, self.duration, int(self.duration * self.fs), endpoint=False)

    def test_demodulate_pure_sine(self) -> None:
        """Vérifie la récupération exacte d'une sinusoïde pure avec phase décalée."""
        amplitude = 2.5
        phase_in = np.pi / 6.0

        w0 = 2.0 * np.pi * self.f_ref
        v_in = amplitude * np.sin(w0 * self.t + phase_in)

        lockin_cfg = LockInConfig(f_ref=self.f_ref, lpf_cutoff=20.0, lpf_order=2)
        X, Y, R, phase_out = demodulate_signal(self.t, v_in, f_ref=self.f_ref, lockin_cfg=lockin_cfg)

        expected_X = amplitude * np.cos(phase_in)
        expected_Y = amplitude * np.sin(phase_in)

        self.assertAlmostEqual(X, expected_X, places=3)
        self.assertAlmostEqual(Y, expected_Y, places=3)
        self.assertAlmostEqual(R, amplitude, places=3)
        self.assertAlmostEqual(phase_out, phase_in, places=3)

    def test_lockin_noise_rejection(self) -> None:
        """Vérifie l'extraction du signal au milieu d'un bruit fort."""
        amplitude = 0.5
        w0 = 2.0 * np.pi * self.f_ref
        v_clean = amplitude * np.sin(w0 * self.t)

        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, 2.0, size=len(self.t))
        v_noisy = v_clean + noise

        lockin_cfg = LockInConfig(f_ref=self.f_ref, lpf_cutoff=5.0, lpf_order=4)
        X, Y, R, phase = demodulate_signal(self.t, v_noisy, f_ref=self.f_ref, lockin_cfg=lockin_cfg)

        self.assertAlmostEqual(R, amplitude, delta=0.05)

    def test_demodulate_sweep_rc(self) -> None:
        """Test de balayage sur un circuit RC."""
        rc_cfg = RCCircuitConfig(R=1000.0, C=1e-6, num_freqs=5, f_min=10.0, f_max=1000.0)
        freqs = np.array([10.0, 50.0, 159.15, 500.0, 1000.0])

        spec = demodulate_sweep(rc_config=rc_cfg, freqs=freqs, duration=0.1, fs=50000.0)
        self.assertEqual(len(spec.freqs), 5)

if __name__ == "__main__":
    unittest.main()