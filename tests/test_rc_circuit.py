"""Tests unitaires pour le module physics.rc_circuit."""

import unittest
import numpy as np

from simu_lockin_kk.config import RCCircuitConfig
from simu_lockin_kk.physics.rc_circuit import (
    RCCircuit,
    compute_spectrum,
    compute_transfer_function,
    generate_time_response,
)

class TestRCCircuitConfig(unittest.TestCase):
    """Tests de validation de la classe de configuration RCCircuitConfig."""

    def test_valid_config(self) -> None:
        cfg = RCCircuitConfig(R=1000.0, C=1e-6)
        self.assertEqual(cfg.R, 1000.0)
        self.assertEqual(cfg.C, 1e-6)
        self.assertAlmostEqual(cfg.tau, 1e-3, places=7)
        self.assertAlmostEqual(cfg.fc, 1000.0 / (2.0 * np.pi), places=4)
        self.assertAlmostEqual(cfg.wc, 1000.0, places=4)

    def test_invalid_parameters(self) -> None:
        """Vérifie le déclenchement d'erreurs en cas de valeurs d'entrée physiques incohérentes."""
        with self.assertRaises(ValueError):
            RCCircuitConfig(R=-10.0, C=1e-6)
        with self.assertRaises(ValueError):
            RCCircuitConfig(R=1000.0, C=0.0)
        with self.assertRaises(ValueError):
            RCCircuitConfig(R=1000.0, C=1e-6, f_min=1000.0, f_max=100.0)

class TestComputeTransferFunction(unittest.TestCase):
    """Tests mathématiques et physiques pour la fonction de transfert H(w)."""

    def setUp(self) -> None:
        self.R = 1000.0
        self.C = 1e-6
        self.tau = self.R * self.C  # 1 ms
        self.wc = 1.0 / self.tau    # 1000 rad/s

    def test_dc_limit(self) -> None:
        """À basse fréquence (w -> 0), H(0) = 1 + 0j (pas d'atténuation, pas de déphasage)."""
        H_dc = compute_transfer_function(0.0, self.R, self.C)
        self.assertAlmostEqual(np.real(H_dc), 1.0, places=6)
        self.assertAlmostEqual(np.imag(H_dc), 0.0, places=6)

    def test_cutoff_frequency(self) -> None:
        """À la pulsation de coupure (w = wc), |H| = 1/sqrt(2) ≈ 0.7071 et phase = -pi/4 (-45°)."""
        H_c = compute_transfer_function(self.wc, self.R, self.C)
        gain = np.abs(H_c)
        phase = np.angle(H_c)

        self.assertAlmostEqual(gain, 1.0 / np.sqrt(2.0), places=6)
        self.assertAlmostEqual(phase, -np.pi / 4.0, places=6)
        self.assertAlmostEqual(np.real(H_c), 0.5, places=6)
        self.assertAlmostEqual(np.imag(H_c), -0.5, places=6)

    def test_high_frequency_limit(self) -> None:
        """À très haute fréquence (w >> wc), |H| tend vers 0."""
        w_high = 1e6 * self.wc
        H_high = compute_transfer_function(w_high, self.R, self.C)
        self.assertLess(np.abs(H_high), 1e-5)


class TestComputeSpectrum(unittest.TestCase):
    """Tests pour la génération du spectre complet SpectrumData."""

    def setUp(self) -> None:
        self.cfg = RCCircuitConfig(R=1000.0, C=1e-6, num_freqs=50)

    def test_spectrum_dimensions(self) -> None:
        spec = compute_spectrum(self.cfg)
        self.assertEqual(len(spec.freqs), 50)
        self.assertEqual(len(spec.omega), 50)
        self.assertEqual(len(spec.X), 50)
        self.assertEqual(len(spec.Y), 50)
        self.assertEqual(len(spec.R), 50)
        self.assertEqual(len(spec.phase), 50)

    def test_custom_freqs(self) -> None:
        custom_freqs = np.array([10.0, 100.0, 1000.0])
        spec = compute_spectrum(self.cfg, freqs=custom_freqs)
        self.assertEqual(len(spec.freqs), 3)
        np.testing.assert_array_equal(spec.freqs, custom_freqs)

class TestGenerateTimeResponse(unittest.TestCase):
    """Tests de simulation temporelle."""

    def setUp(self) -> None:
        self.cfg = RCCircuitConfig(R=1000.0, C=1e-6, amplitude=2.0)
        self.f_ref = 100.0  # 100 Hz

    def test_time_vector_length(self) -> None:
        duration = 0.05
        fs = 50000.0
        data = generate_time_response(self.cfg, f_ref=self.f_ref, duration=duration, fs=fs)
        expected_samples = int(duration * fs)
        self.assertEqual(len(data.t), expected_samples)
        self.assertEqual(len(data.v_in), expected_samples)
        self.assertEqual(len(data.v_out_clean), expected_samples)

    def test_shannon_validation(self) -> None:
        """Vérifie le rejet d'une fréquence d'échantillonnage sous le seuil de Shannon."""
        with self.assertRaises(ValueError):
            generate_time_response(self.cfg, f_ref=1000.0, fs=1500.0)

    def test_output_amplitude_and_phase(self) -> None:
        """Vérifie l'amplitude maximale de la réponse en régime stabilisé."""
        data = generate_time_response(self.cfg, f_ref=self.f_ref, duration=0.1, fs=100000.0)
        omega_ref = 2.0 * np.pi * self.f_ref
        H_ref = compute_transfer_function(omega_ref, self.cfg.R, self.cfg.C)
        expected_max = self.cfg.amplitude * np.abs(H_ref)

        self.assertAlmostEqual(np.max(data.v_out_clean), expected_max, places=4)

class TestRCCircuitClass(unittest.TestCase):
    """Tests de la classe orientée objet RCCircuit."""

    def test_wrapper_methods(self) -> None:
        cfg = RCCircuitConfig(R=2000.0, C=500e-9)
        circuit = RCCircuit(cfg)

        self.assertEqual(circuit.R, 2000.0)
        self.assertEqual(circuit.C, 500e-9)
        self.assertAlmostEqual(circuit.tau, 1e-3, places=7)

        H_val = circuit.H(1000.0)
        self.assertIsInstance(H_val, (complex, np.ndarray))

        spec = circuit.get_spectrum()
        self.assertEqual(len(spec.freqs), 100)

if __name__ == "__main__":
    unittest.main()