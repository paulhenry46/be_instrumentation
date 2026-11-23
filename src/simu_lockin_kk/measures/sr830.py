from __future__ import annotations

from time import perf_counter, sleep

import numpy as np
from numpy.typing import NDArray
from pymeasure.adapters import SerialAdapter
from pymeasure.instruments.srs import SR830

from simu_lockin_kk.config import SpectrumData

# Table des constantes de temps du SR830. Utilisé pour le filtre pass-bas de la fin, compromis à trouver, etc.
SR830_TIME_CONSTANTS: tuple[float, ...] = (
    10e-6,
    30e-6,
    100e-6,
    300e-6,
    1e-3,
    3e-3,
    10e-3,
    30e-3,
    100e-3,
    300e-3,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1e3,
    3e3,
    10e3,
    30e3,
)


def nearest_time_constant_index(tau: float) -> tuple[int, float]:
  """Trouve l'indice et la valeur de constante de temps SR830 les plus proches de tau.

  Parameters
  ----------
  tau : float
      Constante de temps cible en secondes.

  Returns
  -------
  tuple[int, float]
      Indice pour la commande OFLT et constante de temps effective
      correspondante en secondes.

  Raises
  ------
  ValueError
      Si tau n'est pas strictement positif ou n'est pas fini.
  """
  # Vérifie que la constante de temps demandée est un nombre fini strictement positif
  if not np.isfinite(tau) or tau <= 0.0:
    raise ValueError("tau doit être un nombre fini strictement positif en secondes.")

  # Convertit la table en array NumPy pour calculer l'écart absolu à tau
  tc_array = np.asarray(SR830_TIME_CONSTANTS, dtype=np.float64)

  # Récupère l'indice minimisant |tc - tau|
  index = int(np.argmin(np.abs(tc_array - tau)))

  # Retourne l'indice OFLT et la constante de temps effective correspondante
  return index, float(SR830_TIME_CONSTANTS[index])


def parse_csv_floats(raw_response: str) -> list[float]:
  """Convertit une chaîne ASCII de flottants séparés par des virgules en liste Python.

  Parameters
  ----------
  raw_response : str
      Chaîne brute renvoyée par l'instrument (par exemple "1.0,2.0,3.0").

  Returns
  -------
  list[float]
      Liste des valeurs flottantes extraites.
  """
  # Découpe la chaîne nettoyée par virgule et convertit chaque élément non vide en float, il faudra utiliser la lib csv une fois que l'on aura importé pour la partie io.py
  return [
      float(token.strip())
      for token in raw_response.strip().split(",")
      if token.strip()
  ]


def sweep_frequencies(
    lia: SR830,
    frequencies: NDArray[np.float64] | list[float],
    tau: float,
    amplitude_v: float = 1.0,
    wait_factor: float = 5.0,
    output_interface: int = 0,
) -> dict[str, NDArray[np.float64] | NDArray[np.int64]]:
  """Effectue un balayage en fréquence sur un SR830 et retourne des tableaux NumPy.

  Parameters
  ----------
  lia : SR830
      Instance connectée de l'instrument PyMeasure SR830.
  frequencies : NDArray[np.float64] | list[float]
      Séquence des fréquences d'excitation en Hz.
  tau : float
      Constante de temps souhaitée du filtre passe-bas en secondes.
  amplitude_v : float, optional
      Amplitude de la sortie sinusoïdale interne en Vrms (1.0 par défaut).
  wait_factor : float, optional
      Facteur multiplicateur appliqué à la constante de temps effective pour
      le temps d'attente de stabilisation (5.0 par défaut).
  output_interface : int, optional
      Interface cible pour les réponses via OUTX : 0 pour RS-232,
      1 pour GPIB (0 par défaut).

  Returns
  -------
  dict[str, NDArray[np.float64] | NDArray[np.int64]]
      Dictionnaire de tableaux NumPy 1D contenant les fréquences demandées et
      mesurées, les métadonnées de constante de temps, X, Y, R,
      theta, CH1, CH2, les entrées auxiliaires 1..4 et
      l'octet d'état LIA.

  Raises
  ------
  ValueError
      Si frequencies est vide, contient des valeurs négatives, nulles ou
      non finies, ou si wait_factor ou output_interface est invalide.
  """
  # Convertit l'entrée en tableau NumPy 1D de flottants double précision
  freq_array = np.asarray(frequencies, dtype=np.float64).ravel()# applatir sans copie en mémoire, on aurait pu utiliser flatten aussi mais ça copie en mémoire

  # Vérifie que le tableau de fréquences n'est pas vide
  if freq_array.size == 0:
    raise ValueError("Le tableau frequencies ne doit pas être vide.")

  # Vérifie que toutes les fréquences sont finies et >0
  if not np.all(np.isfinite(freq_array)) or np.any(freq_array <= 0.0):
    raise ValueError("Toutes les fréquences doivent être finies et strictement positives (en Hz).")

  # Vérifie que le multiplicateur d'attente est positif ou nul et fini
  if not np.isfinite(wait_factor) or wait_factor < 0.0:
    raise ValueError("wait_factor doit être un nombre fini positif ou nul.")

  # il faut que l'interface de sortie correspond bien à RS-232 (0) ou GPIB (1)
  if output_interface not in (0, 1):
    raise ValueError("output_interface doit valoir 0 (RS-232) ou 1 (GPIB).")

  lia.write("*RST")
  # Dirige les réponses de l'appareil vers l'interface active (RS-232 ou GPIB)
  lia.write(f"OUTX {output_interface}")

  # Sélectionne l'oscillateur de référence interne (FMOD 1)
  lia.write("FMOD 1")

  # Règle l'amplitude de sortie sinusoïdale interne en Vrms
  lia.write(f"SLVL {amplitude_v}")

  # Détermine l'indice OFLT et la constante de temps effective la plus proche de tau
  tc_index, tc_effective = nearest_time_constant_index(tau)

  # Applique la constante de temps sur le SR830
  lia.write(f"OFLT {tc_index}")

  # Calcule le temps d'attente de stabilisation après chaque saut en fréquence
  wait_time_s = wait_factor * tc_effective

  # Nombre total de points de mesure
  n_points = freq_array.size

  # on prealloue les tableaux pour éviter que tout foire en cours d'acquisition
  measured_freq_hz = np.empty(n_points, dtype=np.float64)
  actual_wait_s = np.empty(n_points, dtype=np.float64)
  x_volts = np.empty(n_points, dtype=np.float64)
  y_volts = np.empty(n_points, dtype=np.float64)
  r_volts = np.empty(n_points, dtype=np.float64)
  theta_deg = np.empty(n_points, dtype=np.float64)
  ch1_val = np.empty(n_points, dtype=np.float64)
  ch2_val = np.empty(n_points, dtype=np.float64)
  aux_in_1_v = np.empty(n_points, dtype=np.float64)
  aux_in_2_v = np.empty(n_points, dtype=np.float64)
  aux_in_3_v = np.empty(n_points, dtype=np.float64)
  aux_in_4_v = np.empty(n_points, dtype=np.float64)
  lia_status = np.empty(n_points, dtype=np.int64)
# conf usine    a remettre avec commande
# liste des vérifs à éffectuer
# on va plutot utiliser le port GBPI
  # Parcourt chaque fréquence cible du tableau
  for idx, freq_target in enumerate(freq_array):
    # Envoie la nouvelle fréquence de référence interne au SR830
    lia.write(f"FREQ {freq_target}")

    # Mesure le temps d'attente réel pour assurer la stabilisation du filtre
    t_start = perf_counter()
    sleep(wait_time_s)
    actual_wait_s[idx] = perf_counter() - t_start

    # Premier SNAP? (6 paramètres max) : X (1), Y (2), R (3), theta (4), Freq (9), CH1 (10)
    x_val, y_val, r_val, th_val, f_meas, c1_val = parse_csv_floats(
        lia.ask("SNAP? 1,2,3,4,9,10")
    )

    # Deuxième SNAP? : Aux In 1..4 (5, 6, 7, 8) et CH2 (11)
    a1_val, a2_val, a3_val, a4_val, c2_val = parse_csv_floats(
        lia.ask("SNAP? 5,6,7,8,11")
    )

    # Interroge l'octet de statut LIA pour voir d'éventuelles erreurs
    status_val = int(float(lia.ask("LIAS?").strip()))

    # Enregistre les valeurs mesurées à l'indice courant
    measured_freq_hz[idx] = f_meas
    x_volts[idx] = x_val
    y_volts[idx] = y_val
    r_volts[idx] = r_val
    theta_deg[idx] = th_val
    ch1_val[idx] = c1_val
    ch2_val[idx] = c2_val
    aux_in_1_v[idx] = a1_val
    aux_in_2_v[idx] = a2_val
    aux_in_3_v[idx] = a3_val
    aux_in_4_v[idx] = a4_val
    lia_status[idx] = status_val

  # Retourne l'ensemble des résultats en np
  return {
      "target_freq_hz": freq_array,
      "measured_freq_hz": measured_freq_hz,
      "requested_tau_s": np.full(n_points, float(tau), dtype=np.float64),
      "effective_tau_s": np.full(n_points, tc_effective, dtype=np.float64),
      "actual_wait_s": actual_wait_s,
      "x_v": x_volts,
      "y_v": y_volts,
      "r_v": r_volts,
      "theta_deg": theta_deg,
      "ch1": ch1_val,
      "ch2": ch2_val,
      "aux_in_1_v": aux_in_1_v,
      "aux_in_2_v": aux_in_2_v,
      "aux_in_3_v": aux_in_3_v,
      "aux_in_4_v": aux_in_4_v,
      "lia_status": lia_status,
  }

def test_sr830(
    lia: SR830,
    output_interface: int = 0,
) -> dict[str, NDArray[np.float64] | NDArray[np.int64]]:
  
  if output_interface not in (0, 1):
    raise ValueError("output_interface doit valoir 0 (RS-232) ou 1 (GPIB).")

  lia.write("*RST")
  # Dirige les réponses de l'appareil vers l'interface active (RS-232 ou GPIB)
  lia.write(f"OUTX {output_interface}")
  result = lia.ask("IDN?")
  return result


def sweep_to_spectrum_data(
    sweep_data: dict[str, NDArray[np.float64] | NDArray[np.int64]],
    *,
    amplitude_v: float = 1.0,
    normalize_by_amplitude: bool = True,
    use_measured_freq: bool = True,
) -> SpectrumData:
  """Convertit les tableaux issus du balayge en objet SpectrumData.

  Parameters
  ----------
  sweep_data : dict[str, NDArray[np.float64] | NDArray[np.int64]]
      Dictionnaire renvoyé par sweep_frequencies.
  amplitude_v : float, optional
      Amplitude d'excitation en Volts utilisée pour normaliser X, Y
      et R lorsque normalize_by_amplitude est True (1.0 par défaut).
  normalize_by_amplitude : bool, optional
      Si True, divise les composantes par amplitude_v pour obtenir une
      réponse adimensionnelle comparable à H(w) (True par défaut).
  use_measured_freq : bool, optional
      Si True, utilise measured_freq_hz ; sinon utilise target_freq_hz
      (True par défaut).

  Returns
  -------
  SpectrumData
      Objet spectral prêt pour l'analyse Kramers-Kronig.

  Raises
  ------
  ValueError
      Si amplitude_v est nul, négatif ou non fini, ou si les fréquences
      ne sont pas triées de manière strictement croissante.
  """
  # amplitude de normalisation positive finie
  if not np.isfinite(amplitude_v) or amplitude_v <= 0.0:
    raise ValueError(
        "amplitude_v doit être un nombre fini strictement positif."
    )

  freq_key = "measured_freq_hz" if use_measured_freq else "target_freq_hz"
  freqs = np.asarray(sweep_data[freq_key], dtype=np.float64)

  # Vérifie que les fréquences sont croissantes pour Kramers-Kronig!!!
  if np.any(np.diff(freqs) <= 0.0):
    raise ValueError(
        "Les fréquences doivent être triées de manière strictement croissante"
        " pour Kramers-Kronig."
    )

  # Calcule omega = 2pif
  omega = 2.0 * np.pi * freqs

  # Récupère X, Y, R (en Volts) et la phase (convertie des degrés vers les radians)
  x_raw = np.asarray(sweep_data["x_v"], dtype=np.float64)
  y_raw = np.asarray(sweep_data["y_v"], dtype=np.float64)
  r_raw = np.asarray(sweep_data["r_v"], dtype=np.float64)
  phase_rad = np.deg2rad(np.asarray(sweep_data["theta_deg"], dtype=np.float64))

  # Applique éventuellement la normalisation par l'amplitude mesurée (à voir en pratique)
  scale = amplitude_v if normalize_by_amplitude else 1.0
  x_comp = x_raw / scale
  y_comp = y_raw / scale
  r_mag = r_raw / scale

  # Reconstruit le spectre
  h_complex = (x_comp + 1j * y_comp).astype(np.complex128) # = as en TS

  return SpectrumData(
      freqs=freqs,
      omega=omega,
      X=x_comp,
      Y=y_comp,
      R=r_mag,
      phase=phase_rad,
      H_complex=h_complex,
  )

def connect_sr830_serial(
    port: str = "/dev/ttyUSB0",# ATTENTION, valeur par défaut pour ma machine, peut changer, il faut bien penser à vérif le bon port!
    baudrate: int = 9600,
    timeout: float = 2.0,
) -> SR830:
  """Ouvre une connexion série vers un instrument SR830 sous Linux.

  Parameters
  ----------
  port : str, optional
      Chemin du périphérique
  baudrate : int, optional
      Débit de la liaison série en bits par seconde (9600 par défaut).
  timeout : float, optional
      Délai maximal de lecture/écriture en secondes (2.0 par défaut).

  Returns
  -------
  SR830
      Contrôleur PyMeasure SR830 instancié.
  """
  # Crée l'adaptateur série PyMeasure avec les paramètres de port Linux
  adapter = SerialAdapter(port=port, baudrate=baudrate, timeout=timeout)

  return SR830(adapter)