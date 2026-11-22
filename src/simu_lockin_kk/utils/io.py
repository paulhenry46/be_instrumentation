from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

# Définition d'un segment de grille : (f_min, f_max, num_points, spacing) 


def _generate_single_segment(segment: tuple[float, float, int, str]) -> NDArray[np.float64]:
  """Génère un sous-tableau de fréquences pour un segment donné.

  Parameters
  ----------
  segment : Tuple (f_min, f_max, num_points, spacing).

  Returns
  -------
  NDArray[np.float64]
      Tableau NumPy 1D des fréquences du segment en Hz.

  Raises
  ------
  ValueError
      Si la structure du segment, les bornes, le nombre de points ou le type
      d'espacement sont invalides.
  """

  if len(segment) == 4:
    f_min, f_max, num_points, spacing = segment
  else:
    raise ValueError(
        "Chaque segment doit contenir 3 ou 4 éléments : "
        "(f_min, f_max, num_points) ou (f_min, f_max, num_points, spacing)."
    )

  # Vérifie que les bornes sont des nombres finis avec f_max > f_min
  if not (np.isfinite(f_min) and np.isfinite(f_max)):
    raise ValueError("f_min et f_max doivent être des nombres finis.")
  if f_max <= f_min:
    raise ValueError("f_max doit être strictement supérieur à f_min.")

  # Vérifie que le nombre de points est valide
  if num_points < 2:
    raise ValueError("num_points doit être supérieur ou égal à 2.")

  # Vérifie le mode d'espacement
  mode = spacing.strip().lower()
  if mode not in ("log", "linear"):
    raise ValueError("spacing doit valoir 'log' ou 'linear'.")

  # En échelle logarithmique, les fréquences doivent être strictement positives
  if mode == "log" and (f_min <= 0.0 or f_max <= 0.0):
    raise ValueError(
        "f_min et f_max doivent être strictement positifs pour spacing='log'."
    )

  # Génère le sous-tableau correspondant à la précision/résolution demandée
  if mode == "log":
    return np.geomspace(float(f_min), float(f_max), num=num_points)
  return np.linspace(float(f_min), float(f_max), num=num_points)


def build_multi_precision_grid(
    segments: Sequence[tuple[float, float, int, str]],
    deduplicate: bool = True,
    dedup_tol: float = 1e-12,
) -> NDArray[np.float64]:
  """Génère plusieurs sous-tableaux de fréquences et les concacetène.

  Parameters
  ----------
  segments : Sequence[tuple[float, float, int, str]]
      Séquence de segments décrivant chaque zone de fréquence et sa précision
      (nombre de points et type d'espacement).
  sort_ascending : bool, optional
      Si True, trie le tableau concaténé par ordre croissant
      (True par défaut).
  deduplicate : bool, optional
      Si True, supprime les fréquences quasi identiques à la jonction des
      segments (True par défaut).
  dedup_tol : float, optional
      Tolérance relative pour fusionner deux fréquences très proches lors de la
      dédoublonnisation (1e-12 par défaut).

  Returns
  -------
  NDArray[np.float64]
      Tableau NumPy 1D unique contenant toutes les fréquences en Hz.

  Raises
  ------
  ValueError
      Si segments est vide ou si dedup_tol est négatif ou non fini.
  """
  # Vérifie qu'au moins un segment est fourni
  if len(segments) == 0:
    raise ValueError("segments ne doit pas être vide.")

  # Vérifie la tolérance de dédoublonnage
  if not np.isfinite(dedup_tol) or dedup_tol < 0.0:
    raise ValueError("dedup_tol doit être >=0.")

  # Génère chaque sous-tableau individuellement puis concatène l'ensemble
  sub_grids = [_generate_single_segment(seg) for seg in segments]
  combined = np.concatenate(sub_grids).astype(np.float64, copy=False)
  combined = np.sort(combined)

  # Supprime les points quasi identiques aux frontières entre deux segments
  if deduplicate and combined.size > 1:
    keep = np.ones(combined.size, dtype=bool)
    rel_diff = np.abs(np.diff(combined)) / np.maximum(
        np.abs(combined[1:]), 1e-300
    )
    keep[1:] = rel_diff > dedup_tol
    combined = combined[keep]

  return combined