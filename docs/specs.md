# Spécifications Techniques

## 1. Présentation et Objectifs du Projet

Ce projet a pour objectif de concevoir une suite Python modulaire et réutilisable permettant de :
1. **Simuler la réponse physique et temporelle d'un circuit RC passe-bas** soumis à une excitation sinusoïdale ou quelconque avec injection de bruit (blanc et $1/f$).
2. **Implémenter un amplificateur synchrone numérique (Lock-in Amplifier)** basé sur la détection sensible à la phase (PSD) et un filtrage passe-bas (LPF) pour extraire l'amplitude et la phase (ou composantes $X$ et $Y$) du signal.
3. **Vérifier expérimentalement et théoriquement les relations de Kramers-Kronig (KK)** par intégration en Valeur Principale de Cauchy à partir du spectre mesuré (simulé ou issu de données réelles de laboratoire).

## 2. Architecture
```text
simu_lockin_kk/
├── .gitignore                  # Exclusion Git (__pycache__, env venv, etc.)
├── README.md                   # Documentation d'accueil et guide de démarrage rapide
├── pyproject.toml              # Configuration centrale (packaging, ruff, pytest, mypy)
├── requirements.txt            # Dépendances de production et de test
├── docs/                       # Documentation technique (Sphinx / MkDocs)
│   └── index.md
├── notebooks/                  # Prototypage et démonstration interactive
│   └── 01_demo_lockin_kk.ipynb
├── tests/                      # Suite de tests automatisés (pytest)
│   ├── __init__.py
│   ├── test_rc_circuit.py
│   ├── test_lockin.py
│   └── test_kramers_kronig.py
└── src/
    └── simu_lockin_kk/         # Package principal
        ├── __init__.py
        ├── config.py           # Dataclasses de configuration globale
        ├── main.py             # Point d'entrée de ligne de commande (CLI)
        ├── physics/            # Modélisation physique du composant
        │   ├── __init__.py
        │   └── rc_circuit.py   # Circuit RC : transfert complexe H(ω), réponse temporelle
        ├── signal/             # Traitement du signal et Lock-in
        │   ├── __init__.py
        │   ├── lockin.py       # PSD et filtrage passe-bas (LPF)
        │   └── noise.py        # Générateurs de bruit blanc et 1/f
        ├── analysis/           # Analyse théorique avancée
        │   ├── __init__.py
        │   └── kramers_kronig.py # Intégration en Valeur Principale de Cauchy (VPC)
        ├── utils/              # Entrées/Sorties et prétraitement de données
        │   ├── __init__.py
        │   └── io.py           # Chargement de CSV réels, ré-échantillonnage, extrapolation
        └── visualization/      # Modules graphiques
            ├── __init__.py
            └── plots.py        # Bode, spectres réels/imaginaires et résidus KK
```
## 3. Contrats d'Interface et Flux de Données

Le flux de données entre les différents maillons de la chaîne est strictement encapsulé dans des structures typées (`dataclass` ou `numpy.ndarray`).

```text
  [ Générateur / Circuit RC ]  ---> (v_out(t), t, f_ref)
               |
               v
    [ Amplificateur Lock-in ]  ---> (X(ω), Y(ω), ω)  <--- [ Données Expérimentales Réelles (io.py) ]
               |                                                   |
               +-----------------------+---------------------------+
                                       |
                                       v
                     [ Module Kramers-Kronig (VPC) ]  ---> (χ'_KK(ω), Erreur RMS)
```

### 3.1. Maillon 1 : Circuit RC (Physique & Signal Excité)

- **Responsabilité** : Simuler la réponse complexe $H(\omega)$ et générer le signal temporel bruité d'entrée du Lock-in.
- **Données d'Entrée** :
  - `R` (`float`) : Résistance ($\Omega$).
  - `C` (`float`) : Capacité ($F$).
  - `t` (`np.ndarray`) : Vecteur des temps d'échantillonnage ($s$).
  - `v_in` (`np.ndarray`) : Signal d'excitation temporel ($V$).
  - `noise_params` (`NoiseConfig`) : Dataclass définissant l'amplitude $\sigma$ et le type de bruit (blanc ou $1/f$).
- **Données de Sortie** :
  - `t` (`np.ndarray`) : Vecteur temps ($s$).
  - `v_in` (`np.ndarray`) : Signal d'entrée sans bruit ($V$).
  - `v_out_noisy` (`np.ndarray`) : Signal temporel de sortie bruité ($V$).
  - `H_theoretical` (`np.ndarray`) : Réponse fréquentielle théorique $H(\omega) = \frac{1}{1 + j\omega RC}$.

### 3.2. Maillon 2 : Amplificateur Synchrone (Lock-in)

- **Responsabilité** : Extraire les composantes en phase $X(\omega)$ et en quadrature $Y(\omega)$ du signal temporel pour chaque fréquence.
- **Données d'Entrée** :
  - `v_measured` (`np.ndarray`) : Signal temporel à mesurer ($V$), simulé ou réel.
  - `t` (`np.ndarray`) : Vecteur temporel associé ($s$).
  - `f_ref` (`float`) : Fréquence de référence de la détection ($Hz$).
  - `lockin_cfg` (`LockInConfig`) : Dataclass définissant la fréquence de coupure du filtre passe-bas $f_c$, l'ordre du filtre Butterworth, et la constante de temps $\tau$.
- **Données de Sortie** :
  - `omega` (`np.ndarray`) : Vecteur des pulsations de mesure ($\text{rad/s}$).
  - `X` (`np.ndarray`) : Composante en phase $\text{Re}\{H(\omega)\}$.
  - `Y` (`np.ndarray`) : Composante en quadrature $\text{Im}\{H(\omega)\}$.
  - `R` (`np.ndarray`) : Amplitude $R = \sqrt{X^2 + Y^2}$.
  - `phase` (`np.ndarray`) : Phase $\theta = \operatorname{atan2}(Y, X)$ ($\text{rad}$).

### 3.3. Maillon 3 : Vérification de Kramers-Kronig (VPC)

- **Responsabilité** : Reconstruire la partie réelle $\chi'(\omega)$ à partir de la partie imaginaire $\chi''(\omega)$ (ou réciproquement) via l'intégration en Valeur Principale de Cauchy.
- **Données d'Entrée** :
  - `omega` (`np.ndarray`) : Vecteur des pulsations échantillonnées, trié de manière strictly croissante ($\text{rad/s}$).
  - `chi_target` (`np.ndarray`) : Partie mesurée ($\chi''(\omega)$ ou $\chi'(\omega)$).
  - `kk_cfg` (`KKConfig`) : Paramètres d'intégration (méthode de traitement de la singularité en $\omega = \omega_0$, type d'extrapolation asymptotique).
- **Données de Sortie** :
  - `chi_reconstructed` (`np.ndarray`) : Partie réelle (ou imaginaire) reconstruite $\chi'_{KK}(\omega)$.
  - `residuals` (`np.ndarray`) : Écart point par point $\Delta(\omega) = \chi'_{mes}(\omega) - \chi'_{KK}(\omega)$.
  - `metrics` (`dict[str, float]`) : Indicateurs de qualité (Erreur RMS relative, $R^2$).

### 3.4. Maillon 4 : Entrées/Sorties et Données Expérimentales Réelles (`utils/io.py`)

- **Responsabilité** : Charger des fichiers de mesures réelles (CSV, DAT) et adapter les spectres aux exigences du module Kramers-Kronig.
- **Données d'Entrée** :
  - `file_path` (`str` | `Path`) : Chemin du fichier expérimental.
  - `column_mapping` (`dict`) : Association des colonnes (ex: `{"freq": 0, "real": 1, "imag": 2}`).
- **Données de Sortie** :
  - `omega_clean` (`np.ndarray`) : Grille de pulsations ré-échantillonnée et triée.
  - `chi_real_clean` (`np.ndarray`) : Partie réelle mesurée et nettoyée.
  - `chi_imag_clean` (`np.ndarray`) : Partie imaginaire mesurée et nettoyée.

## 4. Spécifications Physiques et Algorithmiques

### 4.1. Modélisation du Circuit RC

La fonction de transfert d'un circuit RC passe-bas s'écrit sous forme complexe :
$$H(\omega) = \frac{1}{1 + j\omega RC} = \frac{1}{1 + (\omega RC)^2} - j \frac{\omega RC}{1 + (\omega RC)^2}$$

La partie réelle $\chi'(\omega)$ et la partie imaginaire $\chi''(\omega)$ sont associées par :
$$\chi'(\omega) = \frac{1}{1 + (\omega RC)^2}, \quad \chi''(\omega) = -\frac{\omega RC}{1 + (\omega RC)^2}$$

### 4.2. Algorithme de Lock-in (PSD + LPF)

1. **Multiplication par la référence** :
   $$s_X(t) = v_{out}(t) \cdot \sqrt{2} \sin(\omega_0 t)$$
   $$s_Y(t) = v_{out}(t) \cdot \sqrt{2} \cos(\omega_0 t)$$
2. **Filtrage passe-bas** : Application d'un filtre numérique IIR (Butterworth d'ordre $N$) avec une fréquence de coupure $f_c \ll f_0$ pour isoler la composante continue :
   $$X = \text{LPF}(s_X(t)), \quad Y = \text{LPF}(s_Y(t))$$

### 4.3. Intégration par Valeur Principale de Cauchy (VPC)

Les relations de Kramers-Kronig s'expriment sous la forme d'intégrales singulières :
$$\chi'(\omega_0) = \frac{2}{\pi} \, P.V. \int_{0}^{\infty} \frac{\omega \, \chi''(\omega)}{\omega^2 - \omega_0^2} \, d\omega$$

Pour gérer la singularité en $\omega = \omega_0$, l'algorithme met en œuvre la méthode de soustraction de la singularité (technique de Maclaurin) :
$$P.V. \int_{0}^{\infty} \frac{\omega \, \chi''(\omega)}{\omega^2 - \omega_0^2} \, d\omega = \int_{0}^{\infty} \frac{\omega \chi''(\omega) - \omega_0 \chi''(\omega_0)}{\omega^2 - \omega_0^2} \, d\omega + \chi''(\omega_0) \ln\left|\frac{\omega_{max} - \omega_0}{\omega_{max} + \omega_0}\right|$$

Pour les données expérimentales à domaine borné $[\omega_{min}, \omega_{max}]$, le module applique une extrapolation asymptotique en loi de puissance aux limites ($\omega \to 0$ et $\omega \to \infty$) afin de minimiser les erreurs de troncature bord.

## 5. Standards de Développement et Qualité Code

1. **PEP 8 & Formatting** : Formatage automatique strict via `ruff format` ou `black`.
2. **Static Typing** : Utilisation de `mypy --strict` pour valider le typage explicite de chaque fonction.
3. **Docstrings NumPy Format** : Chaque fonction/classe doit être documentée avec ses sections `Parameters`, `Returns`, et `Raises`.
4. **Validation Pytest** :
   - `test_rc_circuit.py` : Vérification du point $-3\text{ dB}$ ($\omega = 1/RC$).
   - `test_lockin.py` : Vérification du rejet de bruit ($SNR < 0\text{ dB}$).
   - `test_kramers_kronig.py` : Vérification de la reconstruction KK théorique avec une erreur RMS $< 1\%$.
