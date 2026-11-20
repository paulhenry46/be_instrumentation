# Présentation du module de génération de bruit
Cette documentation correspond au module `simu_lockin_kk.signal.noise`.
## Vue d'ensemble
### Fonction `generate_white_noise`

Modélise un bruit blanc gaussien centré (moyenne nulle) d'écart-type `std` sur `n_samples` points : chaque échantillon est tiré indépendamment selon $\mathcal{N}(0, \sigma^2)$, ce qui donne une densité spectrale de puissance uniforme (toutes les fréquences sont représentées à puissance égale).

- Validation stricte (`n_samples > 0`, `std >= 0`) : évite les tableaux vides ou négatifs et les écarts-types impossibles (`std == 0` reste valide et produit un vecteur nul).
- Générateur `np.random.default_rng(seed)` : on utilise l'API moderne de NumPy plutôt que l'état global `np.random`, et permet la reproductibilité via `seed` pour le débogage.

### Fonction `apply_noise`

Fait le lien entre un signal temporel propre (`signal`), obtenu avec le module précédent et une configuration déclarative (`NoiseConfig`).

- Copie de précaution : on protège le signal d'origine contre toute modification sur place, que `noise_type` soit `"none"` ou actif.
- on centralise la sélection du mode de bruit et prépare l'ajout d'autres composantes sans changer l'appelant (`generate_time_response`).

## `generate_white_noise` vs `apply_noise`

* `generate_white_noise` = brique élémentaire de génération :
Crée un vecteur brut de bruit blanc gaussien à partir de sa taille (`n_samples`), de son amplitude (`std`) et d'une graine (`seed`), sans connaître le signal ni le contexte de simulation. Utile pour le débogage et les tests.
* `apply_noise` = couche d'application pilotée par configuration :
Lit le `noise_cfg`, génère les composantes demandées à la bonne longueur `len(signal)` et les additionne au signal propre.

### CheatSheet

| Critère | `generate_white_noise` | `apply_noise` |
| --- | --- | --- |
| Entrée principale | `n_samples`, `std`, `seed` | `signal`, `fs`, `noise_cfg` |
| Connaissance du signal | Non (génère un vecteur seul) | Oui (adapte la longueur à `len(signal)`) |
| Sortie | `np.ndarray` de bruit seul (`(n_samples,)`) | `np.ndarray` : signal bruité (`signal + bruit`) |
| Gestion de `NoiseConfig` | Non | Oui (`"none"`, `"white"`, `"combined"`) |