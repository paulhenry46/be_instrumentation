# Présentation du module d'amplification synchrone (Lock-in)

## Vue d'ensemble

Ce module simule les composants les plus importants d'un lockin, i.e. la PSD + filtrage passe-bas  afin d'extraire  les composantes X et Y d'un signal à une fréquence de référence `f_ref`, même en présence de bruit. Meme si cela sort du scope de ce module, j'ai également la fonction qui permet aussi d'enchaîner un balayage en fréquence complet pour reconstruire un spectre mesuré.

### Fonction privée `_apply_lpf`

Isole la composante continue (DC) après multiplication par la référence, en éliminant les termes à $2 f_{\text{ref}}$.

- Plafonnement de `cutoff` à `0.1 * f_ref` : garantit une forte atténuation de la composante à $2 f_{\text{ref}}$ et garde une bande étroite autour du DC. Ce plaffonement a été rajouté suite à des erreurs de traitement, voir le fichier `filter.md`.
- Une première méthode est la troncature à un nombre entier de périodes : on peut montrer que moyenner sur un nombre entier de périodes de référence annule exactement les contributions périodiques à  $2 f_{\text{ref}}$ (lorsque le signal couvre au moins une période). Cette méthode est toujours utilisée en premier.
- Deuxième façon : `"butterworth"` (`sos` + `sosfiltfilt`) : utilise des sections d'ordre 2 (`sos`, numériquement plus stables qu'une forme polynomiale) et un filtrage avant-arrière (`sosfiltfilt`) sans déphasage propre, avec extension périodique si la fenêtre est courte (`3 * fs / cutoff`) pour limiter les effets de bord, puis moyenne sur la fin du signal stabilisé.
* Mode `"moving_average"` : moyenne directe sur le signal (tronqué au nombre entier de périodes si possible), plus simple et rapide qu'un filtre IIR.

### Fonction `demodulate_signal`

Exécute la projection à partir d'un coupe `(t, v_measured)` à la fréquence `f_ref` (et à l'harmonique `lockin_cfg.harmonic` (il vaut 1 par défaut, mais on a choisit de pouvoir spécifier l'harmonique si c'était nécessaire à un moment)) :

$$s_X(t) = v_{\text{mesurée}}(t) \cdot C \sin(\omega_0 t), \qquad s_Y(t) = v_{\text{mesurée}}(t) \cdot C \cos(\omega_0 t)$$

où $\omega_0 = 2\pi \cdot \text{harmonic} \cdot f_{\text{ref}}$, et $C = 2$ (`"peak"`, restitue l'amplitude crête) ou $C = \sqrt{2}$ (`"rms"`, restitue la valeur efficace).

- Déduction de `dt` et `fs` depuis `t` : évite de demander `fs` et on est sur que c'est la bonne valeur.

### Fonction `demodulate_time_data`

Adapte juste `demodulate_signal` à l'objet `TimeDomainData` renvoyé par la simulation temporelle du circuit RC.

- Choix `use_noisy` (`true` par défaut) : permet de choisir entre entre le signal bruité et `v_out_clean`.

### Classe `LockInAmplifier`

L'objectif est toujours le meme : conserver l'état de conf : Éviter de repasser l'objet de conf à chaque réglage.

### Fonction `demodulate_sweep`

Simule un balayage complet en fréquence pour reconstruire un objet `SpectrumData`.

- Pourquoi appeler `generate_time_response` à chaque fréquence `f` ? : pour obtenir une mesure simulée avec bruit et filtrage lock-in à chaque fréquence du balayage, comparable aux données expérimentales ou à un calcul de Kramers-Kronig (contrairement à `compute_spectrum`, qui donne la formule théorique directe).
Voici une version plus précise de ce point, prête à remplacer ou compléter ton bullet :

- `fs_point = max(fs, 10.0 * f)` (fréquence d'échantillonnage, en Hz) : fixe le pas temporel `dt = 1 / fs_point` pour ce point de fréquence. Une fréquence d'échantillonnage fixe `fs` peut devenir trop juste si `f` est élevée : imposer `>= 10 * f` garantit au moins 10 échantillons par période, ce qui  au-dessus de la limite de Shannon, donc ok.
- `duration_point = max(duration, 5.0 / f)` (durée d'acquisition locale, en secondes) : fixe la longueur totale de la fenêtre temporelle `t` pour ce point. Une durée fixe `duration` peut ne couvrir qu'une fraction de période quand `f` est très basse : imposer `>= 5 / f` garantit au moins 5 périodes complètes, ce qui donne normalement assez d'échantillons et de cycles pour que le filtrage passe-bas (ou le moyennage sur des périodes entières) puisse bien fonctionner.

- Attention, importer `generate_time_response` en local pour éviter les dep circulaires.

## `demodulate_signal` vs `demodulate_sweep`

- `demodulate_signal` = mesure ponctuelle à une fréquence fixe :
Prend un signal temporel déjà existant `(t, v_measured)` et `f_ref`, puis renvoie un quadruplet `(X, Y, R, phase)` pour cette fréquence unique.

- `demodulate_sweep` = balayage multi-fréquences complet :
Boucle sur un vecteur de fréquences, simule à chaque fréquence la réponse temporelle du circuit RC, applique la démodulation lock-in et assemble $X(f) + jY(f)$ vers `SpectrumData`.
C'est ce résultat qu'il faudra confronter à `compute_spectrum` pour vérifier Kramers-Kronig.
### CheatSheet

| Critère | `demodulate_signal` | `demodulate_sweep` |
| --- | --- | --- |
| Entrée principale | `t`, `v_measured`, `f_ref`, `lockin_cfg` | `rc_config`, `freqs`, `duration`, `fs`, `noise_cfg`, paramètres LPF |
| Nombre de fréquences | Une seule (`f_ref`) | Plusieurs (balayage sur `freqs_arr`) |
| Avec simu RC | Non (reçoit déjà `v_measured`) | Oui (appelle `generate_time_response` pour chaque `f`) |
| Sortie | Tuple `(X, Y, R, phase)` | Objet `SpectrumData` (`freqs`, `H_complex`, `X`, `Y`, module, phase) |
