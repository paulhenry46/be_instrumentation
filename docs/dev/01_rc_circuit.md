# Présentation du module de simulation du circuit RC
Cette documentation correspond au module `simu_lockin_kk.physics.rc_circuit`.
## Vue d'ensemble

### fonction `compute_transfer_function`

On modélise la physique du filtre passe-bas tient dans :

$$H(\omega) = \frac{1}{1 + j\omega RC} = \frac{1}{1 + (\omega\tau)^2} - j\frac{\omega\tau}{1 + (\omega\tau)^2}$$



où $\tau = RC$. Centraliser cette formule évite de la réécrire ailleurs et garantit une validation unique.
- plutôt que d'écrire `1 / (1 + 1j * w_arr * tau)` directement, le code sépare explicitement partie réelle et partie imaginaire. Cela rend les termes plus lisibles.
- On essaie de faire au maximum des fonctions vectorielles ou vectorisables par soucis de performance (et de simplicité). On accepte donc aussi bien un `float` que `np.ndarray`.

### fonction `compute_spectrum`

- On utilise une échelle logarithmique par défaut (`np.logspace`) : on va observer les phénomènes sur plusieurs décades de fréquence. C'est donc plus pratique.
- Paramètre `freqs` facultatif : permet de garder un comportement pratique par défaut tout en nous laissant imposer une grille précise si nécessaire.

### fonction `generate_time_response` 

- au lieu d'intégrer l'équation différentielle qui créerait un régime transitoire de mise en route, la fonction calcule directement directment 'expression en régime permanent :

$$v_{\text{out,clean}}(t) = A \, \vert{}H(\omega_{\text{ref}})\vert{} \, \sin\big(\omega_{\text{ref}} t + \arg H(\omega_{\text{ref}})\big)$$

- On utilise le critère de Shannon : empêche l'aliasing
- On ajoute aussi par sécurité `endpoint=False` dans `np.linspace` : sur un intervalle `[0, duration[`, exclure `duration` évite de répéter le même instant de phase si l'on concatène ou analyse des périodes successives.
- Conservation de `v_out_clean` et `np.copy` : garder séparément le signal propre et le signal bruité permet de comparer « vérité terrain vs mesure bruitée » (si besoin de mesurer le SNR ou faire une analyse des incertitudes). 
- Attention, ne pas importer `apply_noise` en global, merci. L'import local de `apply_noise` évite une dépendance circulaire

### classe `RCCircuit`

- La classe `RCCircuit` est utilisée pour respecter la POO et encapsule `config` et expose `R`, `C`, `tau`, `fc` en propriétés en lecture seule.

## `generate_time_response` vs `compute_spectrum`


- `compute_spectrum` = ce que fait le circuit à toutes les fréquences :
Calcule la réponse harmonique complexe $H(\omega)$ (ou $A \cdot H(\omega)$) sur un ensemble de fréquences (par défaut logarithmique de `f_min` à `f_max`).
On peut s'en servir pour tracer un diagramme de Bode, vérifier la fréquence de coupure. C'est une vision stationnaire en fréquence.


- `generate_time_response` = domaine temporel : à quoi ressemble le signal mesuré à une fréquence fixe  :
Fixe une seule fréquence `f_ref`, échantillonne le temps (`t` à fréquence `fs` pendant `duration`), génère l'entrée sinusoïdale $v_{\text{in}}(t)$, applique l'atténuation et le déphasage de régime permanent pour obtenir $v_{\text{out,clean}}(t)$, puis ajoute éventuellement du bruit (`v_out_noisy`).
On peut s'en servir pour simuler un acquisition réelle. On ne peut en effet pas injecter du bruit ni tester l'algo de traitement du signal sur un simple spectre théorique sans échantillons temporels.


### CheatSheet
| Critère | `compute_spectrum` | `generate_time_response` |
| --- | --- | --- |
| Axe [0] | Fréquence `freqs` / pulsation $\omega$ | Temps `t` (cadencé par `fs` sur `duration`) |
| Fréquence étudiée | Plein | fréquence unique `f_ref` |
| Sortie | `SpectrumData` ( = réponse complexe, X, Y, phase, etc...) | `TimeDomainData` (`t`, `v_in`, `v_out_clean`, `v_out_noisy`) |
| Bruit | Non  | facultatif, avec `noise_cfg` |

