# Filtre passe bas du Lockin
Le filtrage passe bas pour enlever la fréquence double est éffectué de deux façons

## Filtrage bidirectionnel `sosfiltfilt` à phase nulle

### Forme SOS
Un filtre numérique d'ordre élevé possède des polynômes de grande dimension. Avec des floats de 64 bits, cela peut créer une instabilité numérique à cause d'erreurs d'arrondi sur les coefficients. La mise sous forme SOS (Second-Order Sections) découpe le filtre complexe en une cascade de petits filtres biquadratiques d'ordre 2. Cela revient au meme.

### Filtrage bidirectionnel
Un filtre passe-bas temps réel classique introduit un déphasage qui dépendent de la fréquence $\phi(\omega)$. 

L'objectif du Lock-in est de mesurer la phase exacte $\theta$. Si le filtre passe-bas du Lock-in ajoute lui-même un déphasage instrumental $\phi_{filtre}$, la mesure de phase devient incorrecte !
* La solution avec `sosfiltfilt` : 
  1. Le signal est filtré dans le sens direct ($t = 0 \to T$), ce qui applique un déphasage $+\phi(\omega)$.
  2. Le signal filtré est inversé dans le temps et filtré une seconde fois dans le sens inverse ($t = T \to 0$), ce qui applique un déphasage $-\phi(\omega)$.
  3. Résultat : $\phi_{total}(\omega) = +\phi(\omega) - \phi(\omega) = 0$. Le filtre a une phase strictement nulle ($\theta_{mesuré} = \theta_{réel}$) et l'atténuation du filtre est doublée (un ordre 2 devient équivalent à un ordre 4).

## Moyennage glissant sur un nombre entier de périodes ($N \cdot T_{ref}$)

Lors de la détection sensible à la phase, le signal bruité $v(t) = V_0 \sin(\omega_0 t + \phi)$ est multiplié par la référence $2\sin(\omega_0 t)$. En utilisant les identités trigonométriques :

$s_X(t) = 2 V_0 \sin(\omega_0 t + \phi) \sin(\omega_0 t) = \underbrace{V_0 \cos(\phi)}_{DC} - \underbrace{V_0 \cos(2\omega_0 t + \phi)}_{os} 2\omega_0$

Après multiplication, le signal contient deux termes :
1. Une valeur continue $V_0 \cos(\phi)$ qui est la quantité physique exacte que l'on veut mesurer.
2. Une oscillation parasite à la fréquence double $2\omega_0$.

### L'idée physique du moyennage sur $N \cdot T_{ref}$
L'intégrale (ou la moyenne) d'une fonction sinusoïdale sur une période complète $T_0 = \frac{2\pi}{\omega_0}$ (ou sur un multiple entier $N \cdot T_0$) est nulle :

$\int_{0}^{N T_0} \cos(2\omega_0 t + \phi) \, dt = 0$

En moyennat sur une durée quelconque, l'oscillation à $2\omega_0$ ne s'annule pas complètement. Le "morceau de sinusoïde" restant crée une erreur. Si l'on tronque exactement sur $N \cdot T_{ref}$ : La moyenne du terme à $2\omega_0$ tombe exactement à 0, isolant la composante continue DC pure $V_0 \cos(\phi)$ sans aucune ondulation résiduelle.

## Fréquence de coupure
Dans `fixed_freq.png` produit par la première version du programme, on peut observer une discontuité des points observés, accompagnée d'un défaut de lissage observés autour de $\omega \approx 10^2 \text{ rad/s}$ ($f \approx 15,9 \text{ Hz}$).

La pulsation et la fréquence sont liées par $\omega = 2\pi f$. La zone autour de $\omega = 10^2 \text{ rad/s}$ correspond donc à $f = \frac{100}{2\pi} \approx 15,9 \text{ Hz}$. Pour $\omega = 10^1 \text{ rad/s}$, on a $f \approx 1,59 \text{ Hz}$. Le rôle de la fréquence de coupure $f_c^{LPF} = 10 \text{ Hz}$ : Dans le code de balayage du Lock-in, le filtre passe-bas est configuré avec une fréquence de coupure fixe $f_c^{LPF} = 10 \text{ Hz}$.

Le problème à basse fréquence ($f_{ref} < 10 \text{ Hz}$) : 
  Lors de la démodulation synchrone (PSD), la multiplication par la référence sinusoïdale génère deux composantes : la composante continue DC (le signal utile) et une composante à la fréquence double $2 f_{ref}$.
  - Pour $f_{ref} < 10 \text{ Hz}$ ($\omega < 62,8 \text{ rad/s}$), la fréquence parasite $2 f_{ref}$ se retrouve dans ou très proche de la bande passante du filtre ($2 f_{ref} < 20 \text{ Hz}$). 
  - Le filtre passe-bas à $10 \text{ Hz}$ n'atténue pas cette oscillation à $2f_{ref}$, qui vient polluer la moyenne DC et faire gonfler artificiellement la valeur de $\chi'(X)$ au-dessus de $1.0$ (la "bosse" entre $\omega = 10^1$ et $10^2$).

La transition brutale à $f_{ref} \approx 15,9 \text{ Hz}$ ($\omega \approx 10^2 \text{ rad/s}$) :
  Dès que $f_{ref}$ dépasse $\sim 15 \text{ Hz}$, la fréquence double $2 f_{ref} > 30 \text{ Hz}$ entre enfin dans la bande d'atténuation du filtre. Le terme parasite est brusquement coupé et la mesure de $\chi'(X)$ retombe sur sa valeur physique réelle ($1.0$), créant la chute/discontinuité soudaine à $\omega = 10^2 \text{ rad/s}$.

Pour éliminer ce défaut et obtenir un spectre mesuré et reconstruit parfaitement lisses sur tout le domaine :

Il faut donc rendre la fréquence de coupure du Lock-in adaptative à la fréquence de référence : `lpf_cutoff = min(10.0, 0.1 * f_ref)`