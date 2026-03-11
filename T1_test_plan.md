# Plan de tests — T1 : Recherche de grandes sous-matrices denses ($\gamma$)

## Contexte
Données de départ : matrice binaire de taille $m\times n$ et un paramètre de densité $\gamma$.
Objectif : identifier la sous-matrice la plus large et dense possible (maximiser surface/ones sous contrainte de densité).

## Objectifs de l'expérimentation
- Comparer les algorithmes exacts et heuristiques en termes de **temps** et **qualité**.
- Mesurer robustesse et scalabilité selon taille des instances et densité globale.
- Produire un protocole reproductible et des sorties prêtes pour analyse (CSV / plots).

## Méthodes à évaluer

- Méthodes exactes
  - `max-one` (fichiers `max_one_final.py` ou variantes dans `model/`)
  - `max-surface` (fichier `max_surface_final.py` ou variantes dans `model/`)
  - Autre idée

- Méthodes heuristiques
  - Approche heuristique A : réduction/sélection de blocs de lignes puis colonnes (voir `model/heuristic.py`)
  - Approche heuristique B : Autre idée

## Jeux de données et paramètres

- Jeux synthétiques
  - Tailles (exemples) : petit 50×50, moyen 200×200, grand 1000×1000.
  - Densité globale (probabilité d'un 1) p ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95}.
  - Valeurs de $\gamma$ à tester : {0.9, 0.95, 0.99, 1} (selon définition de densité souhaitée).

- Données réelles
  - inclure comme cas réels, avec solution connue.

- Répétitions
  - Pour chaque combinaison (instance, $\gamma$), faire N runs (ex. N=10) pour heuristiques (variantes aléatoires), et 1 run pour exacts.

- Durée limite
  - Exacts : 10min
  - Heuristiques : 2m30s

## Protocole d'exécution

1. Pour chaque instance et valeur de $\gamma$ :
   - Lancer les solveurs exacts (`max-one`, `max-surface`, ect) avec une limites de temps T = 600s.
   - Lancer chaque heuristique pour N seeds distincts (ex. N=10), enregistrer chaque run.
2. Capturer pour chaque run : temps, status (optimal / timeout / error), solution obtenue.
3. Centraliser les résultats dans un fichier CSV unique (format décrit ci-dessous).

Remarque : pour les solveurs exacts qui utilisent un solveur externe (ex. Gurobi), noter la version et la licence.

## Métriques à collecter

- Temps : `time` (s), `time_to_best` (s si disponible), `status`.
- Qualité :
  - `objective` = nombre de 1s dans la sous-matrice trouvée.
  - `area` = (#lignes sélectionnées) × (#colonnes sélectionnées).
  - `density` = `objective` / `area`.
  - `gap` relatif (%) par rapport au meilleur connu sur l'instance : $\text{gap}=100\times(\text{best}-\text{sol})/\text{best}$.
- Ressources : usage mémoire si mesurable (optionnel).

## Format CSV (une ligne par run)

`instance_id,m,n,base_dens,gamma,solver,count,heuristic,time,status,objective,area,density,gap`

Exemple d'entrée : `toto_001,200,200,0.7,0.99,max-one,1,NA,300,timeout,4750,5000,0.95,XXX`

## Analyses et visualisations proposées

- Frontière de Pareto qualité/temps
  - Pour chaque méthode, tracer les points (`time`, `objective` ou `area`) et identifier les méthodes non dominées.
  - Lecture attendue : quelles méthodes donnent le meilleur compromis vitesse/qualité.

- Courbes anytime (qualité au fil du temps)
  - Pour chaque solveur, suivre la meilleure solution trouvée à t = 1s, 5s, 10s, 30s, 60s, 150s, 300s, 600s.
  - Lecture attendue : comparer les méthodes qui convergent vite vs celles qui convergent mieux mais tard.

- Analyse de robustesse des heuristiques
  - Boxplots du `gap` par heuristique et par `gamma` (N seeds).
  - Ajouter moyenne, médiane, écart-type, p90.
  - Lecture attendue : méthode la plus stable (faible variance) et pas seulement la meilleure moyenne.

- Sensibilité à `gamma` (phase transition)
  - Tracer `area`, `objective` et taux de faisabilité en fonction de `gamma`.
  - Lecture attendue : identifier le seuil où le problème devient brutalement plus difficile.

- Scalabilité forte
  - Tracer le temps vs taille (`m*n`) en échelle log-log, fit simple (linéaire en log-espace).
  - Lecture attendue : comparaison des pentes pour estimer la croissance empirique des coûts.

- Exact vs heuristique sous budget fixe
  - À budgets {10s, 30s, 60s, 150s, 300s, 600s}, comparer directement la qualité atteinte.
  - Lecture attendue : à partir de quel budget l'exact devient compétitif face aux heuristiques.

- Analyse des timeouts
  - Heatmap du taux de timeout par méthode selon (taille, `gamma`, densité de base).
  - Lecture attendue : zones de difficulté du problème et limites pratiques des solveurs.

- Classement global multi-instances
  - Utiliser un score normalisé par instance : `score = objective / best_instance`.
  - Produire un classement moyen et le nombre de victoires par méthode.
  - Lecture attendue : classement robuste qui évite qu'une seule grosse instance domine la conclusion.

- Tests statistiques (recommandé)
  - Comparaison paire-à-paire des méthodes via Wilcoxon signé-rang sur les scores par instance.
  - Ajouter taille d'effet (Cliff's delta) pour quantifier l'importance pratique.
  - Lecture attendue : séparer différences réelles des écarts dus au bruit expérimental.

- Stabilité structurelle des solutions
  - Sur heuristiques multi-seeds, mesurer similarité Jaccard entre ensembles de lignes/colonnes sélectionnés.
  - Lecture attendue : détecter si une méthode produit des solutions de qualité proche mais structurellement instables.

## Top 5 analyses prioritaires (si temps limité)

- Frontière de Pareto qualité/temps.
- Courbes anytime.
- Robustesse heuristiques (gap + variance par seed).
- Sensibilité à `gamma` (faisabilité + qualité).
- Exact vs heuristique sous budgets fixes.
