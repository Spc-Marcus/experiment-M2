# T1 — Pipeline d'expérimentation

> **Périmètre : expérimentation uniquement — pas d'analyse, pas de visualisation.**
>
> Ce dossier contient un pipeline reproductible qui exécute des solveurs exacts et
> des heuristiques, collecte des métriques brutes, et produit un CSV + des logs JSON
> par exécution. L'analyse et les graphiques sont hors périmètre ici (voir
> `T1_test_plan.md` pour le plan d'analyse complet).

---

## Structure du répertoire

```
T1/
├── run_experiment.py      # Point d'entrée CLI (orchestre le pipeline)
├── config.arg             # Configuration de travail (valeurs par défaut pour test rapide)
├── config.arg.example     # Exemple de configuration entièrement documenté
├── README.md              # Ce fichier
├── pipeline/              # Modules fonctionnels (une responsabilité chacun)
│   ├── __init__.py
│   ├── config.py          # parse_file / build — lecture et valeurs par défaut de la config
│   ├── discovery.py       # discover_solvers / discover_heuristics / resolve_all
│   ├── metrics.py         # matrix_to_model_inputs / compute_metrics / compute_gap
│   ├── io.py              # CSV_HEADER / init_csv / append_csv_row / write_json_log
│   ├── executor.py        # run_exact_solver / run_heuristic (ne lève jamais)
│   ├── planner.py         # plan_runs / print_plan / discover_instances
│   └── runner.py          # execute_pipeline / run_quick_check
└── results/               # Créé à l'exécution
    ├── results_<ts>.csv   # Un CSV par invocation (toutes les exécutions)
    └── logs/
        └── *.json         # Un log JSON par exécution

utils/ (au niveau du dépôt, réutilisable)
├── env_info.py            # collect() — hash git, version python, pip freeze
└── matrix_io.py           # load_csv_matrix() — chargeur CSV avec détection automatique
```

---

## Démarrage rapide

```bash
# Depuis la racine du dépôt

# 1. Valider le pipeline avec une matrice minimale 5×5
python T1/run_experiment.py --quick-check

# 2. Prévisualiser les exécutions planifiées sans les lancer (dry run)
python T1/run_experiment.py --dry-run

# 3. Exécution complète (lit T1/config.arg)
python T1/run_experiment.py

# 4. Spécifier un fichier de configuration différent
python T1/run_experiment.py chemin/vers/ma_config.arg

# 5. Modifier le niveau de journalisation
python T1/run_experiment.py --log-level DEBUG
```

---

## Format de configuration (`clé=valeur`)

Le fichier de configuration utilise la syntaxe simple `clé=valeur`. Les lignes
commençant par `#` sont des commentaires. Les listes sont des valeurs
**séparées par des virgules** sur une seule ligne.

### Référence complète des clés

| Clé | Défaut | Description |
|-----|--------|-------------|
| `instances_dir` | `Mat` | Répertoire des instances CSV réelles |
| `instances` | *(non défini)* | Liste explicite de noms de fichiers CSV dans `instances_dir` ; si absent, tous les fichiers `*.csv` sont utilisés |
| `synthetic` | `false` | `true` → générer des matrices ; `false` → charger depuis `instances_dir` |
| `synthetic_specs` | `L:50,C:50,density:0.35` | Dimensions et densité de base pour les matrices synthétiques |
| `repetitions` | `5` | Nombre d'exécutions indépendantes par paire `(instance, γ)` ; chaque exécution génère une graine aléatoire (enregistrée dans CSV/logs) |
| `gammas` | `0.9,0.95,0.99,1.0` | Densités minimales cibles de la sous-matrice ; `error_rate = 1 − γ` |
| `solvers` | *(non défini)* | Noms des classes de solveurs exacts (voir ci-dessous) |
| `heuristics` | *(non défini)* | Noms des fonctions heuristiques (voir ci-dessous) |
| `heuristic_solver` | `ALL` | Solveur injecté comme `model_class` dans les heuristiques : `ALL` → chaque solveur configuré est utilisé par exécution heuristique ; un nom de classe → seul ce solveur est utilisé (doit figurer dans `solvers`) |
| `timeout_exact` | `600` | Limite de temps (s) pour chaque exécution de solveur exact |
| `timeout_heuristic` | `150` | Limite de temps (s) pour chaque exécution heuristique |
| `output_dir` | `T1/results` | Répertoire de sortie (relatif à la racine du dépôt, ou absolu) |
| `parallel_jobs` | `1` | `1` = séquentiel ; `N` = pool de N threads (ordre exact→heuristique préservé dans un groupe) |
| `dry_run` | `false` | Affiche les exécutions planifiées, sans exécution |
| `quick_check` | `false` | Exécution de validation minimale sur une matrice 5×5 |

### Comment écrire `solvers`

Le pipeline parcourt `model/final/` pour toutes les sous-classes de `BiclusterModelBase`.

| Format | Exemple | Comportement |
|--------|---------|--------------|
| `NomClasse` | `MaxOneModel` | Recherche dans tous les modules sous `model/final` |
| `module:NomClasse` | `max_one_final:MaxOneModel` | Cible un fichier spécifique |
| `model.final.module:NomClasse` | `model.final.max_one_final:MaxOneModel` | Chemin complet |

Plusieurs solveurs séparés par des virgules :

```
solvers=MaxOneModel,MaxSurfaceModel
```

### Comment écrire `heuristics`

Le pipeline parcourt `model/heuristics/` pour toutes les fonctions appelables.

| Format | Exemple | Comportement |
|--------|---------|--------------|
| `nom_fonction` | `heuristicA` | Recherche dans tous les modules sous `model/heuristics` |
| `module:nom_fonction` | `heuristicA:heuristicA` | Cible un fichier spécifique |
| `model.heuristics.module:fonction` | `model.heuristics.heuristicA:heuristicA` | Chemin complet |

### Fonctionnement de `heuristic_solver`

Chaque fonction heuristique reçoit un argument `model_class` de type
`Type[BiclusterModelBase]`. C'est la classe de solveur qu'elle utilise en
interne pour les sous-problèmes MIP de chaque phase. La colonne `solver` de
la ligne de résultat enregistre quelle classe a été injectée.

| Valeur | Effet |
|--------|-------|
| `heuristic_solver=ALL` | Chaque solveur listé dans `solvers` est injecté à tour de rôle — une ligne CSV par paire `(heuristique, solveur)` |
| `heuristic_solver=MaxOneModel` | Seul `MaxOneModel` est injecté, quel que soit le nombre de solveurs dans `solvers` |

Exemple :
```
solvers=MaxOneModel,MaxSurfaceModel
heuristics=heuristicA
heuristic_solver=MaxOneModel   # heuristicA s'exécute uniquement avec MaxOneModel
```

---

## Format de sortie CSV

Un fichier CSV par invocation est écrit dans `T1/results/results_<horodatage>.csv`.

**En-tête :**

```
instance_id,m,n,base_dens,gamma,solver,seed,heuristic,time,status,objective,area,density,gap
```

| Colonne | Description |
|---------|-------------|
| `instance_id` | Nom du fichier sans extension (réel) ou `synthetic_L{L}_C{C}_d{density}_s{seed}` |
| `m`, `n` | Dimensions de la matrice |
| `base_dens` | Densité globale de la matrice d'entrée |
| `gamma` | Densité cible minimale (valeur de configuration) |
| `solver` | Nom de la classe de solveur utilisée |
| `seed` | Graine générée dynamiquement pour cette exécution (génération de matrice synthétique + RNG heuristique) |
| `heuristic` | Nom de la fonction heuristique, ou `NA` pour les exécutions exactes |
| `time` | Temps réel en secondes |
| `status` | `optimal` \| `time_limit` \| `error` \| chaîne de statut Gurobi |
| `objective` | Nombre de 1 dans la sous-matrice sélectionnée (calculé à partir des indices ligne/colonne) |
| `area` | `#lignes_sélectionnées × #colonnes_sélectionnées` |
| `density` | `objective / area` (0 si area = 0) |
| `gap` | `100 × (best_known − objective) / best_known` (%) ou `NA` |

`best_known` est le maximum d'`objective` observé sur tous les solveurs exacts pour
le même groupe `(instance, gamma)`.

---

## Format des logs JSON

Un fichier par exécution dans `T1/results/logs/<run_id>.json`. Le contenu comprend :

- `run_id`, `instance_id`, `solver`, `heuristic`, `gamma`, `error_rate`, `seed`
- `timeout`, `elapsed`
- `env` : `git_hash`, `python_version`, `platform`, `pip_freeze`
- `assumptions` : liste des hypothèses auto-appliquées lors de cette session
- `import_path` : chemin Python du module du solveur/heuristique appelé
- `introspected_params` (heuristique uniquement) : liste des noms de paramètres détectés
- `raw_status` : code entier retourné par le solveur
- `selected_rows`, `selected_cols` : listes d'indices
- `model_ObjVal` (exact uniquement) : ObjVal rapporté par le modèle
- `traceback` : chaîne de traceback complète en cas d'erreur, sinon `null`
- `csv_row` : le dict exact écrit dans le CSV

---

## Reproductibilité

- **Matrices synthétiques** : générées avec `utils/create_matrix_V2.create_matrix(L, C, density, seed)`.
  Seule la `seed` est stockée (pas la matrice), elle peut donc être régénérée exactement
  en appelant la même fonction.
- **Heuristiques** : reçoivent la `seed` configurée et le runner fixe également
  `random.seed(seed)` et `numpy.random.seed(seed)` avant chaque appel heuristique.

---

## Hypothèses par défaut (appliquées quand la configuration est incomplète)

Ces hypothèses sont journalisées à l'exécution et stockées dans chaque log JSON.

| Situation | Comportement par défaut |
|-----------|------------------------|
| `instances_dir` absent | Utiliser `Mat` |
| `solvers` vide et `synthetic=true` | Auto-sélectionner la première classe trouvée dans `model/final` |
| `heuristic_solver` nomme un solveur absent de `solvers` | Repli sur ALL les solveurs avec un AVERTISSEMENT |
| `repetitions` absent | Utiliser `5` |
| `gammas` vide | Utiliser `[0.95]` |
| `synthetic_specs` absent | `L=50, C=50, density=0.35` |
| Séparateur CSV réel inconnu | Détection automatique à partir de la première ligne ; journaliser la décision |
| Première ligne CSV réelle est un en-tête | Détection automatique de la première ligne non numérique ; l'ignorer ; journaliser la décision |
| Signature de l'heuristique différente | Introspection avec `inspect.signature` ; correspondance des paramètres par nom ; ignorer les paramètres positionnels requis inconnus avec `status=error` |
| `setParam` du solveur échoue | Journaliser l'avertissement et continuer (le solveur gère son propre timeout en interne si possible) |
| `gurobipy` indisponible / pas de licence | L'exécution est ignorée ; `status=error` avec message complet ; le pipeline continue |

---

## Reproduire `quick_check`

```bash
python T1/run_experiment.py --quick-check
```

Lance un solveur exact + (si configuré) une heuristique sur une matrice synthétique 5×5
(`density=0.35, seed=42, gamma=0.9`) et vérifie que :

1. Le runner se termine sans exception non gérée.
2. Un CSV avec au moins une ligne de données est produit.
3. Au moins un fichier log JSON est produit.

En cas d'échec du `quick_check`, un fichier de diagnostic est écrit dans
`T1/results/quick_check_diagnostic.txt` et le processus se termine avec le code 1.

---

## Dépendances

Bibliothèque standard uniquement (`csv`, `concurrent.futures`, `inspect`, `json`, …) plus
`numpy` (déjà utilisé par `model/heuristics`). Aucun paquet supplémentaire requis
au-delà de ce qui est déjà présent dans l'environnement du dépôt.

Un `requirements.txt` n'est pas ajouté car aucune nouvelle dépendance externe n'est
introduite.
