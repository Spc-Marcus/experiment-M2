# experiment-M2

Outils ILP (Integer Linear Programming) pour la détection de sous-matrices denses dans des matrices binaires, avec plusieurs variantes de modèles Gurobi et un système de logs configurable.

---

## Structure du projet

```
experiment-M2/
├── Max-cli/          # CLI simple : 1 matrice → 1 sous-matrice dense
│   ├── config.arg    # Fichier de configuration
│   ├── main.py       # Point d'entrée
│   └── model_call.py # Couche d'appel aux modèles ILP
├── No-error/         # Expériences multi-itérations (matrices synthétiques sans bruit)
│   ├── config.arg
│   ├── ilp_grb.py    # Implémentation des modèles ILP (V1–V3c)
│   ├── ilp.py        # Orchestration du biclustering exhaustif
│   ├── main.py
│   └── post_processing.py
├── onlyOne/          # Variante "only-ones" avec pré-processing
│   ├── config.arg
│   ├── ilp.py
│   ├── main.py
│   ├── pre_processing.py
│   └── post_processing.py
├── model/            # Modèles Gurobi (max_one, max_e_r)
├── utils/
│   ├── create_matrix.py    # Génération de matrices synthétiques (V1)
│   ├── create_matrix_V2.py # Génération déterministe par seed (V2)
│   ├── logger.py           # Système de logs configurable
│   ├── parser.py           # Lecture des fichiers .arg
│   └── reconstruire.py
├── graphique/        # Visualisation des résultats
│   └── plot_results.py
├── visu/
│   └── matrix_visualizer.html
├── doc/              # Documentation technique
└── run.sh            # Script SLURM
```

---

## Prérequis

- Python ≥ 3.10  
- [Gurobi](https://www.gurobi.com/) avec une licence valide (WLS ou académique)  
- Bibliothèques Python : `numpy`, `gurobipy`

### Installation des dépendances Python

```bash
pip install numpy gurobipy
```

### Configuration de la licence Gurobi

Les clés de licence sont lues depuis les variables d'environnement. Vous pouvez aussi les définir directement dans `No-error/ilp_grb.py` ou `onlyOne/ilp_grb.py` :

```python
os.environ['GRB_WLSACCESSID'] = '<votre_access_id>'
os.environ['GRB_WLSSECRET']   = '<votre_secret>'
os.environ['GRB_LICENSEID']   = '<votre_license_id>'
```

---

## Utilisation

### 1. Max-cli – Trouver la plus grande sous-matrice dense

Le module **Max-cli** est l'outil le plus simple : il prend une matrice (générée aléatoirement ou chargée depuis un CSV) et retourne la plus grande sous-matrice dense selon un paramètre `gamma`.

#### Configuration (`Max-cli/config.arg`)

```
new-gen: 1            # 1 = générer une nouvelle matrice, 0 = charger depuis 'source'
source: Mat/toto.csv  # Fichier source (utilisé si new-gen=0)
nom: output           # Préfixe du fichier de résultat
seed:                 # Seed aléatoire (vide = auto)
rows: 20              # Nombre de lignes (new-gen=1 seulement)
cols: 15              # Nombre de colonnes (new-gen=1 seulement)
density: 0.7          # Densité de la matrice générée (new-gen=1 seulement)
gamma: 0.025          # error_rate
model: max_one_v2     # Modèle ILP : max_one et/ou max_e_r_v2
heuristic: 1          # Utilisation d'une heuristique
output_dir: Max-cli/results  # Dossier de sortie
log_level: INFO       # Niveau de log : DEBUG, INFO, WARNING, ERROR
log_file:             # Fichier de log (vide = console uniquement)
```

#### Exécution

```bash
# Depuis la racine du projet
python Max-cli/main.py

# Avec un fichier de configuration personnalisé
python Max-cli/main.py chemin/vers/mon_config.arg
```

#### Exemple de sortie

```
2026-01-01 12:00:00 [INFO] experiment: Configuration chargée …
2026-01-01 12:00:00 [INFO] experiment: Génération d'une matrice aléatoire 20x15 (densité=0.70, seed=42).
2026-01-01 12:00:05 [INFO] experiment.model_call: Sous-matrice trouvée : 14 lignes × 12 colonnes (densité réelle = 0.9762).
Sous-matrice trouvée : 14 lignes × 12 colonnes — résultats dans 'Max-cli/results/output_1735689600.txt'
```

Le fichier texte de résultat contient la sous-matrice et les indices des lignes/colonnes sélectionnées.

---

### 2. No-error – Expériences multi-itérations

Ce module effectue des expériences comparatives sur plusieurs modèles ILP avec des matrices synthétiques.

#### Configuration (`No-error/config.arg`)

```
stripe: 7,8        # Nombre de bandes (liste)
haplotypes: 9,10   # Nombre d'haplotypes (liste)
min_row: 3
max_row: 50
min_col: 5
max_col: 25
nb_it: 10          # Nombre d'itérations par combinaison
max_one: 1         # Activer le modèle max_one (0/1)
max_one_v2: 1
max_one_v3: 1
max_e_r: 1
resultat_file: results.csv
log_error: 1       # Sauvegarder les erreurs dans temp/
```

#### Exécution

```bash
python No-error/main.py
```

Les résultats sont écrits dans `No-error/results.csv`.

---

### 3. onlyOne – Variante avec pré-processing HCA

Similaire à No-error mais avec un pré-processing basé sur le clustering hiérarchique pour identifier les colonnes évidentes.

#### Exécution

```bash
python onlyOne/main.py
```

---

### 4. Système de logs configurable

Tous les modules utilisent le logger centralisé `utils/logger.py`.

```python
from utils.logger import setup_logger

logger = setup_logger(
    name="mon_module",
    level="DEBUG",          # DEBUG | INFO | WARNING | ERROR | CRITICAL
    log_file="logs/run.log" # None = console uniquement
)

logger.info("Démarrage")
logger.debug("Valeur : %s", 42)
logger.warning("Attention !")
logger.error("Erreur critique : %s", "message")
```

---

### 5. Génération de matrices aléatoires (standalone)

```bash
python utils/create_matrix_V2.py
```

Ou depuis Python :

```python
from utils.create_matrix_V2 import create_matrix
import numpy as np

matrix = create_matrix(L=10, C=8, density=0.75, seed=42)
arr = np.array(matrix)
print(arr)
```

---

## Exemples rapides

### Exemple 1 – Générer une matrice et trouver la sous-matrice dense

```bash
# Éditer Max-cli/config.arg :
#   new-gen: 1
#   rows: 15
#   cols: 10
#   density: 0.8
#   gamma: 0.95
#   model: max_one_v2
#   log_level: INFO

python Max-cli/main.py
```

### Exemple 2 – Charger une matrice CSV existante

```bash
# Éditer Max-cli/config.arg :
#   new-gen: 0
#   source: Mat/ma_matrice.csv
#   gamma: 0.975
#   model: max_e_r_v2

python Max-cli/main.py
```

### Exemple 3 – Activer les logs détaillés vers un fichier

```bash
# Éditer Max-cli/config.arg :
#   log_level: DEBUG
#   log_file: logs/debug.log

python Max-cli/main.py
```

### Exemple 4 – Lancer une campagne d'expériences No-error

```bash
# Éditer No-error/config.arg selon vos besoins, puis :
python No-error/main.py
```

---

## Script SLURM

Pour exécuter sur un cluster HPC avec SLURM :

```bash
sbatch run.sh
```
