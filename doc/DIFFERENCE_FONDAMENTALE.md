# Pourquoi max_e_r fusionne les clusters

## Le Problème

**max_one trouve 7 haplotypes** ✅  
**max_e_r trouve 6 haplotypes** ⚠️ (fusion de 2 clusters)

## Exemple Concret Réel (Stripe 6, fichier 1769521400.txt)

Dans ce test concret avec **9 haplotypes attendus** et **matrice 62×106** :

### Résultats
- **max_one** : trouve **9 haplotypes** en 6 itérations ✅
- **max_e_r** : trouve **8 haplotypes** en 5 itérations ⚠️ (FUSIONNE 2 clusters)

### La Fusion Problématique

**max_e_r fusionne en un seul grand cluster :**
```
Cluster fusionné (16 reads) :
read_20, read_21, read_22, read_23, read_24, read_25, read_26, read_27, read_28,
read_34, read_35, read_36, read_37, read_38, read_39, read_40
```

**max_one garde séparés (2 clusters distincts) :**
```
Cluster A (7 reads) : read_34, read_35, read_36, read_37, read_38, read_39, read_40
Cluster B (9 reads) : read_20, read_21, read_22, read_23, read_24, read_25, read_26, read_27, read_28
```

### Pourquoi max_e_r Fusionne ?

Parce que **max_e_r maximise l'AIRE** (lignes × colonnes) :
- 16 reads fusionnés × colonnes communes = **GRANDE AIRE** → max_e_r préfère ✓
- Mais cette fusion **perd les colonnes privées** de chaque sous-groupe

max_one maximise le **NOMBRE DE 1s RÉELS** :
- Cluster A capte des 1s dans ses colonnes propres
- Cluster B capte des 1s dans ses colonnes propres  
- Total 1s capturés séparés **>** 1s capturés après fusion (intersection) → max_one préfère les séparer ✓

### Visualisation ASCII du phénomène

```
Situation : 2 sous-groupes avec colonnes partiellement partagées

Cluster A (7 reads) : colonnes [0-56] avec beaucoup de 1s
Cluster B (9 reads) : colonnes [0-42] avec beaucoup de 1s

Colonnes communes : [0-42] (intersection)
Colonnes privées A : [43-56] (uniquement dans A)
```

```
         Colonnes →  [0----42][43-56]
Cluster A (7 reads)  ████████ ██████  ← 1s dans toutes les colonnes
Cluster B (9 reads)  ████████        ← 1s seulement dans colonnes [0-42]
                     ↑        ↑
                    Commun   Privé A
```

**Stratégie max_e_r (maximise AIRE = lignes × colonnes) :**
```
FUSION : 16 reads × colonnes communes [0-42] = 16 × 43 = 688 cellules d'aire
         ⚠️ PERD les colonnes [43-56] pour Cluster A
```

**Stratégie max_one (maximise NOMBRE DE 1s RÉELS) :**
```
SÉPARÉS :
  - Cluster A: 7 reads × toutes colonnes [0-56] = capture TOUS les 1s de A
  - Cluster B: 9 reads × toutes colonnes [0-42] = capture TOUS les 1s de B
  - Total 1s capturés > fusion car on garde les colonnes privées [43-56] ✅
```

### La Règle Générale

**Même avec 0% d'erreur** :
- **max_e_r favorise les fusions** quand l'intersection des colonnes communes × nombre total de reads > aires séparées
  - Pense "géométrie" : grand rectangle d'aire maximale
  - Accepte de perdre des colonnes privées si cela augmente l'aire globale

- **max_one refuse les fusions** quand les colonnes privées contiennent des 1s réels qu'on perdrait
  - Pense "données" : compte tous les 1s disponibles dans la matrice originale
  - Préfère plusieurs petits rectangles qui capturent tous les 1s plutôt qu'un grand qui en perd

**C'est pourquoi max_one trouve plus d'haplotypes que max_e_r** : il garde séparés les clusters qui ont des colonnes privées différentes, tandis que max_e_r les fusionne pour maximiser l'aire géométrique.

---

## Où se Produit la "Fusion" ?

**IMPORTANT** : La fusion n'est PAS explicite dans le code. Elle est implicite dans le choix du modèle ILP.

### Code Responsable

1. **Algorithme glouton** ([No-error/ilp.py](No-error/ilp.py)) :
```python
def clustering_full_matrix(...):
    while len(remain_cols) >= min_col_quality and status:
        # Appelle le modèle ILP pour trouver LE MEILLEUR rectangle
        (reads1, reads0, cols), metrics = clustering_step(...)
        
        # Retire les lignes/colonnes couvertes
        # Répète jusqu'à épuisement
```

2. **Modèle ILP max_e_r** ([model/max_e_r_V2_grb.py](model/max_e_r_V2_grb.py#L125)) :
```python
def build_max_e_r(...):
    # OBJECTIF : Maximiser la SOMME DES CELLULES sélectionnées
    self.model.setObjective(gp.quicksum(self.lp_cells.values()), GRB.MAXIMIZE)
    #                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                        Nombre de cellules (row,col) = AIRE
```

3. **Modèle ILP max_one** ([model/max_one_grb_v2.py](model/max_one_grb_v2.py)) :
```python
# OBJECTIF : Maximiser la SOMME DES 1s RÉELS de la matrice
objective = gp.quicksum(self.lp_cells.values())
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#           Compte uniquement les cellules qui contiennent un 1 réel dans M[i,j]
```

### Comment la "Fusion" Apparaît

Le modèle ILP cherche **le meilleur rectangle possible** selon son objectif :

**Avec max_e_r** :
```
Matrice actuelle (tous les reads disponibles) :
  - read_34-40 : colonnes [0-56] avec 1s
  - read_20-28 : colonnes [0-42] avec 1s

Le solver ILP explore TOUTES les combinaisons de lignes×colonnes possibles :
  
  Option A : Prendre read_34-40 uniquement
    → 7 reads × ~57 colonnes = ~400 cellules
  
  Option B : Prendre read_20-28 uniquement  
    → 9 reads × ~43 colonnes = ~387 cellules
  
  Option C : Prendre read_34-40 + read_20-28 ensemble
    → 16 reads × colonnes [0-42] (intersection obligatoire pour 0% erreur)
    → 16 × 43 = 688 cellules ✅ MEILLEUR OBJECTIF !

Le solver choisit automatiquement Option C car 688 > 400 > 387
```

**Avec max_one** :
```
Le solver ILP explore les mêmes options mais compte les 1s RÉELS :

  Option A : read_34-40 sur colonnes [0-56]
    → Capture TOUS les 1s de ces 7 reads = ~X 1s réels
  
  Option B : read_20-28 sur colonnes [0-42]
    → Capture TOUS les 1s de ces 9 reads = ~Y 1s réels
  
  Option C : read_34-40 + read_20-28 sur colonnes [0-42] (intersection)
    → PERD les 1s dans colonnes [43-56] pour read_34-40
    → Total 1s < X (car on perd les colonnes privées)

Le solver choisit Option A car elle capture plus de 1s réels
À l'itération suivante, il choisira Option B sur la matrice résiduelle
→ Résultat : 2 clusters séparés au lieu de 1 fusionné
```

### En Résumé

- **La fusion n'est pas une opération explicite** "fusionner A et B"
- **Le solver ILP choisit librement** la meilleure combinaison lignes×colonnes selon l'objectif
- **Avec max_e_r** : le meilleur rectangle = souvent plusieurs sous-groupes fusionnés (grande aire)
- **Avec max_one** : le meilleur rectangle = souvent un seul sous-groupe pur (plus de 1s réels)
- **L'algorithme glouton** applique ce choix itérativement, créant plus ou moins de clusters selon le modèle

---

## La Raison : Deux Objectifs Différents

### Max One (Max nb de 1)
```
Objectif = Σ(i,j) M[i,j] × cell[i,j]
         = Compter les 1s RÉELS qui existent dans la matrice d'origine
```

### Max E_R (Max taille/aire)
```
Objectif = Σ(i,j) cell[i,j]
         = Compter l'AIRE du rectangle (lignes × colonnes sélectionnées)
```

**ATTENTION** : Même avec 0% erreur, ces deux objectifs sont DIFFÉRENTS !

- **Max taille** = nombre de CELLULES dans le rectangle = `|lignes| × |colonnes|`
- **Max nb de 1** = nombre de 1s RÉELS dans ces cellules = dépend de la matrice d'origine M[i,j]

Avec 0% erreur, dans UN rectangle parfait : aire = nb de 1s ✓
Mais lors d'une FUSION avec intersection de colonnes : aire ≠ nb de 1s réels disponibles ✗

---

## Avec taux d'erreur = 0 : pourquoi la différence persiste

Même avec un taux d'erreur nul (aucun 0 autorisé dans le rectangle), les deux objectifs restent différents :

- Contraintes (zéro erreur) : le rectangle ne peut couvrir que les colonnes où toutes les lignes sélectionnées valent 1. En pratique, fusionner deux clusters revient à empiler les lignes et à se restreindre à l'INTERSECTION de leurs colonnes à 1.
- Conséquence géométrique : la fusion augmente la hauteur (plus de lignes) mais réduit la largeur (colonnes partagées seulement). L'aire peut ainsi dépasser celle de chaque rectangle pris séparément.
- Conséquence sur les 1s réels : l'intersection perd les colonnes propres à chaque cluster. Le nombre total de 1s capturés baisse par rapport à « deux rectangles séparés ».

Formellement :
```
Max nb de 1  = Σ(i,j) M[i,j] × cell[i,j]  (fidélité aux données)
Max taille   = Σ(i,j) cell[i,j]            (géométrie/aire)
```

Dans l'exemple :
- Deux rectangles séparés capturent 720 + 568 = 1288 « 1s ».
- La fusion (intersection des colonnes 0–47) donne une aire 23 × 48 = 1104.

Donc, même à erreur = 0 :
- « Max taille » préfère la fusion si l'aire unique (1104) > aire de chacun des rectangles pris isolément.
- « Max nb de 1 » refuse la fusion car elle perd des 1s réels (1104 < 1288), il garde deux rectangles.

### Cas « on cherche toujours le plus grand » (greedy, un seul ensemble à la fois)

- À chaque itération, l'algorithme choisit le meilleur rectangle selon l'objectif puis retire/neutralise ce qui a été couvert.
- Avec « Max taille », empiler des lignes et réduire aux colonnes en intersection peut augmenter l’aire plus vite que la perte de largeur; le meilleur rectangle devient souvent une fusion, ce qui couvre deux clusters d’un coup et diminue le nombre final d’ensembles.
- Avec « Max nb de 1 », perdre les colonnes propres à chaque cluster est pénalisant; le meilleur rectangle est typiquement un cluster « pur » (toutes ses colonnes), laissant l’autre cluster intact pour les itérations suivantes.
- Résultat: même en ne cherchant qu’un seul ensemble à la fois, le choix du « plus grand » dépend de la métrique. La métrique « aire » favorise les fusions via intersection de colonnes; la métrique « nb de 1 » favorise la préservation des colonnes propres, donc des ensembles séparés.

### Condition de fusion sous « Max nb de 1 » (erreur = 0)

Soient deux clusters parfaits (tous 1s) `A = (R_A, C_A)` et `B = (R_B, C_B)`.
La fusion admissible (zéro erreur) ne peut utiliser que `C = C_A ∩ C_B`.

- Ones séparés: `|R_A|·|C_A| + |R_B|·|C_B|`.
- Ones fusion: `( |R_A| + |R_B| ) · |C|` avec `|C| ≤ min(|C_A|, |C_B|)`.

Deux comparaisons utiles:
- Pour battre un seul rectangle: fusion est choisie en mode greedy si
    `( |R_A| + |R_B| ) · |C| ≥ max( |R_A|·|C_A|, |R_B|·|C_B| )`.
    Cela requiert un `|C|` assez large; si l’intersection est étroite, le meilleur reste un cluster « pur ».
- Pour battre la somme des deux: il faudrait
    `( |R_A| + |R_B| ) · |C| ≥ |R_A|·|C_A| + |R_B|·|C_B|`,
    i.e. `|C| ≥ ( |R_A|·|C_A| + |R_B|·|C_B| ) / ( |R_A| + |R_B| )`.
    Comme `|C| ≤ min(|C_A|, |C_B|)`, cette condition est rarement satisfaite sauf si `C_A = C_B`.

Conclusion: « Max nb de 1 » ne fusionne pas car la perte de colonnes non partagées par tous les reads réduit le nombre de 1s capturés; la plus grande matrice en termes de 1s est souvent un des clusters d’origine, pas leur fusion.

---

## LE POINT CRUCIAL : Aire ≠ Nombre de 1s RÉELS

**AVEC 0% D'ERREUR, ON A TOUJOURS** :
- Dans un rectangle parfait (sans fusion) : aire = nombre de 1s ✓
- Mais lors d'une FUSION : les colonnes propres sont PERDUES !

### Exemple visuel — PERTE DE 1s RÉELS

```
Colonnes:   0.......................47 | 48.............70
            ↑_____INTERSECTION______↑   ↑__PROPRES B__↑

Cluster A (15 lignes):
            [111111111111111111111111] | [000000000000000]
            └── 48 colonnes de 1s ──┘   └─ 23 cols de 0 ┘
            1s réels dans A = 15 × 48 = 720 ✓

Cluster B (8 lignes):
            [111111111111111111111111] | [111111111111111]
            └── 48 colonnes de 1s ──┘   └─ 23 cols de 1s┘
            1s réels dans B = 8 × 71 = 568 ✓

Fusion C (23 lignes, SEULEMENT cols 0-47):
            [111111111111111111111111]
            └──── 48 colonnes ──────┘
            Aire = 23 × 48 = 1104
            1s réels = 1104
            ❌ PERTE : les 8×23=184 vrais 1s des cols 48-70 !
```

**COMPTAGE** :

| Choix | Rectangles | Aire totale | 1s RÉELS | 1s PERDUS |
|-------|------------|-------------|----------|-----------|
| Fusion C | 1 | 1104 | 1104 | 184 |
| A + B séparés | 2 | 1288 | **1288** | 0 |

**DONC** :
- **Max taille** : cherche la plus grande AIRE d'un rectangle → choisit C (1104 > 720)
- **Max nb de 1** : cherche le plus de 1s RÉELS TOTAUX → choisit A+B (1288 > 1104)

C'est ça la différence ! Max taille ne voit PAS les 1s perdus, max nb de 1 les voit.

---

## Exemple Concret (anciennes données)

```
Colonnes:   0................................47 | 48....................70

Cluster A (rows 11–25):
Rows:      [111111111111111111111111111111111111111111111111] | [000000000000000000000000000]

Cluster B (rows 62–69):
Rows:      [111111111111111111111111111111111111111111111111] | [111111111111111111111111111]
```

- Aire/1s de A: `|R_A|·|C_A| = 15 × 48 = 720`.
- Aire/1s de B: `|R_B|·|C_B| = 8 × 71 = 568`.
- Fusion admissible (zéro erreur) = intersection des colonnes: `C = C_A ∩ C_B = 0..47`.
- Aire/1s de la fusion C: `( |R_A| + |R_B| ) · |C| = 23 × 48 = 1104`.

Ce que choisissent les objectifs en mode « on prend un seul rectangle »:
- « Max taille » (aire): compare 1104 vs 720 et 568 → choisit la fusion C car 1104 est la plus grande aire parmi les candidats pris séparément.
- « Max nb de 1 » (1s dans le rectangle choisi): sur un seul rectangle et zéro erreur, aire = nb de 1s, donc C est aussi le plus grand parmi A ou B.

Pourquoi le résultat final peut différer (6 vs 7 haplotypes):
- Le cumul optimal en 1s sur deux rectangles séparés est `720 + 568 = 1288 > 1104`.
- Si l’algorithme privilégie la fusion à la première itération (aire maximale), il “consomme” l’intersection et peut empêcher de récupérer ensuite toutes les colonnes propres, réduisant le nombre total d’ensembles finaux.
- À l’inverse, si l’algorithme vise à maximiser le total de 1s sur plusieurs rectangles (comptabilité globale), il tend à garder A et B séparés pour atteindre 1288 plutôt que 1104.

En bref: sur un seul rectangle, « plus grand » = « plus de 1s » quand erreur = 0. La différence vient du compromis fusion vs séparation à l’échelle de plusieurs rectangles (total cumulé) et des contraintes/heuristiques de sélection dans les phases d’extension (colonnes en intersection vs colonnes propres).
### Le vrai distinguo : comptage CUMULATIF vs comptage PAR RECTANGLE

**Point clé** : l'algorithme extrait plusieurs rectangles successifs (itératif/greedy). À chaque itération, on retire les lignes/colonnes couvertes puis on recommence.

**Deux façons de compter "le meilleur"** :

1. **Max taille (aire)** : À chaque itération, cherche le rectangle avec la plus grande AIRE parmi les candidats du moment.
   - Itération 1: fusion C (aire 1104) est meilleure que A seul (720) ou B seul (568) → on prend C.
   - Résultat: 1 rectangle de 1104 cellules.
   - Les colonnes propres de B (48–70) ne sont plus exploitables car les lignes de B ont été "consommées" par C.
   - Total final: moins de rectangles (fusion a "regroupé").

2. **Max nb de 1 (total cumulé)** : Cherche à maximiser la SOMME TOTALE des 1s capturés sur tous les rectangles finaux.
   - Itération 1: si on prend C (1104), les colonnes 48–70 de B sont perdues.
   - Si on prend A (720), puis itération 2 prend B (568), total = 1288 > 1104.
   - L'algorithme anticipe (ou découvre via contraintes ILP) que garder A et B séparés capture plus de 1s au TOTAL.
   - Résultat: 2 rectangles, somme 1288 > 1 rectangle de 1104.

**Donc oui, ils comptent différemment** :
- Max taille compte l'aire d'UN rectangle à la fois, favorisant les grandes aires même si ça réduit le nombre de rectangles finaux.
- Max nb de 1 compte le TOTAL de 1s sur TOUS les rectangles successifs, favorisant la préservation des colonnes propres même si chaque rectangle individuel est plus petit.
## Exemple Concret

**Cluster 1** : Rows 11-25, Cols 0-47
- Aire : 15 × 48 = **720 cellules**
- Vraies valeurs 1 : **720**

**Cluster 2** : Rows 62-69, Cols 0-70
- Aire : 8 × 71 = **568 cellules**
- Vraies valeurs 1 : **568**

### Si on fusionne (rows 11-25+62-69, cols 0-47)

**Le problème** : Rows 11-25 n'ont PAS de 1 aux colonnes 48-70, mais rows 62-69 les ont.

**Max One** : 
```
Fusion = perte de 1s réels → Ne fusionne PAS ✅
```

**Max E_R** :
```
Aire fusionnée = 23 × 48 = 1104 > max(720, 568)
→ Fusionne car c'est plus grand ! ⚠️
```

---

## Conclusion

**max_e_r fusionne parce que c'est optimal POUR SA FONCTION OBJECTIF.**

→ **Utiliser max_one**
