# Comparaison des variantes de modèles de quasi-biclique

## Résumé rapide

Ce document explique les 7 variantes d'algorithmes d'optimisation pour la détection de quasi-bicliques. Elles diffèrent principalement par :
1. **La gestion du modèle Gurobi** : un seul modèle réutilisé vs. reconstruction à chaque phase
2. **L'utilisation de WarmStart** : suggestion d'une solution initiale pour accélérer Gurobi

---

## Les 7 variantes détaillées

### 1. **max_one (Version 1)**

**Clé config:** `max_one: 1`  
**Version:** 1  
**Fonction:** `find_quasi_dens_matrix_max_ones()`

#### Stratégie
- **Création du modèle:** Un **nouveau modèle Gurobi créé à chaque phase** (Phase 1, 2, 3)
- **WarmStart:** ❌ **NON** utilisé
- **Contraintes dynamiques:** ❌ **NON** (le modèle est entièrement reconstruit)
- **Données du modèle:** À chaque phase, seules les lignes/colonnes pertinentes sont ajoutées au modèle

#### Flux détaillé
1. **Phase 1 (Seed):** Modèle créé avec `rows_sorted × cols_sorted[:seed_cols]`
2. **Phase 2 (Extension colonnes):** Nouveau modèle créé avec `rw × all_col_indices`
3. **Phase 3 (Extension lignes):** Nouveau modèle créé avec `all_row_indices × cl`

#### Avantages
- Modèle plus petit à chaque phase → résolution potentiellement plus rapide
- Moins d'espace mémoire utilisé

#### Inconvénients
- **Perte d'information contextuelle** : les degrés des variables sont recalculés localement
- Pas de cohérence entre les modèles des différentes phases
- **Peut donner des résultats légèrement différents** des autres versions

---

### 2. **max_one_v2 (Version 2)**

**Clé config:** `max_one_v2: 1`  
**Version:** 2  
**Fonction:** `find_quasi_biclique_max_one_V2()`

#### Stratégie
- **Création du modèle:** Un **seul modèle Gurobi créé une fois au départ**
- **WarmStart:** ❌ **NON** utilisé
- **Contraintes dynamiques:** ✅ **OUI** (utilise `add_forced_cols_zero()` et `add_forced_rows_zero()`)
- **Données du modèle:** Constant - toutes les lignes/colonnes sont dans le modèle dès le départ

#### Flux détaillé
1. **Phase 1 (Seed):** 
   - Modèle créé avec `rows_sorted × cols_sorted`
   - Colonnes hors `seed_cols` forcées à 0 via contrainte `col = 0`
   
2. **Phase 2 (Extension colonnes):**
   - Libération des contraintes de la Phase 1
   - Nouvelles contraintes : forcer à 0 les colonnes hors `all_col_indices` et les lignes hors `select_rows`
   - Le même modèle est réutilisé
   
3. **Phase 3 (Extension lignes):**
   - Libération des contraintes de la Phase 2
   - Nouvelles contraintes : forcer à 0 les lignes hors `all_row_indices` et les colonnes hors `select_cols`
   - Le même modèle est réutilisé

#### Avantages
- **Cohérence totale:** tous les degrés et l'information contextuelle sont conservés
- **Réutilisation du modèle:** un seul modèle à compiler
- **Résultats déterministes:** donne toujours exactement les mêmes résultats (avant modularisation)

#### Inconvénients
- Modèle plus grand (toutes les variables présentes)
- Gurobi peut être légèrement plus lent à résoudre un gros modèle avec des contraintes

---

### 3. **max_one_v3 (Version 4)**

**Clé config:** `max_one_v3: 1`  
**Version:** 4  
**Fonction:** `find_quasi_biclique_max_one_V3()`

#### Stratégie
- **Création du modèle:** Un **seul modèle Gurobi créé une fois au départ** (identique à V2)
- **WarmStart:** ✅ **OUI** utilisé (optimisation supplémentaire)
- **Contraintes dynamiques:** ✅ **OUI** (comme V2)
- **Données du modèle:** Constant - toutes les lignes/colonnes (comme V2)

#### Flux détaillé
1. **Phase 1 (Seed):** Identique à V2
   
2. **Phase 2 (Extension colonnes):**
   - Identique à V2 pour les contraintes
   - **BONUS:** `model.add_WarmStart(rw, cl)` suggère à Gurobi une solution initiale basée sur la solution précédente
   
3. **Phase 3 (Extension lignes):**
   - Identique à V2 pour les contraintes
   - **BONUS:** `model.add_WarmStart(rw, cl)` suggère une solution initiale

#### Avantages
- **Tous les avantages de V2** (cohérence, réutilisation du modèle)
- **+ WarmStart:** Gurobi peut converger **plus rapidement** car il a une "bonne" solution de départ
- Temps de résolution potentiellement réduit sans change les résultats finaux

#### Inconvénients
- Surcharge mineure pour préparer le WarmStart (négligeable)

#### ℹ️ **Attendu:**
V3 devrait donner exactement les mêmes résultats que V2, mais potentiellement **plus rapidement**.

---

### 4. **max_one_v3a (Version 5)**

**Clé config:** `max_one_v3a: 1`  
**Version:** 5  
**Fonction:** `find_quasi_biclique_max_one_V3a()`

#### Stratégie
- **Création du modèle:** Un **seul modèle Gurobi** (comme V2/V3)
- **WarmStart:** ❌ **NON** utilisé (contrairement à V3)
- **Contraintes dynamiques:** ✅ **OUI** (comme V2)
- **Données du modèle:** Constant (comme V2/V3)

#### Flux détaillé
Identique à V3, **sauf pas de WarmStart**.

#### Avantages
- Permet de mesurer l'impact du WarmStart (comparaison V3a vs V3)
- Baseline pour tester l'efficacité du WarmStart

#### Inconvénients
- Plus lent que V3 (pas d'optimisation)

#### ℹ️ **Attendu:**
- Mêmes résultats que V2 et V3
- Temps de résolution entre V2/V1 et V3

---

### 5. **max_one_v3b (Version 6)**

**Clé config:** `max_one_v3b: 1`  
**Version:** 6  
**Fonction:** `find_quasi_biclique_max_one_V3b()`

#### Stratégie
- **Création du modèle:** Un **nouveau modèle créé à chaque phase** (comme V1)
- **WarmStart:** ✅ **OUI** utilisé
- **Contraintes dynamiques:** ❌ **NON** (le modèle est entièrement reconstruit)
- **Données du modèle:** Changent à chaque phase (seeds only)

#### Flux détaillé
1. **Phase 1 (Seed):** 
   - Modèle créé avec `rows_sorted × cols_sorted[:seed_cols]`
   
2. **Phase 2 (Extension colonnes):**
   - Nouveau modèle créé avec `rw × all_col_indices`
   - **BONUS:** `model.add_WarmStart(rw, cl)` suggère la solution précédente
   
3. **Phase 3 (Extension lignes):**
   - Nouveau modèle créé avec `all_row_indices × cl`
   - **BONUS:** `model.add_WarmStart(rw, cl)` suggère la solution précédente

#### Avantages
- Modèles plus petits → potentiellement résolution plus rapide
- WarmStart aide à converger rapidement dans le petit modèle
- Bonne balance entre taille du modèle et optimisation

#### Inconvénients
- Perte d'information contextuelle (comme V1)
- **Les résultats peuvent être légèrement différents** de V2/V3 car les degrés sont recalculés
- **Les résultats peuvent être différents de V1** car le WarmStart influence la solution

#### ℹ️ **Attendu:**
- Résultats **potentiellement différents** de V1 (cause : WarmStart)
- Résultats **potentiellement différents** de V2/V3 (cause : information locale vs globale)
- Temps de résolution potentiellement similaire à V1

---

### 6. **max_one_v3c (Version 7)**

**Clé config:** `max_one_v3c: 1`  
**Version:** 7  
**Fonction:** `find_quasi_biclique_max_one_V3c()`

#### Stratégie
- **Création du modèle:** Un **nouveau modèle créé à chaque phase** (comme V1)
- **WarmStart:** ❌ **NON** utilisé (contrairement à V3b)
- **Contraintes dynamiques:** ❌ **NON** (le modèle est entièrement reconstruit)
- **Données du modèle:** Changent à chaque phase (seeds only)

#### Flux détaillé
Identique à V1, avec modèles recréés à chaque phase **sans WarmStart**.

#### Avantages
- Identique à V1 : modèles petits, résolution potentiellement rapide
- Baseline pour mesurer l'impact du WarmStart sur modèles recréés (comparaison V3c vs V3b)

#### Inconvénients
- Identique à V1 : perte d'information contextuelle
- **Résultats peuvent différer** de V1/V2/V3

#### ℹ️ **Attendu:**
- Mêmes résultats que V1
- Temps de résolution similaire à V1

---

### 7. **max_e_r (Version 3)**

**Clé config:** `max_e_r: 1`  
**Version:** 3  
**Fonction:** `find_quasi_biclique_max_e_r_V2()`

#### Stratégie
- **Modèle mathématique différent** : utilise `MaxERModel` au lieu de `MaxOneModel`
- **Objectif:** Maximise le ratio `error_rate / total_cells` (plutôt que le nombre de 1s)
- **Approche:** Très différente des autres variantes

#### Avantages
- Peut trouver des bicliques avec une meilleure balance erreur/taille
- Approche mathématique distincte

#### Inconvénients
- **Résultats très différents** des autres versions (modèle mathématique différent)
- Ne peut pas être comparé directement avec max_one variants

#### ℹ️ **Attendu:**
- **Résultats potentiellement très différents** des autres variantes

---

## Tableau comparatif complet

| Aspect | V1 (max_one) | V2 (max_one_v2) | V3 (max_one_v3) | V3a | V3b | V3c | ER |
|--------|--------------|-----------------|-----------------|-----|-----|-----|---------------|
| **Modèle unique** | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **WarmStart** | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Contraintes dynamiques** | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| **Info contextuelle** | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| **Résultats identiques** | Baseline | ✅ à V2 | ✅ à V2 | ✅ à V2 | ✅ à ? | ✅ à V1 | ❌ |
| **Performance attendue** | Moyenne | Moyenne | **Rapide** | Lent | **Rapide** | Moyenne | Lente |

---

## Comparaisons utiles pour l'optimisation

### 1. **Impact du WarmStart sur modèle unique**
```
V3a (sans) vs V3 (avec)
→ Mesure l'accélération due au WarmStart sur un modèle stable
```

### 2. **Impact du WarmStart sur modèles recréés**
```
V3c (sans) vs V3b (avec)
→ Mesure l'accélération due au WarmStart sur modèles dynamiques
```

### 3. **Un seul modèle vs reconstruction**
```
V3 (un seul + WarmStart) vs V3b (reconstruction + WarmStart)
→ Mesure le compromis : cohérence vs taille du modèle
```

### 4. **Information contextuelle (degrés globaux vs locaux)**
```
V2/V3 vs V1/V3b/V3c
→ Impact de la recalcul des degrés sur les résultats
```

### 5. **Baseline de chaque stratégie**
```
V1 (reconstruction, sans WarmStart)
V2 (un seul modèle, sans WarmStart)
→ Comparer l'impact du design sans confondre avec l'optimisation
```

---

## Recommandations pour les tests

### Pour mesurer la **performance en temps**
```
Comparer les résultats CSV pour les colonnes "*-Time"
V3 devrait être le plus rapide
V1 et V3c peuvent être plus lents car plus de compilation de modèle
```

### Pour mesurer la **cohérence des résultats**
```
Comparer les colonnes "*-Haplotypes" et "*-Is-Equal"
V2 et V3 donneront les mêmes résultats
V3a donnera aussi les mêmes résultats que V2/V3
V1, V3b, V3c donneront potentiellement des résultats différents
```

### Pour mesurer l'**efficacité du WarmStart**
```
Comparer les temps de résolution :
- V3 vs V3a (même modèle, avec/sans WarmStart)
- V3b vs V3c (modèles recréés, avec/sans WarmStart)
```

---

## Structure du code

- **[max_one_grb.py](model/max_one_grb.py)** : Fonctions pour V1
- **[max_one_grb_v2.py](model/max_one_grb_v2.py)** : Classe `MaxOneModel` pour V2/V3/V3a
- **[max_one_grb_v3.py](model/max_one_grb_v3.py)** : Classe `MaxOneModel` pour V3b/V3c (même que V2 en contenu)
- **[ilp_grb.py](ilp_grb.py)** : Implémentations de toutes les variantes
- **[ilp.py](ilp.py)** : Dispatcher `ilp()` vers les bonnes fonctions

---

## Notes finales

### Résultats empiriques réels (120 iterations de tests)

#### Performance en temps d'exécution
```
🏆 1. max_one_v3c     : 1.4676s (PLUS RAPIDE)
🥈 2. max_e_r         : 1.4723s
🥉 3. max_one_v3b     : 1.5816s
   4. max_one          : 1.7218s
   5. max_one_v2       : 2.4239s
   6. max_one_v3       : 2.4329s
   7. max_one_v3a      : 2.4253s (PLUS LENT)
```

#### Exactitude des résultats - BASÉE SUR LES HAPLOTYPES
```
🏆 PARFAITS (0% d'erreurs haplotypes):
   - max_one        : 0/120 (0.0%)
   - max_one_v2     : 0/120 (0.0%)
   - max_one_v3     : 0/120 (0.0%)
   - max_one_v3a    : 0/120 (0.0%)
   - max_one_v3b    : 0/120 (0.0%)
   - max_one_v3c    : 0/120 (0.0%)

❌ AVEC ERREURS:
   - max_e_r        : 9/120 (7.5%)
```

#### Analyse détaillée par variante

**V1 (max_one)**
- ⏱️ Temps : 1.7218s (rapide)
- ✅ Exactitude haplotypes : 0% erreurs (PARFAIT)
- 💡 **Verdict:** Fiable, temps acceptable, version de référence

**V2 (max_one_v2) - SURPASSÉ**
- ⏱️ Temps : 2.4239s (40% plus lent que V1)
- ✅ Exactitude haplotypes : 0% erreurs (PARFAIT)
- ❌ Désavantage : Bien plus lent que V1 sans aucun bénéfice
- 💡 **Verdict:** À éviter - même résultats que V1 mais beaucoup plus lent

**V3 (max_one_v3) - IDENTIQUE À V2**
- ⏱️ Temps : 2.4329s (même que V2, WarmStart ne fonctionne pas)
- ✅ Exactitude haplotypes : 0% erreurs (PARFAIT)
- ❌ Désavantage : Le WarmStart n'apporte AUCUNE accélération
- 💡 **Verdict:** À éviter - V2 et V3 sont pratiquement identiques (même modèle global = même problème)

**V3a (max_one_v3a) - IDENTIQUE À V2 ET V3**
- ⏱️ Temps : 2.4253s (même que V2/V3)
- ✅ Exactitude haplotypes : 0% erreurs (PARFAIT)
- 💡 **Verdict:** À éviter - confirme que le WarmStart ne fonctionne pas sur modèles globaux

**V3b (max_one_v3b)**
- ⏱️ Temps : 1.5816s (7% plus rapide que V1)
- ✅ Exactitude haplotypes : 0% erreurs (PARFAIT)
- 💡 **Verdict:** Rapide, résultats corrects, mais modèles recréés compliquent la maintenance

**V3c (max_one_v3c) - LE MEILLEUR**
- 🏆 Temps : 1.4676s (14% plus rapide que V1, 40% plus rapide que V2/V3)
- ✅ Exactitude haplotypes : 0% erreurs (PARFAIT)
- ✅ Simple : modèles locaux, pas de WarmStart complexe
- 💡 **Verdict:** OPTIMAL - Combine rapidité et exactitude

**max_e_r - APPROCHE DIFFÉRENTE**
- ⏱️ Temps : 1.4723s (très rapide)
- ❌ Exactitude haplotypes : 9/120 (7.5% erreurs)
- 💡 **Verdict:** Modèle mathématique différent, moins fiable
