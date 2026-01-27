# Pourquoi max_e_r fusionne les clusters

## Le Problème

**max_one trouve 7 haplotypes** ✅  
**max_e_r trouve 6 haplotypes** ⚠️ (fusion de 2 clusters)

## La Raison : Deux Objectifs Différents

### Max One
```
Objectif = Σ(i,j) M[i,j] × cell[i,j]
         = Compter les 1s réels dans la matrice
```

### Max E_R
```
Objectif = Σ(i,j) cell[i,j]
         = Compter l'aire du rectangle (indépendamment des vraies valeurs)
```

---

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
