import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Chemin vers le CSV
csv_path = Path(__file__).parent.parent / "No-error" / "results.csv"

# Vérifier si le fichier existe
if not csv_path.exists():
    print(f"Erreur : Le fichier {csv_path} n'existe pas")
    exit(1)

# Charger les données
df = pd.read_csv(csv_path)

# Identifier les modèles disponibles
models = []
for col in df.columns:
    if '-Time' in col:
        model_name = col.replace('-Time', '')
        models.append(model_name)

print(f"Modèles détectés : {models}")
print(f"Nombre de résultats : {len(df)}")

# === CALCUL DES ERREURS ===
# Pour chaque modèle, calculer les erreurs
errors_data = {}

for model in models:
    # Différence haplotypes trouvés vs attendus
    haplotypes_diff = abs(df[f'{model}-Haplotypes'] - df['Haplotype'])
    haplotypes_errors = (haplotypes_diff > 0).sum()  # nombre de fois où il y a une erreur
    
    # Différence stripes trouvés vs attendus
    stripes_diff = abs(df[f'{model}-Stripe'] - df['Strip'])
    stripes_errors = (stripes_diff > 0).sum()
    
    # Total erreurs
    total_errors = haplotypes_errors + stripes_errors
    
    errors_data[model] = {
        'haplotypes_errors': haplotypes_errors,
        'stripes_errors': stripes_errors,
        'total_errors': total_errors,
        'haplotypes_diff_mean': haplotypes_diff.mean(),
        'stripes_diff_mean': stripes_diff.mean(),
    }

# === 1. TEMPS : ANALYSE DÉTAILLÉE ===
colors = sns.color_palette("husl", len(models))

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Analyse détaillée du temps d\'exécution', fontsize=14, fontweight='bold')

# 1.1 Temps moyen global
ax = axes[0, 0]
times = {model: df[f'{model}-Time'].mean() for model in models}
bars = ax.bar(models, times.values(), color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Temps moyen (secondes)', fontsize=11, fontweight='bold')
ax.set_title('Temps d\'exécution moyen par modèle', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=0)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 1.2 Temps par nombre d'haplotypes
ax = axes[0, 1]
haplotypes_unique = sorted(df['Haplotype'].unique())
for model, color in zip(models, colors):
    times_by_hap = [df[df['Haplotype'] == h][f'{model}-Time'].mean() for h in haplotypes_unique]
    ax.plot(haplotypes_unique, times_by_hap, marker='o', label=model, linewidth=2.5, 
            markersize=8, color=color)
ax.set_xlabel('Nombre d\'haplotypes', fontsize=11, fontweight='bold')
ax.set_ylabel('Temps moyen (secondes)', fontsize=11, fontweight='bold')
ax.set_title('Temps en fonction du nombre d\'haplotypes', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 1.3 Temps par nombre de stripes
ax = axes[1, 0]
stripes_unique = sorted(df['Strip'].unique())
for model, color in zip(models, colors):
    times_by_stripe = [df[df['Strip'] == s][f'{model}-Time'].mean() for s in stripes_unique]
    ax.plot(stripes_unique, times_by_stripe, marker='s', label=model, linewidth=2.5, 
            markersize=8, color=color)
ax.set_xlabel('Nombre de stripes', fontsize=11, fontweight='bold')
ax.set_ylabel('Temps moyen (secondes)', fontsize=11, fontweight='bold')
ax.set_title('Temps en fonction du nombre de stripes', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 1.4 Distribution des temps (boxplot)
ax = axes[1, 1]
time_data = [df[f'{model}-Time'].values for model in models]
bp = ax.boxplot(time_data, tick_labels=models, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Temps (secondes)', fontsize=11, fontweight='bold')
ax.set_title('Distribution des temps par modèle', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'temps.png', dpi=300, bbox_inches='tight')
print("\n✓ Graphique 'temps.png' généré (analyse détaillée)")
plt.close()

# === 2. ERREURS ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Qualité : Erreurs par modèle', fontsize=14, fontweight='bold')

# 2.1 Nombre d'erreurs (haplotypes vs stripes)
ax = axes[0]
x = np.arange(len(models))
width = 0.35

haplotypes_err = [errors_data[m]['haplotypes_errors'] for m in models]
stripes_err = [errors_data[m]['stripes_errors'] for m in models]

bars1 = ax.bar(x - width/2, haplotypes_err, width, label='Erreurs Haplotypes', 
               color='coral', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, stripes_err, width, label='Erreurs Stripes', 
               color='skyblue', alpha=0.8, edgecolor='black')

ax.set_ylabel('Nombre d\'erreurs', fontsize=11, fontweight='bold')
ax.set_title('Nombre de cas avec erreurs', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Ajouter les valeurs sur les barres
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 2.2 Différence moyenne
ax = axes[1]
haplotypes_diff_mean = [errors_data[m]['haplotypes_diff_mean'] for m in models]
stripes_diff_mean = [errors_data[m]['stripes_diff_mean'] for m in models]

bars1 = ax.bar(x - width/2, haplotypes_diff_mean, width, label='Diff. Haplotypes', 
               color='coral', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, stripes_diff_mean, width, label='Diff. Stripes', 
               color='skyblue', alpha=0.8, edgecolor='black')

ax.set_ylabel('Différence moyenne', fontsize=11, fontweight='bold')
ax.set_title('Différence moyenne trouvé vs attendu', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Ajouter les valeurs sur les barres
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'qualite.png', dpi=300, bbox_inches='tight')
print("✓ Graphique 'qualite.png' généré")
plt.close()

# === AFFICHER LES STATISTIQUES ===
print("\n" + "="*70)
print("VÉRIFICATION DES CALCULS D'ERREURS")
print("="*70)

for model in models:
    # Haplotypes
    hap_diff = abs(df[f'{model}-Haplotypes'] - df['Haplotype'])
    hap_errors = (hap_diff > 0).sum()
    
    # Stripes  
    stripe_diff = abs(df[f'{model}-Stripe'] - df['Strip'])
    stripe_errors = (stripe_diff > 0).sum()
    
    print(f"\n📊 {model.upper()}")
    print("-" * 70)
    print(f"  Haplotypes:")
    print(f"    - Erreurs détectées: {hap_errors}/{len(df)} ({hap_errors/len(df)*100:.1f}%)")
    if hap_errors > 0:
        indices = hap_diff[hap_diff > 0].index.tolist()
        print(f"    - Lignes avec erreurs: {indices}")
        for idx in indices[:5]:  # Afficher les 5 premières
            print(f"      Ligne {idx}: attendu={df.loc[idx, 'Haplotype']}, trouvé={df.loc[idx, f'{model}-Haplotypes']}, diff={hap_diff[idx]}")
    
    print(f"  Stripes:")
    print(f"    - Erreurs détectées: {stripe_errors}/{len(df)} ({stripe_errors/len(df)*100:.1f}%)")
    if stripe_errors > 0:
        indices = stripe_diff[stripe_diff > 0].index.tolist()
        print(f"    - Lignes avec erreurs: {indices}")
        for idx in indices[:5]:  # Afficher les 5 premières
            print(f"      Ligne {idx}: attendu={df.loc[idx, 'Strip']}, trouvé={df.loc[idx, f'{model}-Stripe']}, diff={stripe_diff[idx]}")

print("\n" + "="*70)

for model in models:
    print(f"\n📊 {model.upper()}")
    print("-" * 70)
    print(f"  Temps moyen : {times[model]:.4f}s")
    print(f"  Erreurs :")
    print(f"    - Haplotypes  : {errors_data[model]['haplotypes_errors']} cas / {len(df)} ({errors_data[model]['haplotypes_errors']/len(df)*100:.1f}%)")
    print(f"    - Stripes     : {errors_data[model]['stripes_errors']} cas / {len(df)} ({errors_data[model]['stripes_errors']/len(df)*100:.1f}%)")
    print(f"  Différence moyenne :")
    print(f"    - Haplotypes  : {errors_data[model]['haplotypes_diff_mean']:.2f}")
    print(f"    - Stripes     : {errors_data[model]['stripes_diff_mean']:.2f}")

print("\n" + "="*70)
print("Graphiques générés :")
print("  1. temps.png - Temps d'exécution moyen")
print("  2. qualite.png - Erreurs et différences")
print("="*70)
