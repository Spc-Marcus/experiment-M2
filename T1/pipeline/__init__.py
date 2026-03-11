# T1/pipeline — modules internes du pipeline d'expérimentation T1.
#
# Carte des modules
# -----------------
# config.py    — lecture et construction de la configuration clé=valeur
# discovery.py — découverte et résolution des solveurs / heuristiques
# metrics.py   — calcul objectif / surface / densité / écart
# io.py        — ajout de ligne CSV thread-safe + écriture du journal JSON
# executor.py  — exécution d'un solveur exact ou d'une heuristique
# planner.py   — construction du plan d'exécution plat (mode dry-run)
# runner.py    — orchestration du pipeline complet + quick_check
