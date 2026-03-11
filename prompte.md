Contexte
- Tu disposes uniquement de ce prompt et du code du dépôt racine. Tu n’as pas d’autre interlocuteur humain pour clarifier ; progresse sans demander d’avis, mais consigne strictement toutes les hypothèses prises dans les logs et le README.
- Tu dois implémenter UNIQUEMENT la partie “expérimentation” décrite dans T1_test_plan.md. Ne fais AUCUNE analyse ni visualisation.

But
- Produire un pipeline reproductible qui exécute des runs (solveurs exacts + heuristiques), collecte les métriques brutes et écrit CSV + logs.

Livrables à créer (sous le dépôt racine)
- Dossier T1/
  - T1/config.arg.example (exemple prêt à l’emploi)
  - T1/config.arg (si tu écris un fichier réel pour tests rapide)
  - T1/run_experiment.py — CLI/runner Python qui lit config.arg et exécute les expériences
  - T1/README.md — instruction d’usage, conventions d’import, hypothèses, fallback par défaut
  - T1/results/ (créé à l’exécution)
  - T1/results/logs/ (logs JSON par run)
  - T1/requirements.txt (seulement si tu utilises des dépendances externes non présentes)

Contraintes et autorisations
- Tu peux importer directement le code existant : model/final (solveurs exacts), model/heuristics (heuristiques), utils/create_matrix_V2.py.
- Tu ne dois exécuter que les solveurs/heuristiques dont les noms figurent dans le fichier de config.
- Pour les matrices synthétiques, utilise utils/create_matrix_V2.create_matrix(L, C, density, seed). Tu NE sauvegardes PAS la matrice binaire elle-même — tu sauvegardes la seed dans le CSV/log pour pouvoir la régénérer plus tard.
- Pour les synthétiques, exécute d’abord les solveurs exacts listés (pour obtenir best_known), puis les heuristiques (pour calculer gap).
- gamma correspond à la densité minimale attendue ; calcule error_rate = 1 - gamma et passe-la aux modèles.

Spécification du format de config (clé=valeur, listes par virgule)
- Exemples de clés à supporter :
  - instances_dir=Mat
  - instances=file1.csv,file2.csv      (optionnel — si absent, balaye instances_dir)
  - synthetic=true|false
  - synthetic_specs=L:200,C:200,density:0.1
  - seeds=1,2,3,4,5
  - gammas=0.9,0.95,0.99,1.0
  - solvers=NameOrModule:ClassName,...     (classes présentes sous model/final)
  - heuristics=func_name,module:func_name  (préférence : noms de fonction en model/heuristics)
  - timeout_exact=600
  - timeout_heuristic=150
  - output_dir=T1/results
  - parallel_jobs=1
  - dry_run=false
  - quick_check=false
- Dans README, documente comment tu veux que l’utilisateur écrive `solvers` et `heuristics` (classe simple vs module:path). Par défaut, accepte :
  - pour solvers : "ClassName" (cherche la classe dans tous les modules sous model/final) ou "module:ClassName"
  - pour heuristics : "func_name" (cherche dans model/heuristics) ou "module:func_name"

API et appels (contrat attendu)
- Classes exactes (sous model/final) implémentent BiclusterModelBase (voir base.py) :
  - constructeur : __init__(rows_data, cols_data, edges, error_rate)
  - méthodes : setParam(param, value), optimize()
  - propriétés/méthodes résultats : status (int), ObjVal (float), get_selected_rows(), get_selected_cols()
- Heuristiques (préférer fonctions sous model/heuristics) : signature souhaitée
  heuristic(input_matrix: numpy.ndarray, model_class, error_rate: float, time_limit: float, seed: int) -> (row_indices, col_indices, status_code)
  - MAIS la signature peut varier : effectue une introspection (inspect.signature) et appelle en adaptant les paramètres disponibles ; si la fonction nécessite moins d’args, fournis les minimums (toujours garantir reproductibilité en fixant numpy.random.seed et random.seed avec le seed).

Conversion matrice → entrées modèle
- Construis :
  - edges = [(i, j) for i in range(m) for j in range(n) if matrix[i, j] == 1]
  - rows_data = [(i, int(matrix[i, :].sum())) for i in range(m)]
  - cols_data = [(j, int(matrix[:, j].sum())) for j in range(n)]
- Instancie : model = ModelClass(rows_data, cols_data, edges, error_rate)
- Si ModelClass expose setParam, appelez setParam('TimeLimit', timeout) (essai raisonnable pour forcer limite de temps). Si échec, log et continue (les classes doivent gérer la timeout interne si possible).

Ordre d’exécution par instance (réel ou synthétique)
- Pour chaque instance (ou matrice synthétique générée) × chaque gamma × chaque seed :
  1. error_rate = 1 - gamma
  2. Si synthetic=true : génère matrix = create_matrix(L,C,density,seed)
     - enregistre seed dans la ligne CSV/log pour reproduction
  3. Lancer tous les solveurs listés dans `solvers` (exacts) avec timeout_exact. Conserve la meilleure solution observée (best_known = max objective parmi exacts — même si timeout renvoie une solution partielle).
  4. Lancer chaque heuristique listée avec timeout_heuristic et seed donné.
  5. Pour chaque run (exact ou heuristique), enregistrer une ligne CSV et un log JSON.

Calcul des métriques (une ligne CSV par run)
- En-tête attendue :
  instance_id,m,n,base_dens,gamma,solver,seed,heuristic,time,status,objective,area,density,gap
  - seed = seed utilisé pour synthétique, ou NA pour instance existante si pas applicable.
  - base_dens = ones / (m*n) de la matrice de départ.
  - objective = nombre réel de 1s dans la sous-matrice sélectionnée (compute à partir du choix de lignes/cols; si Model.ObjVal semble cohérent, tu peux l’utiliser).
  - area = (#rows_selected)*(#cols_selected)
  - density = objective/area (si area>0, sinon 0)
  - gap = 100*(best_known - objective)/best_known si best_known disponible et >0, sinon NA
  - status : normalise si possible en 'optimal' | 'timeout' | 'error' ; mais écris aussi le code brut retourné par le solver dans le log JSON.

Logs par run (T1/results/logs/)
- Un fichier JSON par run contenant : config locale, ligne CSV correspondante, full traceback en cas d’erreur, git short hash (si dispo, sinon 'no-git'), python version, import path des modules appelés, durée, seed, et toutes les hypothèses prises.
- Si un solver/propriétaire (ex. gurobipy) ne peut pas s’importer ou la licence est absente, SAUTE le run, enregistre status='error' et le message d’erreur détaillé ; NE fais pas crasher tout le pipeline.

Règles de robustesse / fallback (IMPORTANT — tu n’as pas d’utilisateur pour répondre)
- Si une entrée de config manque, applique ces choix par défaut mais documente clairement l’hypothèse dans README et logs :
  - instances_dir -> Mat
  - format des instances CSV : fichier CSV de 0/1, séparateur ',' sans en-tête (si différent, essaie de détecter et logue la détection)
  - si `solvers` vide et `synthetic=true` : au moins exécute un solveur exact si un modèle final standard existe (cherche dans model/final et choisis le 1er disponible), log la décision.
- Si signature d’une fonction heuristique diffère, adapte l’appel via introspection ; si impossible, marque run en error et continue.
- Pour la mapping du status int des solveurs exacts, essaie de détecter des codes Gurobi usuels (2=optimal, 9=time limit) et produire statut lisible ; si inconnu, laisse 'unknown' et log le code brut.
- Toujours continuer sur erreur d’un run ; ne stoppe le pipeline que sur erreur fatale (ex : faute d’accès totale au filesystem).

Fonctionnalités pratiques à implémenter
- dry_run : valide la config et affiche la liste des runs prévus sans exécution.
- quick_check : effectue 1 run minimal (ex. synthetic L=5,C=5,density=0.35,seed=42) pour valider le pipeline.
- parallel_jobs : parallélise les runs (simple concurrent.futures) en respectant limites et logs.
- record environment : git short hash (ou 'no-git'), python -V, et list of installed packages (pip freeze) si possible sans bloquer.

README (T1/README.md) — contenu minimal exigé
- Rôle du dossier T1 et rappel : "expérimentation uniquement, pas d’analyse".
- Format et exemples concrets pour config.arg (expliquer `solvers` et `heuristics` — comment écrire ClassName vs module:ClassName et function vs module:function).
- Exemple d’exécution (dry_run / quick_check / full run).
- Emplacement des résultats et logs (T1/results/, T1/results/logs/).
- Hypothèses par défaut prises si config absent / partiel et comment les modifier.
- Note sur la reproductibilité : seules les seeds sont sauvegardées pour régénérer les matrices synthétiques via utils/create_matrix_V2.

Tests et validation
- Avant tout run complet, exécute quick_check et valide que :
  - Le runner s’exécute sans lever d’exception non gérée.
  - Un CSV unique est produit avec l’en-tête demandé.
  - Au moins un log JSON est produit.
- Si quick_check échoue, enregistre l’erreur dans un fichier de diagnostic et arrête.

Sortie attendue immédiate de ton travail
- Le prompt est suffisant pour que l’agent autonome crée la structure T1/ et implémente run_experiment.py. Si pendant l’implémentation l’agent trouve une signature de fonction / classe incompatible avec ces conventions, l’agent doit :
  - Adapter l’appel par introspection si possible,
  - Sinon, skip le run, écrire une entrée status='error' avec message détaillé dans le log JSON, et continuer.
- L’agent doit consigner TOUTES les hypothèses dans T1/README.md et dans les logs.

Restrictions finales (ne pas outrepasser)
- Ne pas implémenter ou lancer d’analyses / visualisations (aucun matplotlib / plotting).
- N’invente pas résultats : si une valeur manque (best_known absent), inscris gap=NA.
- Tu peux t’inspirer d’autres dossiers du dépôt mais NE PAS importer directement des modules en dehors de model/final, model/heuristics et utils.
- Sauvegarde uniquement la seed pour les matrices synthétiques.

Concision sur la prise de décision automatique
- Si un choix dépend d’une info manquante, applique un fallback raisonnable (documenté) plutôt que d’arrêter et de poser une question.
- Exemple de fallback documenté : choisir la première classe trouvée dans model/final si aucune classe nommée n’est fournie.

Fin du prompt — agis maintenant et produis :
- Les fichiers listés dans "Livrables" et un commit local (optionnel) si tu implémentes.
- Un court rapport dans T1/README.md listant les principales hypothèses et la manière de reproduire quick_check.