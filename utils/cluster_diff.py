"""Outils pour comparer deux partitions (clusters).

Ce module fournit `count_different_clusters(expected, found)` qui retourne
le nombre de clusters différents entre les deux partitions en comparant
les clusters comme des ensembles (l'ordre n'a pas d'importance).
"""
from typing import List, Iterable, Set, FrozenSet


def _clusters_to_set(clusters: List[Iterable[int]]) -> Set[FrozenSet[int]]:
    """Convertit une liste de clusters (itérables) en un set de frozensets."""
    return {frozenset(cluster) for cluster in clusters}


def count_different_clusters(expected_clusters: List[List[int]], found_clusters: List[List[int]]) -> int:
    """Retourne le nombre de clusters différents entre expected et found.

    La comparaison se fait par ensemble : on retourne la taille de la
    différence symétrique entre les deux ensembles de clusters.
    """
    expected_set = _clusters_to_set(expected_clusters)
    found_set = _clusters_to_set(found_clusters)
    return len(expected_set.symmetric_difference(found_set))
