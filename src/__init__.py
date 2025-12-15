"""
Decision Trees ML - Implementation from scratch de Decision Stumps et C5.0

Ce package implémente:
- Decision Stumps (arbres de profondeur 1)
- Algorithme C5.0 (évolution de C4.5)
- Méthodes d'ensemble (AdaBoost)
- Utilitaires de visualisation et d'évaluation

Auteur: Équipe ENSAM Meknès
Année: 2025-2026
"""

__version__ = "1.0.0"
__author__ = "Équipe ENSAM Meknès"
__email__ = "m.nankouli@edu.umi.ac.ma"

# Imports des classes principales
from .decision_stump.stump import DecisionStump
from .decision_stump.criteria import (
    gini_impurity,
    entropy,
    classification_error,
    compute_impurity,
    information_gain
)

# TODO: Importer C50Tree quand implémenté
# from .c50.tree import C50Tree

# TODO: Importer AdaBoost quand implémenté  
# from .ensemble.adaboost import AdaBoostStump

__all__ = [
    # Classes principales
    'DecisionStump',
    # 'C50Tree',  # À décommenter quand implémenté
    # 'AdaBoostStump',  # À décommenter quand implémenté
    
    # Critères
    'gini_impurity',
    'entropy',
    'classification_error',
    'compute_impurity',
    'information_gain',
    
    # Métadonnées
    '__version__',
    '__author__',
]