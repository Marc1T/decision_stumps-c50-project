"""
Module Decision Stump.

Ce module implémente les Decision Stumps (arbres de profondeur 1) avec
différents critères d'impureté (Gini, Entropie, Erreur de classification).
"""

from .stump import DecisionStump
from .criteria import (
    gini_impurity,
    entropy, 
    classification_error,
    compute_impurity,
    information_gain,
    gini
)

__all__ = [
    'DecisionStump',
    'gini_impurity',
    'entropy',
    'classification_error', 
    'compute_impurity',
    'information_gain',
    'gini'
]