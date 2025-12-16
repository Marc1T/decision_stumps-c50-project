"""
Module C5.0.

Ce module implémente les algorithmes de la famille C5.0 (successeur de C4.5):
- C50Stump: Version optimisée d'un Decision Stump avec améliorations C5.0
- C50Tree: Arbre de décision complet C5.0 (Phase 3)

Améliorations par rapport aux algorithmes classiques:
- Gain Ratio au lieu de Gain d'Information
- Gestion des valeurs manquantes
- Élagage pessimiste
- Optimisations de vitesse
- Support des coûts d'erreur
"""

from .stump import C50Stump

__all__ = [
    'C50Stump',
]