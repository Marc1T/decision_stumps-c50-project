"""
Critères d'impureté pour Decision Stumps et arbres de décision.

Ce module implémente les trois principaux critères de mesure d'impureté :
- Gini Index (indice de Gini)
- Entropy (entropie de Shannon)
- Classification Error (erreur de classification)

Auteur: Équipe ENSAM Meknès
Date: 2025-2026
"""

import numpy as np
from typing import Union


def gini_impurity(y: np.ndarray, sample_weight: Union[np.ndarray, None] = None) -> float:
    """
    Calcule l'indice de Gini pour un ensemble d'étiquettes.
    
    L'indice de Gini mesure la probabilité qu'un élément pris au hasard
    soit mal classé si on lui attribue aléatoirement une classe selon
    la distribution des classes dans l'ensemble.
    
    Formule:
        Gini(S) = 1 - Σ(p_k²)
        où p_k est la proportion d'exemples de la classe k
    
    Args:
        y: Array des étiquettes de classe (n_samples,)
        sample_weight: Poids optionnels des échantillons (n_samples,)
        
    Returns:
        float: Indice de Gini entre 0 (pur) et 0.5 (impureté maximale pour 2 classes)
        
    Examples:
        >>> y = np.array([0, 0, 0, 1, 1, 1])
        >>> gini_impurity(y)
        0.5
        
        >>> y_pure = np.array([0, 0, 0, 0])
        >>> gini_impurity(y_pure)
        0.0
    """
    if len(y) == 0:
        return 0.0
    
    # Compter les occurrences de chaque classe avec pondération
    classes, counts = np.unique(y, return_counts=True)
    
    if sample_weight is not None:
        # Calculer les comptes pondérés
        weighted_counts = np.zeros(len(classes))
        for i, cls in enumerate(classes):
            mask = (y == cls)
            weighted_counts[i] = np.sum(sample_weight[mask])
        total_weight = np.sum(sample_weight)
    else:
        weighted_counts = counts.astype(float)
        total_weight = len(y)
    
    # Calculer les probabilités
    probabilities = weighted_counts / total_weight
    
    # Calculer Gini: 1 - Σ(p_k²)
    gini = 1.0 - np.sum(probabilities ** 2)
    
    return gini


def entropy(y: np.ndarray, sample_weight: Union[np.ndarray, None] = None) -> float:
    """
    Calcule l'entropie de Shannon pour un ensemble d'étiquettes.
    
    L'entropie mesure le degré d'incertitude ou de désordre dans un ensemble.
    Une entropie élevée signifie une grande incertitude.
    
    Formule:
        H(S) = -Σ(p_k * log₂(p_k))
        où p_k est la proportion d'exemples de la classe k
    
    Args:
        y: Array des étiquettes de classe (n_samples,)
        sample_weight: Poids optionnels des échantillons (n_samples,)
        
    Returns:
        float: Entropie en bits, entre 0 (pur) et log₂(n_classes) (impureté maximale)
        
    Examples:
        >>> y = np.array([0, 0, 0, 1, 1, 1])
        >>> entropy(y)
        1.0
        
        >>> y_pure = np.array([0, 0, 0, 0])
        >>> entropy(y_pure)
        0.0
    """
    if len(y) == 0:
        return 0.0
    
    # Compter les occurrences de chaque classe avec pondération
    classes, counts = np.unique(y, return_counts=True)
    
    if sample_weight is not None:
        weighted_counts = np.zeros(len(classes))
        for i, cls in enumerate(classes):
            mask = (y == cls)
            weighted_counts[i] = np.sum(sample_weight[mask])
        total_weight = np.sum(sample_weight)
    else:
        weighted_counts = counts.astype(float)
        total_weight = len(y)
    
    # Calculer les probabilités
    probabilities = weighted_counts / total_weight
    
    # Éliminer les probabilités nulles pour éviter log(0)
    probabilities = probabilities[probabilities > 0]
    
    # Calculer entropie: -Σ(p_k * log₂(p_k))
    ent = -np.sum(probabilities * np.log2(probabilities))
    
    return ent


def classification_error(y: np.ndarray, sample_weight: Union[np.ndarray, None] = None) -> float:
    """
    Calcule l'erreur de classification pour un ensemble d'étiquettes.
    
    L'erreur de classification est la proportion d'exemples qui ne
    appartiennent pas à la classe majoritaire.
    
    Formule:
        Error(S) = 1 - max(p_k)
        où p_k est la proportion d'exemples de la classe k
    
    Args:
        y: Array des étiquettes de classe (n_samples,)
        sample_weight: Poids optionnels des échantillons (n_samples,)
        
    Returns:
        float: Erreur entre 0 (pur) et (n_classes-1)/n_classes (impureté maximale)
        
    Examples:
        >>> y = np.array([0, 0, 0, 1, 1, 1])
        >>> classification_error(y)
        0.5
        
        >>> y = np.array([0, 0, 0, 1])
        >>> classification_error(y)
        0.25
    """
    if len(y) == 0:
        return 0.0
    
    # Compter les occurrences de chaque classe avec pondération
    classes, counts = np.unique(y, return_counts=True)
    
    if sample_weight is not None:
        weighted_counts = np.zeros(len(classes))
        for i, cls in enumerate(classes):
            mask = (y == cls)
            weighted_counts[i] = np.sum(sample_weight[mask])
        total_weight = np.sum(sample_weight)
    else:
        weighted_counts = counts.astype(float)
        total_weight = len(y)
    
    # Calculer les probabilités
    probabilities = weighted_counts / total_weight
    
    # Calculer erreur: 1 - max(p_k)
    error = 1.0 - np.max(probabilities)
    
    return error


def compute_impurity(y: np.ndarray, 
                    criterion: str = 'gini',
                    sample_weight: Union[np.ndarray, None] = None) -> float:
    """
    Calcule l'impureté selon le critère spécifié.
    
    Args:
        y: Array des étiquettes de classe
        criterion: 'gini', 'entropy', ou 'error'
        sample_weight: Poids optionnels des échantillons
        
    Returns:
        float: Valeur d'impureté
        
    Raises:
        ValueError: Si le critère n'est pas reconnu
        
    Examples:
        >>> y = np.array([0, 0, 1, 1])
        >>> compute_impurity(y, 'gini')
        0.5
        >>> compute_impurity(y, 'entropy')
        1.0
    """
    if criterion == 'gini':
        return gini_impurity(y, sample_weight)
    elif criterion == 'entropy':
        return entropy(y, sample_weight)
    elif criterion == 'error':
        return classification_error(y, sample_weight)
    else:
        raise ValueError(f"Critère inconnu: {criterion}. "
                        f"Utilisez 'gini', 'entropy', ou 'error'.")


def information_gain(y_parent: np.ndarray,
                    y_left: np.ndarray,
                    y_right: np.ndarray,
                    criterion: str = 'gini',
                    sample_weight_parent: Union[np.ndarray, None] = None,
                    sample_weight_left: Union[np.ndarray, None] = None,
                    sample_weight_right: Union[np.ndarray, None] = None) -> float:
    """
    Calcule le gain d'information (ou gain de Gini/erreur) d'une division.
    
    Le gain mesure la réduction d'impureté obtenue en divisant un nœud parent
    en deux nœuds enfants (gauche et droite).
    
    Formule:
        Gain = Impurity(parent) - [p_left * Impurity(left) + p_right * Impurity(right)]
        où p_left et p_right sont les proportions d'exemples dans chaque enfant
    
    Args:
        y_parent: Étiquettes du nœud parent
        y_left: Étiquettes du nœud gauche
        y_right: Étiquettes du nœud droit
        criterion: 'gini', 'entropy', ou 'error'
        sample_weight_parent: Poids du parent
        sample_weight_left: Poids du nœud gauche
        sample_weight_right: Poids du nœud droit
        
    Returns:
        float: Gain d'information (toujours >= 0)
        
    Examples:
        >>> y_parent = np.array([0, 0, 1, 1, 1, 1])
        >>> y_left = np.array([0, 0])
        >>> y_right = np.array([1, 1, 1, 1])
        >>> information_gain(y_parent, y_left, y_right, 'gini')
        0.4444...
    """
    n_parent = len(y_parent)
    n_left = len(y_left)
    n_right = len(y_right)
    
    if n_parent == 0 or n_left == 0 or n_right == 0:
        return 0.0
    
    # Calculer les poids totaux
    if sample_weight_parent is not None:
        total_weight = np.sum(sample_weight_parent)
        weight_left = np.sum(sample_weight_left)
        weight_right = np.sum(sample_weight_right)
    else:
        total_weight = n_parent
        weight_left = n_left
        weight_right = n_right
    
    # Calculer les proportions
    p_left = weight_left / total_weight
    p_right = weight_right / total_weight
    
    # Calculer les impuretés
    impurity_parent = compute_impurity(y_parent, criterion, sample_weight_parent)
    impurity_left = compute_impurity(y_left, criterion, sample_weight_left)
    impurity_right = compute_impurity(y_right, criterion, sample_weight_right)
    
    # Calculer le gain
    gain = impurity_parent - (p_left * impurity_left + p_right * impurity_right)
    
    return max(0.0, gain)  # S'assurer que le gain est positif


# Alias pour compatibilité avec sklearn
gini = gini_impurity


if __name__ == "__main__":
    # Tests rapides
    print("=== Tests des critères d'impureté ===\n")
    
    # Test 1: Distribution uniforme
    y1 = np.array([0, 0, 1, 1])
    print(f"Distribution uniforme [0,0,1,1]:")
    print(f"  Gini: {gini_impurity(y1):.4f}")
    print(f"  Entropy: {entropy(y1):.4f}")
    print(f"  Error: {classification_error(y1):.4f}\n")
    
    # Test 2: Ensemble pur
    y2 = np.array([0, 0, 0, 0])
    print(f"Ensemble pur [0,0,0,0]:")
    print(f"  Gini: {gini_impurity(y2):.4f}")
    print(f"  Entropy: {entropy(y2):.4f}")
    print(f"  Error: {classification_error(y2):.4f}\n")
    
    # Test 3: Gain d'information
    y_parent = np.array([0, 0, 0, 1, 1, 1])
    y_left = np.array([0, 0, 0])
    y_right = np.array([1, 1, 1])
    gain = information_gain(y_parent, y_left, y_right, 'gini')
    print(f"Gain pour division parfaite: {gain:.4f}")