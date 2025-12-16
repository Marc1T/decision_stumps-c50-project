"""
Decision Stump - Arbre de décision de profondeur 1.

Un Decision Stump est le plus simple des arbres de décision, composé d'un nœud
racine unique et de deux feuilles. Il effectue une seule décision basée sur une
caractéristique et un seuil.

Utilisé principalement comme classifieur faible dans les méthodes d'ensemble
(AdaBoost, Gradient Boosting).

Auteur: Équipe ENSAM Meknès
Date: 2024-2025
"""

import numpy as np
from typing import Union, Optional
from .criteria import information_gain, compute_impurity


class DecisionStump:
    """
    Decision Stump (Souche de Décision) - Arbre de profondeur 1.
    
    Paramètres:
    -----------
    criterion : str, default='gini'
        Critère de mesure de la qualité de la division:
        - 'gini' : Indice de Gini
        - 'entropy' : Entropie de Shannon
        - 'error' : Erreur de classification
        
    Attributs:
    ----------
    feature_index_ : int
        Indice de la caractéristique sélectionnée pour la division
        
    threshold_ : float
        Seuil de division sur la caractéristique sélectionnée
        
    left_class_ : int ou float
        Classe prédite pour les exemples où x[feature_index] <= threshold
        
    right_class_ : int ou float
        Classe prédite pour les exemples où x[feature_index] > threshold
        
    gain_ : float
        Gain d'information de la meilleure division
        
    is_fitted_ : bool
        Indique si le modèle a été entraîné
        
    Examples:
    ---------
    >>> from decision_trees_ml import DecisionStump
    >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    >>> y = np.array([0, 0, 1, 1])
    >>> stump = DecisionStump(criterion='gini')
    >>> stump.fit(X, y)
    >>> stump.predict([[2.5, 3.5]])
    array([0])
    """
    
    def __init__(self, criterion: str = 'gini'):
        """
        Initialise un Decision Stump.
        
        Args:
            criterion: Critère d'impureté ('gini', 'entropy', ou 'error')
        """
        if criterion not in ['gini', 'entropy', 'error']:
            raise ValueError(f"Critère '{criterion}' non reconnu. "
                           f"Utilisez 'gini', 'entropy', ou 'error'.")
        
        self.criterion = criterion
        
        # Attributs appris (initialisés à None)
        self.feature_index_ = None
        self.threshold_ = None
        self.left_class_ = None
        self.right_class_ = None
        self.gain_ = None
        self.is_fitted_ = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            sample_weight: Optional[np.ndarray] = None) -> 'DecisionStump':
        """
        Entraîne le Decision Stump en trouvant la meilleure division.
        
        L'algorithme teste toutes les caractéristiques et tous les seuils possibles
        pour trouver la division qui maximise le gain d'information.
        
        Complexité: O(d * n * log(n)) où d = nombre de features, n = nombre d'exemples
        
        Args:
            X: Matrice des caractéristiques, shape (n_samples, n_features)
            y: Vecteur des étiquettes, shape (n_samples,)
            sample_weight: Poids optionnels des exemples, shape (n_samples,)
            
        Returns:
            self: Instance entraînée
            
        Raises:
            ValueError: Si X et y n'ont pas le même nombre d'exemples
        """
        # Validation des entrées
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X et y doivent avoir le même nombre d'exemples. "
                           f"X: {X.shape[0]}, y: {y.shape[0]}")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        n_samples, n_features = X.shape
        
        # Gérer les poids
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.asarray(sample_weight)
            if sample_weight.shape[0] != n_samples:
                raise ValueError("sample_weight doit avoir la même longueur que X")
        
        # Initialiser la meilleure division
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_left_class = None
        best_right_class = None
        
        # Parcourir toutes les caractéristiques
        for feature_idx in range(n_features):
            # Extraire les valeurs de cette caractéristique
            feature_values = X[:, feature_idx]
            
            # Obtenir les valeurs uniques triées
            unique_values = np.unique(feature_values)
            
            # Si une seule valeur, pas de division possible
            if len(unique_values) <= 1:
                continue
            
            # Tester chaque seuil possible (point milieu entre valeurs consécutives)
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2.0
                
                # Créer les masques pour la division
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                # Vérifier qu'aucun côté n'est vide
                if not (left_mask.any() and right_mask.any()):
                    continue
                
                # Extraire les sous-ensembles
                y_left = y[left_mask]
                y_right = y[right_mask]
                w_left = sample_weight[left_mask]
                w_right = sample_weight[right_mask]
                
                # Calculer le gain
                gain = information_gain(
                    y, y_left, y_right,
                    criterion=self.criterion,
                    sample_weight_parent=sample_weight,
                    sample_weight_left=w_left,
                    sample_weight_right=w_right
                )
                
                # Mettre à jour si meilleur gain
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    
                    # Déterminer les classes majoritaires pondérées
                    best_left_class = self._weighted_majority_class(y_left, w_left)
                    best_right_class = self._weighted_majority_class(y_right, w_right)
        
        # Sauvegarder les meilleurs paramètres
        if best_feature is None:
            # Aucune division trouvée, utiliser la classe majoritaire globale
            self.feature_index_ = 0
            self.threshold_ = 0.0
            self.left_class_ = self._weighted_majority_class(y, sample_weight)
            self.right_class_ = self.left_class_
            self.gain_ = 0.0
        else:
            self.feature_index_ = best_feature
            self.threshold_ = best_threshold
            self.left_class_ = best_left_class
            self.right_class_ = best_right_class
            self.gain_ = best_gain
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les classes pour les nouveaux exemples.
        
        Complexité: O(n) où n = nombre d'exemples
        
        Args:
            X: Matrice des caractéristiques, shape (n_samples, n_features)
            
        Returns:
            y_pred: Prédictions, shape (n_samples,)
            
        Raises:
            RuntimeError: Si le modèle n'a pas été entraîné
        """
        if not self.is_fitted_:
            raise RuntimeError("Le modèle doit être entraîné avant de prédire. "
                             "Appelez fit() d'abord.")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        predictions = np.empty(n_samples, dtype=type(self.left_class_))
        
        # Appliquer la règle de décision
        feature_values = X[:, self.feature_index_]
        left_mask = feature_values <= self.threshold_
        
        predictions[left_mask] = self.left_class_
        predictions[~left_mask] = self.right_class_
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les probabilités des classes (version simplifiée).
        
        Pour un Decision Stump, les probabilités sont 0 ou 1 (décision dure).
        
        Args:
            X: Matrice des caractéristiques
            
        Returns:
            proba: Probabilités, shape (n_samples, n_classes)
        """
        if not self.is_fitted_:
            raise RuntimeError("Le modèle doit être entraîné avant de prédire.")
        
        predictions = self.predict(X)
        classes = np.unique([self.left_class_, self.right_class_]) # type: ignore
        n_classes = len(classes)
        n_samples = len(predictions)
        
        proba = np.zeros((n_samples, n_classes))
        for i, pred in enumerate(predictions):
            class_idx = np.where(classes == pred)[0][0]
            proba[i, class_idx] = 1.0
        
        return proba
    
    def score(self, X: np.ndarray, y: np.ndarray, 
              sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Calcule l'accuracy (précision) du modèle.
        
        Args:
            X: Matrice des caractéristiques
            y: Vraies étiquettes
            sample_weight: Poids optionnels
            
        Returns:
            float: Accuracy entre 0 et 1
        """
        predictions = self.predict(X)
        
        if sample_weight is None:
            accuracy = np.mean(predictions == y)
        else:
            sample_weight = np.asarray(sample_weight)
            correct = (predictions == y) * sample_weight
            accuracy = np.sum(correct) / np.sum(sample_weight)
        
        return accuracy
    
    def get_params(self, deep: bool = True) -> dict:
        """
        Retourne les paramètres du modèle (compatibilité sklearn).
        
        Args:
            deep: Paramètre ignoré (compatibilité sklearn)
            
        Returns:
            dict: Dictionnaire des paramètres
        """
        return {'criterion': self.criterion}
    
    def set_params(self, **params) -> 'DecisionStump':
        """
        Définit les paramètres du modèle (compatibilité sklearn).
        
        Args:
            **params: Paramètres à définir
            
        Returns:
            self: Instance modifiée
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Paramètre invalide: {key}")
        return self
    
    def _weighted_majority_class(self, y: np.ndarray, 
                                 weights: np.ndarray) -> Union[int, float]:
        """
        Trouve la classe majoritaire pondérée.
        
        Args:
            y: Étiquettes
            weights: Poids des exemples
            
        Returns:
            Classe majoritaire (ou 0 si y est vide)
        """
        if len(y) == 0:
            return 0  # Valeur par défaut pour array vide
        
        classes = np.unique(y)
        weighted_counts = np.array([
            np.sum(weights[y == c]) for c in classes
        ])
        return classes[np.argmax(weighted_counts)]
    
    def __repr__(self) -> str:
        """Représentation textuelle du Decision Stump."""
        if not self.is_fitted_:
            return f"DecisionStump(criterion='{self.criterion}', not fitted)"
        
        return (f"DecisionStump(criterion='{self.criterion}', "
                f"feature={self.feature_index_}, "
                f"threshold={self.threshold_:.4f}, "
                f"gain={self.gain_:.4f})")
    
    def __str__(self) -> str:
        """Version lisible du Decision Stump."""
        if not self.is_fitted_:
            return "Decision Stump (non entraîné)"
        
        return (f"Decision Stump:\n"
                f"  IF feature[{self.feature_index_}] <= {self.threshold_:.4f}:\n"
                f"    PREDICT class {self.left_class_}\n"
                f"  ELSE:\n"
                f"    PREDICT class {self.right_class_}\n"
                f"  Gain: {self.gain_:.4f}")


if __name__ == "__main__":
    # Test rapide
    print("=== Test Decision Stump ===\n")
    
    # Données d'exemple simples
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    print("Données d'entraînement:")
    print(f"X:\n{X}")
    print(f"y: {y}\n")
    
    # Entraîner avec différents critères
    for criterion in ['gini', 'entropy', 'error']:
        print(f"--- Critère: {criterion} ---")
        stump = DecisionStump(criterion=criterion)
        stump.fit(X, y)
        print(stump)
        
        # Prédictions
        X_test = np.array([[2.5, 3.5], [4.5, 5.5]])
        y_pred = stump.predict(X_test)
        print(f"Prédictions pour {X_test.tolist()}: {y_pred}")
        
        # Score
        accuracy = stump.score(X, y)
        print(f"Accuracy: {accuracy:.2%}\n")