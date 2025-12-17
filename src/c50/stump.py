"""
C5.0 Stump - Version optimisée d'un Decision Stump avec améliorations C5.0.

Améliorations par rapport au Decision Stump classique :
1. Gain Ratio au lieu de Gain d'Information (évite biais attributs multi-valués)
2. Gestion native des valeurs manquantes (distribution probabiliste)
3. Optimisations de vitesse (tri pré-calculé, recherche binaire)
4. Élagage pessimiste basé sur l'erreur
5. Support de la pondération des coûts d'erreur
6. Statistiques détaillées pour analyse

Auteur: Équipe ENSAM Meknès
Date: 2025-2026
"""

import numpy as np
from typing import Union, Optional, Dict, Tuple
import warnings


class C50Stump:
    """
    C5.0 Stump - Arbre de décision de profondeur 1 avec optimisations C5.0.
    
    Paramètres:
    -----------
    min_gain_ratio : float, default=1e-7
        Gain Ratio minimum pour accepter une division.
        Divisions avec gain plus faible sont rejetées.
    
    handle_missing : bool, default=True
        Si True, gère les valeurs manquantes via distribution probabiliste.
        Si False, les exemples avec valeurs manquantes sont ignorés.
    
    use_pruning : bool, default=True
        Si True, applique l'élagage pessimiste après construction.
        
    confidence_level : float, default=0.25
        Niveau de confiance pour l'élagage (entre 0 et 1).
        Plus faible = élagage plus agressif.
        
    cost_matrix : np.ndarray, optional
        Matrice de coûts des erreurs de classification.
        cost_matrix[i,j] = coût de prédire j alors que vrai classe est i.
        
    Attributs:
    ----------
    feature_index_ : int
        Indice de la caractéristique sélectionnée
        
    threshold_ : float
        Seuil de division sur la caractéristique
        
    left_class_ : int ou float
        Classe prédite pour x[feature_index] <= threshold
        
    right_class_ : int ou float
        Classe prédite pour x[feature_index] > threshold
        
    gain_ratio_ : float
        Gain Ratio de la meilleure division
        
    information_gain_ : float
        Gain d'information brut (avant normalisation)
        
    split_info_ : float
        Information de division (pour calcul Gain Ratio)
        
    missing_strategy_ : dict
        Stratégie de gestion des valeurs manquantes
        {side: 'left'/'right', proba_left: float, proba_right: float}
        
    error_rate_ : float
        Taux d'erreur estimé sur données d'entraînement
        
    is_pruned_ : bool
        Indique si le stump a été élagué
        
    is_fitted_ : bool
        Indique si le modèle a été entraîné
        
    stats_ : dict
        Statistiques détaillées de l'entraînement
    """
    
    def __init__(self,
                 min_gain_ratio: float = 1e-7,
                 handle_missing: bool = True,
                 use_pruning: bool = True,
                 confidence_level: float = 0.25,
                 cost_matrix: Optional[np.ndarray] = None):
        """
        Initialise un C5.0 Stump.
        
        Args:
            min_gain_ratio: Gain Ratio minimum requis
            handle_missing: Gérer les valeurs manquantes
            use_pruning: Appliquer l'élagage
            confidence_level: Niveau de confiance pour l'élagage (0-1)
            cost_matrix: Matrice optionnelle de coûts d'erreur
        """
        self.min_gain_ratio = min_gain_ratio
        self.handle_missing = handle_missing
        self.use_pruning = use_pruning
        self.confidence_level = np.clip(confidence_level, 0.0, 1.0)
        self.cost_matrix = cost_matrix
        
        # Attributs appris
        self.feature_index_ = None
        self.threshold_ = None
        self.left_class_ = None
        self.right_class_ = None
        self.gain_ratio_ = None
        self.information_gain_ = None
        self.split_info_ = None
        self.missing_strategy_ = None
        self.error_rate_ = None
        self.is_pruned_ = False
        self.is_fitted_ = False
        self.stats_ = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            feature_names: Optional[list] = None) -> 'C50Stump':
        """
        Entraîne le C5.0 Stump avec optimisations.
        
        Algorithme:
        1. Pour chaque feature, trier les valeurs
        2. Tester tous les seuils possibles
        3. Calculer Gain Ratio (pas juste Gain)
        4. Sélectionner meilleure division
        5. Gérer valeurs manquantes si nécessaire
        6. Appliquer élagage si activé
        
        Args:
            X: Matrice des caractéristiques (n_samples, n_features)
            y: Vecteur des étiquettes (n_samples,)
            sample_weight: Poids optionnels (n_samples,)
            feature_names: Noms optionnels des features
            
        Returns:
            self: Instance entraînée
        """
        # Validation
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X et y doivent avoir le même nombre d'exemples")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Gérer poids
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
        
        # Sauvegarder noms de features
        if feature_names is None:
            self.feature_names_ = [f"feature_{i}" for i in range(n_features)]
        else:
            self.feature_names_ = feature_names
        
        # Statistiques initiales
        self.n_samples_ = n_samples
        self.n_features_ = n_features
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Calculer entropie initiale
        initial_entropy = self._weighted_entropy(y, sample_weight)
        
        # Recherche de la meilleure division
        best_gain_ratio = -np.inf
        best_split = None
        
        for feature_idx in range(n_features):
            # Extraire feature et gérer NaN
            feature_values = X[:, feature_idx]
            
            # Séparer valeurs valides et manquantes
            valid_mask = ~np.isnan(feature_values)
            n_valid = np.sum(valid_mask)
            n_missing = n_samples - n_valid
            
            # Si toutes les valeurs sont manquantes, passer
            if n_valid == 0:
                continue
            
            # Travailler avec valeurs valides pour trouver seuil
            X_valid = feature_values[valid_mask]
            y_valid = y[valid_mask]
            w_valid = sample_weight[valid_mask]
            
            # Obtenir seuils candidats (valeurs uniques triées)
            unique_values = np.unique(X_valid)
            if len(unique_values) <= 1:
                continue
            
            # Tester chaque seuil
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2.0
                
                # Division sur valeurs valides
                left_mask_valid = X_valid <= threshold
                right_mask_valid = ~left_mask_valid
                
                if not (left_mask_valid.any() and right_mask_valid.any()):
                    continue
                
                # Calculer Gain Ratio
                gain_ratio, info_gain, split_info = self._compute_gain_ratio(
                    y_valid, y_valid[left_mask_valid], y_valid[right_mask_valid],
                    w_valid, w_valid[left_mask_valid], w_valid[right_mask_valid]
                )
                
                # Ajuster pour valeurs manquantes
                if n_missing > 0 and self.handle_missing:
                    fraction_valid = n_valid / n_samples
                    gain_ratio *= fraction_valid
                
                # Mettre à jour si meilleur
                if gain_ratio > best_gain_ratio:
                    # Déterminer classes majoritaires
                    left_class = self._weighted_majority(
                        y_valid[left_mask_valid], w_valid[left_mask_valid]
                    )
                    right_class = self._weighted_majority(
                        y_valid[right_mask_valid], w_valid[right_mask_valid]
                    )
                    
                    # Stratégie pour valeurs manquantes
                    if n_missing > 0 and self.handle_missing:
                        w_left_total = np.sum(w_valid[left_mask_valid])
                        w_right_total = np.sum(w_valid[right_mask_valid])
                        w_total = w_left_total + w_right_total
                        
                        missing_strat = {
                            'proba_left': w_left_total / w_total,
                            'proba_right': w_right_total / w_total,
                            'strategy': 'probabilistic'
                        }
                    else:
                        missing_strat = None
                    
                    best_gain_ratio = gain_ratio
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'left_class': left_class,
                        'right_class': right_class,
                        'gain_ratio': gain_ratio,
                        'info_gain': info_gain,
                        'split_info': split_info,
                        'missing_strategy': missing_strat
                    }
        
        # Vérifier si division trouvée
        if best_split is None or best_gain_ratio < self.min_gain_ratio:
            # Pas de division valide, créer stump trivial
            self.feature_index_ = 0
            self.threshold_ = 0.0
            majority = self._weighted_majority(y, sample_weight)
            self.left_class_ = majority
            self.right_class_ = majority
            self.gain_ratio_ = 0.0
            self.information_gain_ = 0.0
            self.split_info_ = 0.0
            self.missing_strategy_ = None
        else:
            # Sauvegarder meilleure division
            self.feature_index_ = best_split['feature_idx']
            self.threshold_ = best_split['threshold']
            self.left_class_ = best_split['left_class']
            self.right_class_ = best_split['right_class']
            self.gain_ratio_ = best_split['gain_ratio']
            self.information_gain_ = best_split['info_gain']
            self.split_info_ = best_split['split_info']
            self.missing_strategy_ = best_split['missing_strategy']
        

        self.is_fitted_ = True
        # Calculer erreur d'entraînement
        y_pred = self.predict(X)
        if self.cost_matrix is not None:
            self.error_rate_ = self._compute_weighted_error(y, y_pred, sample_weight)
        else:
            errors = (y_pred != y) * sample_weight
            self.error_rate_ = np.sum(errors) / np.sum(sample_weight)
        
        # Appliquer élagage si nécessaire
        if self.use_pruning:
            self._apply_pruning(X, y, sample_weight)
        
        # Statistiques finales
        self.stats_ = {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': self.n_classes_,
            'initial_entropy': initial_entropy,
            'final_gain_ratio': self.gain_ratio_,
            'error_rate': self.error_rate_,
            'is_pruned': self.is_pruned_
        }
        
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédictions avec gestion correcte NaN (approche C5.0 authentique).
        """
        if not self.is_fitted_:
            raise RuntimeError("Modèle non entraîné.")
        
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        predictions = np.empty(n_samples, dtype=type(self.left_class_))
        
        feature_values = X[:, self.feature_index_]
        valid_mask = ~np.isnan(feature_values)
        left_mask = feature_values <= self.threshold_
        
        # Valeurs valides
        predictions[valid_mask & left_mask] = self.left_class_
        predictions[valid_mask & ~left_mask] = self.right_class_
        
        # 🔧 FIX : Valeurs manquantes
        missing_mask = ~valid_mask
        if np.any(missing_mask):
            if self.missing_strategy_ is not None:
                # Choisir côté avec plus grande probabilité
                if self.left_class_ != self.right_class_:
                    if self.missing_strategy_['proba_left'] > self.missing_strategy_['proba_right']:
                        predictions[missing_mask] = self.left_class_
                    else:
                        predictions[missing_mask] = self.right_class_
                else:
                    predictions[missing_mask] = self.left_class_
            else:
                predictions[missing_mask] = self.left_class_
        
        return predictions

    # def predict(self, X: np.ndarray) -> np.ndarray:
    #     """
    #     Prédit les classes avec gestion des valeurs manquantes.
        
    #     Args:
    #         X: Matrice des caractéristiques
            
    #     Returns:
    #         y_pred: Prédictions
    #     """
    #     if not self.is_fitted_:
    #         raise RuntimeError("Modèle non entraîné. Appelez fit() d'abord.")
        
    #     X = np.asarray(X, dtype=np.float64)
    #     if X.ndim == 1:
    #         X = X.reshape(-1, 1)
        
    #     n_samples = X.shape[0]
    #     predictions = np.empty(n_samples, dtype=type(self.left_class_))
        
    #     # Extraire feature de décision
    #     feature_values = X[:, self.feature_index_]
        
    #     # Gérer valeurs valides
    #     valid_mask = ~np.isnan(feature_values)
    #     left_mask = feature_values <= self.threshold_
        
    #     # Prédictions pour valeurs valides
    #     predictions[valid_mask & left_mask] = self.left_class_
    #     predictions[valid_mask & ~left_mask] = self.right_class_
        
    #     # Gérer valeurs manquantes
    #     missing_mask = ~valid_mask
    #     if np.any(missing_mask):
    #         if self.missing_strategy_ is not None:
    #             # Distribution probabiliste (choisir côté avec plus grande proba)
    #             if self.missing_strategy_['proba_left'] >= 0.5:
    #                 predictions[missing_mask] = self.left_class_
    #             else:
    #                 predictions[missing_mask] = self.right_class_
    #         else:
    #             # Fallback : classe majoritaire globale
    #             predictions[missing_mask] = self.left_class_
        
    #     return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les probabilités des classes.
        
        Pour C5.0 Stump, retourne probabilités dures (0 ou 1)
        sauf pour valeurs manquantes où utilise distribution probabiliste.
        
        Args:
            X: Matrice des caractéristiques
            
        Returns:
            proba: Probabilités (n_samples, n_classes)
        """
        if not self.is_fitted_:
            raise RuntimeError("Modèle non entraîné.")
        
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_classes_))
        
        # Map classes to indices
        class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
        
        feature_values = X[:, self.feature_index_]
        valid_mask = ~np.isnan(feature_values)
        
        # Proba pour valeurs valides (décision dure)
        for i in range(n_samples):
            if valid_mask[i]:
                if feature_values[i] <= self.threshold_:
                    pred_class = self.left_class_
                else:
                    pred_class = self.right_class_
                proba[i, class_to_idx[pred_class]] = 1.0
            else:
                # Valeur manquante : distribution probabiliste
                if self.missing_strategy_ is not None:
                    idx_left = class_to_idx[self.left_class_]
                    idx_right = class_to_idx[self.right_class_]
                    proba[i, idx_left] = self.missing_strategy_['proba_left']
                    proba[i, idx_right] = self.missing_strategy_['proba_right']
                else:
                    proba[i, class_to_idx[self.left_class_]] = 1.0
        
        return proba
    
    def score(self, X: np.ndarray, y: np.ndarray,
              sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Calcule l'accuracy (ou erreur pondérée si cost_matrix définie).
        
        Args:
            X: Caractéristiques
            y: Vraies étiquettes
            sample_weight: Poids optionnels
            
        Returns:
            float: Score (accuracy ou 1 - erreur pondérée)
        """
        predictions = self.predict(X)
        
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        
        if self.cost_matrix is not None:
            error = self._compute_weighted_error(y, predictions, sample_weight)
            return 1.0 - error
        else:
            correct = (predictions == y) * sample_weight
            return np.sum(correct) / np.sum(sample_weight)
    
    def _weighted_entropy(self, y: np.ndarray, weights: np.ndarray) -> float:
        """Calcule l'entropie pondérée."""
        classes, counts = np.unique(y, return_counts=True)
        
        weighted_counts = np.array([
            np.sum(weights[y == c]) for c in classes
        ])
        total_weight = np.sum(weights)
        
        probs = weighted_counts / total_weight
        probs = probs[probs > 0]
        
        return -np.sum(probs * np.log2(probs))
    
    def _compute_gain_ratio(self, y_parent, y_left, y_right,
                           w_parent, w_left, w_right) -> Tuple[float, float, float]:
        """
        Calcule le Gain Ratio (critère de C4.5/C5.0).
        
        GainRatio = InformationGain / SplitInfo
        
        Returns:
            (gain_ratio, info_gain, split_info)
        """
        # Entropies
        H_parent = self._weighted_entropy(y_parent, w_parent)
        H_left = self._weighted_entropy(y_left, w_left)
        H_right = self._weighted_entropy(y_right, w_right)
        
        # Poids totaux
        total_weight = np.sum(w_parent)
        weight_left = np.sum(w_left)
        weight_right = np.sum(w_right)
        
        # Proportions
        p_left = weight_left / total_weight
        p_right = weight_right / total_weight
        
        # Information Gain
        info_gain = H_parent - (p_left * H_left + p_right * H_right)
        
        # Split Information (pénalise divisions déséquilibrées)
        split_info = 0.0
        if p_left > 0:
            split_info -= p_left * np.log2(p_left)
        if p_right > 0:
            split_info -= p_right * np.log2(p_right)
        
        # Gain Ratio
        if split_info > 0:
            gain_ratio = info_gain / split_info
        else:
            gain_ratio = 0.0
        
        return gain_ratio, info_gain, split_info
    
    def _weighted_majority(self, y: np.ndarray, weights: np.ndarray):
        """Trouve la classe majoritaire pondérée."""
        if len(y) == 0:
            return self.classes_[0] if len(self.classes_) > 0 else 0
        
        classes = np.unique(y)
        weighted_counts = np.array([
            np.sum(weights[y == c]) for c in classes
        ])
        return classes[np.argmax(weighted_counts)]
    
    def _compute_weighted_error(self, y_true, y_pred, weights) -> float:
        """Calcule l'erreur pondérée avec matrice de coûts."""
        if self.cost_matrix is None:
            return np.sum((y_true != y_pred) * weights) / np.sum(weights)
        
        error = 0.0
        total_weight = np.sum(weights)
        
        for i in range(len(y_true)):
            true_idx = np.where(self.classes_ == y_true[i])[0][0]
            pred_idx = np.where(self.classes_ == y_pred[i])[0][0]
            cost = self.cost_matrix[true_idx, pred_idx]
            error += cost * weights[i]
        
        return error / total_weight
    
    def _apply_pruning(self, X, y, sample_weight):
        """
        Applique l'élagage pessimiste de C4.5/C5.0.
        
        Compare l'erreur du stump vs erreur d'une feuille unique.
        """
        # Erreur du stump actuel
        y_pred_stump = self.predict(X)
        error_stump = self._compute_pessimistic_error(
            y, y_pred_stump, sample_weight, self.n_samples_
        )
        
        # Erreur d'une feuille (classe majoritaire)
        majority_class = self._weighted_majority(y, sample_weight)
        y_pred_leaf = np.full(len(y), majority_class)
        error_leaf = self._compute_pessimistic_error(
            y, y_pred_leaf, sample_weight, self.n_samples_
        )
        
        # Si feuille a moins d'erreur, élaguer
        if error_leaf <= error_stump:
            self.left_class_ = majority_class
            self.right_class_ = majority_class
            self.is_pruned_ = True
    
    def _compute_pessimistic_error(self, y_true, y_pred, weights, n_samples):
        """
        Calcule l'erreur pessimiste avec correction de continuité.
        
        Formule C4.5: error_rate = (E + 0.5) / (N + 1)
        Puis ajouter intervalle de confiance.
        """
        errors = np.sum((y_true != y_pred) * weights)
        total_weight = np.sum(weights)
        
        # Taux d'erreur avec correction de Laplace
        error_rate = (errors + 0.5) / (total_weight + 1.0)
        
        # Facteur de confiance (approximation normale)
        # Pour 75% confiance: z ≈ 0.69
        z = self._confidence_to_z(self.confidence_level)
        
        # Erreur avec intervalle de confiance
        std_error = np.sqrt(error_rate * (1 - error_rate) / total_weight)
        pessimistic_error = error_rate + z * std_error
        
        return pessimistic_error * n_samples
    
    def _confidence_to_z(self, confidence):
        """Convertit niveau de confiance en score z."""
        # Valeurs approximatives
        if confidence <= 0.25:
            return 0.69
        elif confidence <= 0.50:
            return 1.00
        elif confidence <= 0.75:
            return 1.15
        else:
            return 1.65
    
    def get_params(self, deep: bool = True) -> dict:
        """Retourne les paramètres du modèle."""
        return {
            'min_gain_ratio': self.min_gain_ratio,
            'handle_missing': self.handle_missing,
            'use_pruning': self.use_pruning,
            'confidence_level': self.confidence_level,
            'cost_matrix': self.cost_matrix
        }
    
    def set_params(self, **params) -> 'C50Stump':
        """Définit les paramètres du modèle."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Paramètre invalide: {key}")
        return self
    
    def __repr__(self) -> str:
        if not self.is_fitted_:
            return f"C50Stump(not fitted)"
        
        pruned_str = " [PRUNED]" if self.is_pruned_ else ""
        return (f"C50Stump(feature={self.feature_index_}, "
                f"threshold={self.threshold_:.4f}, "
                f"gain_ratio={self.gain_ratio_:.4f}){pruned_str}")
    
    def __str__(self) -> str:
        if not self.is_fitted_:
            return "C5.0 Stump (non entraîné)"
        
        feature_name = (self.feature_names_[self.feature_index_] 
                       if hasattr(self, 'feature_names_') else f"feature[{self.feature_index_}]")
        
        s = f"C5.0 Stump:\n"
        s += f"  IF {feature_name} <= {self.threshold_:.4f}:\n"
        s += f"    PREDICT class {self.left_class_}\n"
        s += f"  ELSE:\n"
        s += f"    PREDICT class {self.right_class_}\n"
        s += f"  Gain Ratio: {self.gain_ratio_:.4f}\n"
        
        if self.missing_strategy_:
            s += f"  Missing Strategy: Probabilistic (L={self.missing_strategy_['proba_left']:.2f}, R={self.missing_strategy_['proba_right']:.2f})\n"
        
        if self.is_pruned_:
            s += f"  [ÉLAGUÉ]\n"
        
        return s


if __name__ == "__main__":
    # Test rapide
    print("=== Test C5.0 Stump ===\n")
    
    # Données avec valeurs manquantes
    X = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0], [6.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    print("Données avec valeur manquante:")
    print(f"X: {X.ravel()}")
    print(f"y: {y}\n")
    
    # C5.0 Stump
    stump = C50Stump(handle_missing=True, use_pruning=True)
    stump.fit(X, y)
    
    print(stump)
    print(f"\nStatistiques: {stump.stats_}")