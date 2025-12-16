"""
Tests unitaires pour DecisionStump.

Ce module teste toutes les fonctionnalités du Decision Stump:
- Entraînement avec différents critères
- Prédictions
- Gestion des poids
- Cas limites
- Compatibilité sklearn

Auteur: Équipe ENSAM Meknès
Date: 2024-2025
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import sys
sys.path.insert(0, 'src')

from decision_stump.stump import DecisionStump # type: ignore
from decision_stump.criteria import gini_impurity, entropy, classification_error # type: ignore


class TestCriteria:
    """Tests des critères d'impureté."""
    
    def test_gini_pure(self):
        """Test Gini sur ensemble pur."""
        y = np.array([0, 0, 0, 0])
        assert gini_impurity(y) == 0.0
    
    def test_gini_uniform(self):
        """Test Gini sur distribution uniforme."""
        y = np.array([0, 0, 1, 1])
        assert_almost_equal(gini_impurity(y), 0.5)
    
    def test_gini_weighted(self):
        """Test Gini avec poids."""
        y = np.array([0, 0, 1])
        weights = np.array([1, 1, 2])  # Plus de poids sur classe 1
        gini = gini_impurity(y, weights)
        # Avec ces poids: p0=2/4=0.5, p1=2/4=0.5 → Gini = 1 - (0.5² + 0.5²) = 0.5
        assert 0 < gini <= 0.5
    
    def test_entropy_pure(self):
        """Test entropie sur ensemble pur."""
        y = np.array([1, 1, 1, 1])
        assert entropy(y) == 0.0
    
    def test_entropy_uniform(self):
        """Test entropie sur distribution uniforme."""
        y = np.array([0, 0, 1, 1])
        assert_almost_equal(entropy(y), 1.0)
    
    def test_entropy_three_classes(self):
        """Test entropie avec 3 classes équiprobables."""
        y = np.array([0, 1, 2])
        expected = np.log2(3)  # log₂(3) ≈ 1.585
        assert_almost_equal(entropy(y), expected, decimal=5)
    
    def test_classification_error(self):
        """Test erreur de classification."""
        y = np.array([0, 0, 0, 1])
        assert classification_error(y) == 0.25
    
    def test_empty_array(self):
        """Test sur array vide."""
        y = np.array([])
        assert gini_impurity(y) == 0.0
        assert entropy(y) == 0.0
        assert classification_error(y) == 0.0


class TestDecisionStumpBasic:
    """Tests de base du Decision Stump."""
    
    def setup_method(self):
        """Données de test communes."""
        # Dataset simple linéairement séparable
        self.X_simple = np.array([[1], [2], [3], [4], [5], [6]])
        self.y_simple = np.array([0, 0, 0, 1, 1, 1])
        
        # Dataset 2D
        self.X_2d = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y_2d = np.array([0, 0, 1, 1])
    
    def test_initialization(self):
        """Test initialisation."""
        stump = DecisionStump(criterion='gini')
        assert stump.criterion == 'gini'
        assert not stump.is_fitted_
    
    def test_invalid_criterion(self):
        """Test critère invalide."""
        with pytest.raises(ValueError):
            DecisionStump(criterion='invalid')
    
    def test_fit_simple(self):
        """Test entraînement simple."""
        stump = DecisionStump()
        stump.fit(self.X_simple, self.y_simple)
        
        assert stump.is_fitted_
        assert stump.feature_index_ is not None
        assert stump.threshold_ is not None
        assert stump.gain_ >= 0
    
    def test_perfect_split(self):
        """Test sur division parfaite."""
        stump = DecisionStump(criterion='gini')
        stump.fit(self.X_simple, self.y_simple)
        
        # Devrait trouver un seuil autour de 3.5
        assert 3.0 < stump.threshold_ < 4.0
        assert stump.left_class_ == 0
        assert stump.right_class_ == 1
        
        # Gain devrait être maximal (Gini initial = 0.5, Gini final = 0)
        assert_almost_equal(stump.gain_, 0.5, decimal=5)
    
    def test_predict_simple(self):
        """Test prédictions simples."""
        stump = DecisionStump()
        stump.fit(self.X_simple, self.y_simple)
        
        # Prédire valeurs extrêmes
        assert stump.predict([[1]]) == 0
        assert stump.predict([[6]]) == 1
        
        # Prédire autour du seuil
        X_test = np.array([[3], [4]])
        y_pred = stump.predict(X_test)
        assert len(y_pred) == 2
    
    def test_predict_before_fit(self):
        """Test prédiction sans entraînement."""
        stump = DecisionStump()
        with pytest.raises(RuntimeError):
            stump.predict([[1, 2]])
    
    def test_score(self):
        """Test calcul du score."""
        stump = DecisionStump()
        stump.fit(self.X_simple, self.y_simple)
        
        accuracy = stump.score(self.X_simple, self.y_simple)
        assert 0.0 <= accuracy <= 1.0
        assert accuracy == 1.0  # Division parfaite
    
    def test_different_criteria(self):
        """Test avec différents critères."""
        for criterion in ['gini', 'entropy', 'error']:
            stump = DecisionStump(criterion=criterion)
            stump.fit(self.X_simple, self.y_simple)
            
            assert stump.is_fitted_
            predictions = stump.predict(self.X_simple)
            assert len(predictions) == len(self.y_simple)


class TestDecisionStumpAdvanced:
    """Tests avancés du Decision Stump."""
    
    def test_multiclass(self):
        """Test sur problème multiclasse."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 1, 1, 2, 2])
        
        stump = DecisionStump()
        stump.fit(X, y)
        
        assert stump.is_fitted_
        predictions = stump.predict(X)
        # Devrait prédire au moins 2 classes différentes
        assert len(np.unique(predictions)) >= 2
    
    def test_sample_weights(self):
        """Test avec poids des échantillons."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        
        # Donner plus de poids aux premiers exemples
        weights = np.array([10, 10, 1, 1])
        
        stump = DecisionStump()
        stump.fit(X, y, sample_weight=weights)
        
        # Le seuil devrait favoriser la classe 0 (plus de poids)
        assert stump.is_fitted_
        
        # Score avec poids
        score = stump.score(X, y, sample_weight=weights)
        assert 0.0 <= score <= 1.0
    
    def test_single_class(self):
        """Test avec une seule classe."""
        X = np.array([[1], [2], [3]])
        y = np.array([0, 0, 0])
        
        stump = DecisionStump()
        stump.fit(X, y)
        
        # Devrait prédire toujours la même classe
        predictions = stump.predict(X)
        assert_array_equal(predictions, y)
    
    def test_single_feature_value(self):
        """Test avec une seule valeur de feature."""
        X = np.array([[1], [1], [1], [1]])
        y = np.array([0, 0, 1, 1])
        
        stump = DecisionStump()
        stump.fit(X, y)
        
        # Pas de division possible, devrait utiliser classe majoritaire
        assert stump.is_fitted_
        predictions = stump.predict(X)
        assert len(np.unique(predictions)) == 1
    
    def test_2d_features(self):
        """Test avec features 2D."""
        X = np.array([[1, 10], [2, 20], [3, 10], [4, 20]])
        y = np.array([0, 0, 1, 1])
        
        stump = DecisionStump()
        stump.fit(X, y)
        
        # Devrait sélectionner une des deux features
        assert stump.feature_index_ in [0, 1]
        
        predictions = stump.predict(X)
        assert len(predictions) == len(y)
    
    def test_predict_proba(self):
        """Test prédiction de probabilités."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        
        stump = DecisionStump()
        stump.fit(X, y)
        
        proba = stump.predict_proba(X)
        
        # Vérifier forme
        assert proba.shape[0] == len(X)
        assert proba.shape[1] == 2  # 2 classes
        
        # Vérifier que chaque ligne somme à 1
        assert_almost_equal(proba.sum(axis=1), np.ones(len(X)))
        
        # Vérifier valeurs 0 ou 1 (décision dure)
        assert np.all((proba == 0) | (proba == 1))


class TestSklearnCompatibility:
    """Tests de compatibilité avec sklearn."""
    
    def test_get_params(self):
        """Test get_params."""
        stump = DecisionStump(criterion='entropy')
        params = stump.get_params()
        assert 'criterion' in params
        assert params['criterion'] == 'entropy'
    
    def test_set_params(self):
        """Test set_params."""
        stump = DecisionStump(criterion='gini')
        stump.set_params(criterion='entropy')
        assert stump.criterion == 'entropy'
    
    def test_invalid_set_params(self):
        """Test set_params avec paramètre invalide."""
        stump = DecisionStump()
        with pytest.raises(ValueError):
            stump.set_params(invalid_param='value')
    
    def test_repr(self):
        """Test représentation textuelle."""
        stump = DecisionStump()
        repr_str = repr(stump)
        assert 'DecisionStump' in repr_str
        assert 'not fitted' in repr_str
        
        # Après entraînement
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        stump.fit(X, y)
        
        repr_str = repr(stump)
        assert 'feature=' in repr_str
        assert 'threshold=' in repr_str
    
    def test_str(self):
        """Test conversion en string."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        
        stump = DecisionStump()
        stump.fit(X, y)
        
        str_repr = str(stump)
        assert 'IF' in str_repr
        assert 'PREDICT' in str_repr


class TestEdgeCases:
    """Tests des cas limites."""
    
    def test_empty_input(self):
        """Test avec entrée vide."""
        X = np.array([]).reshape(0, 2)
        y = np.array([])
        
        stump = DecisionStump()
        # Ne devrait pas planter
        stump.fit(X, y)
    
    def test_mismatched_shapes(self):
        """Test avec formes incompatibles."""
        X = np.array([[1], [2], [3]])
        y = np.array([0, 1])  # Taille différente
        
        stump = DecisionStump()
        with pytest.raises(ValueError):
            stump.fit(X, y)
    
    def test_1d_input(self):
        """Test avec entrée 1D (devrait être reshapé)."""
        X = np.array([1, 2, 3, 4])
        y = np.array([0, 0, 1, 1])
        
        stump = DecisionStump()
        stump.fit(X, y)  # Devrait fonctionner après reshape automatique
        
        predictions = stump.predict([2.5])
        assert len(predictions) == 1
    
    def test_large_dataset(self):
        """Test sur dataset plus grand."""
        np.random.seed(42)
        X = np.random.randn(1000, 5)
        y = (X[:, 0] > 0).astype(int)
        
        stump = DecisionStump()
        stump.fit(X, y)
        
        accuracy = stump.score(X, y)
        assert accuracy > 0.5  # Devrait faire mieux que le hasard


class TestRealDatasets:
    """Tests sur données réelles (si disponibles)."""
    
    def test_iris_like(self):
        """Test sur données type Iris."""
        # Simuler données Iris (2 classes, 2 features)
        np.random.seed(42)
        
        # Classe 0: moyenne faible
        X0 = np.random.randn(50, 2) + np.array([1, 1])
        y0 = np.zeros(50)
        
        # Classe 1: moyenne élevée
        X1 = np.random.randn(50, 2) + np.array([4, 4])
        y1 = np.ones(50)
        
        X = np.vstack([X0, X1])
        y = np.hstack([y0, y1])
        
        stump = DecisionStump()
        stump.fit(X, y)
        
        accuracy = stump.score(X, y)
        assert accuracy > 0.7  # Devrait bien séparer


if __name__ == "__main__":
    # Lancer les tests
    pytest.main([__file__, '-v', '--tb=short'])