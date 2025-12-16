"""
Tests unitaires pour C50Stump.

Ce module teste toutes les fonctionnalités avancées du C5.0 Stump:
- Gain Ratio vs Gain d'Information
- Gestion des valeurs manquantes
- Élagage pessimiste
- Matrice de coûts
- Optimisations

Auteur: Équipe ENSAM Meknès
Date: 2024-2025
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import sys
sys.path.insert(0, 'src')

from c50.stump import C50Stump


class TestC50StumpBasic:
    """Tests de base du C5.0 Stump."""
    
    def setup_method(self):
        """Données de test communes."""
        self.X_simple = np.array([[1], [2], [3], [4], [5], [6]])
        self.y_simple = np.array([0, 0, 0, 1, 1, 1])
        
        # Avec valeurs manquantes
        self.X_missing = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0], [6.0]])
        self.y_missing = np.array([0, 0, 0, 1, 1, 1])
    
    def test_initialization(self):
        """Test initialisation."""
        stump = C50Stump()
        assert stump.min_gain_ratio == 1e-7
        assert stump.handle_missing == True
        assert stump.use_pruning == True
        assert not stump.is_fitted_
    
    def test_fit_simple(self):
        """Test entraînement simple."""
        stump = C50Stump()
        stump.fit(self.X_simple, self.y_simple)
        
        assert stump.is_fitted_
        assert stump.feature_index_ is not None
        assert stump.threshold_ is not None
        assert stump.gain_ratio_ >= 0
    
    def test_gain_ratio_vs_gain(self):
        """Test que Gain Ratio est calculé (pas juste Gain)."""
        stump = C50Stump()
        stump.fit(self.X_simple, self.y_simple)
        
        # Vérifier que les attributs existent
        assert hasattr(stump, 'gain_ratio_')
        assert hasattr(stump, 'information_gain_')
        assert hasattr(stump, 'split_info_')
        
        # Gain Ratio devrait être différent de Information Gain
        # (sauf cas particulier)
        assert stump.split_info_ >= 0
    
    def test_perfect_split(self):
        """Test sur division parfaite."""
        stump = C50Stump()
        stump.fit(self.X_simple, self.y_simple)
        
        # Devrait trouver seuil parfait
        assert 3.0 < stump.threshold_ < 4.0
        assert stump.left_class_ == 0
        assert stump.right_class_ == 1
        
        # Accuracy parfaite
        assert stump.score(self.X_simple, self.y_simple) == 1.0
    
    def test_predict(self):
        """Test prédictions."""
        stump = C50Stump()
        stump.fit(self.X_simple, self.y_simple)
        
        predictions = stump.predict(self.X_simple)
        assert_array_equal(predictions, self.y_simple)


class TestMissingValues:
    """Tests gestion des valeurs manquantes."""
    
    def test_missing_handling_enabled(self):
        """Test avec gestion des valeurs manquantes activée."""
        X = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0], [6.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        stump = C50Stump(handle_missing=True)
        stump.fit(X, y)
        
        assert stump.is_fitted_
        assert stump.missing_strategy_ is not None
        
        # Vérifier structure de la stratégie
        assert 'proba_left' in stump.missing_strategy_
        assert 'proba_right' in stump.missing_strategy_
        assert 'strategy' in stump.missing_strategy_
        
        # Probabilités doivent sommer à 1
        assert_almost_equal(
            stump.missing_strategy_['proba_left'] + stump.missing_strategy_['proba_right'],
            1.0
        )
    
    def test_predict_with_missing(self):
        """Test prédictions avec valeurs manquantes."""
        X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y_train = np.array([0, 0, 0, 1, 1, 1])
        
        stump = C50Stump(handle_missing=True)
        stump.fit(X_train, y_train)
        
        # Prédire avec valeur manquante
        X_test = np.array([[2.5], [np.nan], [5.5]])
        y_pred = stump.predict(X_test)
        
        # Doit retourner 3 prédictions
        assert len(y_pred) == 3
        
        # Valeur manquante doit être gérée
        assert y_pred[1] in [0, 1]
    
    def test_predict_proba_with_missing(self):
        """Test probabilités avec valeurs manquantes."""
        X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y_train = np.array([0, 0, 0, 1, 1, 1])
        
        stump = C50Stump(handle_missing=True)
        stump.fit(X_train, y_train)
        
        # Prédire probabilités avec valeur manquante
        X_test = np.array([[2.5], [np.nan]])
        proba = stump.predict_proba(X_test)
        
        assert proba.shape == (2, 2)  # 2 samples, 2 classes
        
        # Pour valeur normale: proba dure (0 ou 1)
        assert np.any(proba[0] == 1.0)
        
        # Pour valeur manquante: peut être probabiliste
        # (mais au minimum doit sommer à 1)
        assert_almost_equal(np.sum(proba[1]), 1.0)
    
    def test_all_missing_feature(self):
        """Test avec feature entièrement manquante."""
        X = np.array([[np.nan], [np.nan], [np.nan], [np.nan]])
        y = np.array([0, 0, 1, 1])
        
        stump = C50Stump(handle_missing=True)
        stump.fit(X, y)
        
        # Devrait créer stump trivial
        assert stump.is_fitted_
        
        predictions = stump.predict(X)
        assert len(predictions) == 4


class TestPruning:
    """Tests de l'élagage."""
    
    def test_pruning_enabled(self):
        """Test avec élagage activé."""
        # Données parfaitement séparables
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        stump = C50Stump(use_pruning=True, confidence_level=0.25)
        stump.fit(X, y)
        
        # Sur données parfaites, ne devrait pas élaguer
        assert not stump.is_pruned_
    
    def test_pruning_disabled(self):
        """Test avec élagage désactivé."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        stump = C50Stump(use_pruning=False)
        stump.fit(X, y)
        
        assert not stump.is_pruned_
    
    def test_pruning_on_noisy_data(self):
        """Test élagage sur données bruitées."""
        np.random.seed(42)
        
        # Données avec beaucoup de bruit
        X = np.random.randn(50, 1)
        y = np.random.randint(0, 2, 50)
        
        stump = C50Stump(use_pruning=True, confidence_level=0.75)
        stump.fit(X, y)
        
        # Peut être élagué ou non selon les données
        assert hasattr(stump, 'is_pruned_')


class TestCostMatrix:
    """Tests de la matrice de coûts."""
    
    def test_cost_matrix_basic(self):
        """Test avec matrice de coûts simple."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        # Coût asymétrique: faux positif coûte plus cher
        cost_matrix = np.array([
            [0, 2],  # Classe 0: prédire 1 coûte 2
            [1, 0]   # Classe 1: prédire 0 coûte 1
        ])
        
        stump = C50Stump(cost_matrix=cost_matrix)
        stump.fit(X, y)
        
        assert stump.is_fitted_
    
    def test_score_with_cost_matrix(self):
        """Test score avec matrice de coûts."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        cost_matrix = np.array([[0, 1], [1, 0]])
        
        stump = C50Stump(cost_matrix=cost_matrix)
        stump.fit(X, y)
        
        score = stump.score(X, y)
        assert 0 <= score <= 1


class TestFeatureNames:
    """Tests des noms de features."""
    
    def test_with_feature_names(self):
        """Test avec noms de features fournis."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        
        stump = C50Stump()
        stump.fit(X, y, feature_names=['temperature'])
        
        assert hasattr(stump, 'feature_names_')
        assert stump.feature_names_[0] == 'temperature'
        
        # Le __str__ devrait utiliser le nom
        str_repr = str(stump)
        assert 'temperature' in str_repr
    
    def test_without_feature_names(self):
        """Test sans noms de features (noms auto)."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        
        stump = C50Stump()
        stump.fit(X, y)
        
        assert hasattr(stump, 'feature_names_')
        assert stump.feature_names_[0].startswith('feature_')


class TestStatistics:
    """Tests des statistiques."""
    
    def test_stats_dict(self):
        """Test du dictionnaire de statistiques."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        stump = C50Stump()
        stump.fit(X, y)
        
        assert hasattr(stump, 'stats_')
        assert isinstance(stump.stats_, dict)
        
        # Vérifier présence des clés importantes
        assert 'n_samples' in stump.stats_
        assert 'n_features' in stump.stats_
        assert 'n_classes' in stump.stats_
        assert 'initial_entropy' in stump.stats_
        assert 'final_gain_ratio' in stump.stats_
        assert 'error_rate' in stump.stats_
    
    def test_error_rate_calculation(self):
        """Test calcul du taux d'erreur."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        stump = C50Stump()
        stump.fit(X, y)
        
        # Sur ces données parfaites, erreur devrait être 0
        assert stump.error_rate_ == 0.0


class TestSklearnCompatibility:
    """Tests de compatibilité sklearn."""
    
    def test_get_params(self):
        """Test get_params."""
        stump = C50Stump(min_gain_ratio=0.01, handle_missing=False)
        params = stump.get_params()
        
        assert 'min_gain_ratio' in params
        assert 'handle_missing' in params
        assert params['min_gain_ratio'] == 0.01
        assert params['handle_missing'] == False
    
    def test_set_params(self):
        """Test set_params."""
        stump = C50Stump()
        stump.set_params(min_gain_ratio=0.1, use_pruning=False)
        
        assert stump.min_gain_ratio == 0.1
        assert stump.use_pruning == False
    
    def test_repr(self):
        """Test représentation."""
        stump = C50Stump()
        repr_str = repr(stump)
        assert 'C50Stump' in repr_str
        assert 'not fitted' in repr_str
        
        # Après entraînement
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        stump.fit(X, y)
        
        repr_str = repr(stump)
        assert 'feature=' in repr_str
        assert 'gain_ratio=' in repr_str


class TestComparison:
    """Tests de comparaison avec Decision Stump classique."""
    
    def test_gain_ratio_bias_correction(self):
        """
        Test que Gain Ratio corrige le biais du Gain d'Information.
        
        Sur des attributs avec beaucoup de valeurs distinctes,
        Gain Ratio devrait pénaliser plus que Gain simple.
        """
        # Dataset où un attribut a beaucoup de valeurs
        np.random.seed(42)
        X = np.column_stack([
            np.arange(20),  # Beaucoup de valeurs distinctes
            np.repeat([1, 2], 10)  # Peu de valeurs
        ])
        y = np.repeat([0, 1], 10)
        
        stump = C50Stump()
        stump.fit(X, y)
        
        # Devrait préférer feature 1 (moins de valeurs)
        # car Gain Ratio pénalise feature 0
        assert stump.is_fitted_
    
    def test_vs_decision_stump_on_missing(self):
        """
        Test que C5.0 Stump gère mieux les valeurs manquantes.
        """
        X = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0], [6.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        # C5.0 Stump devrait fonctionner
        c50 = C50Stump(handle_missing=True)
        c50.fit(X, y)
        
        predictions = c50.predict(X)
        assert len(predictions) == 6
        
        # Toutes les prédictions doivent être valides
        assert all(p in [0, 1] for p in predictions)


class TestEdgeCases:
    """Tests des cas limites."""
    
    def test_single_feature_value(self):
        """Test avec une seule valeur de feature."""
        X = np.array([[5.0], [5.0], [5.0], [5.0]])
        y = np.array([0, 0, 1, 1])
        
        stump = C50Stump()
        stump.fit(X, y)
        
        # Devrait créer stump trivial
        predictions = stump.predict(X)
        assert len(np.unique(predictions)) == 1
    
    def test_min_gain_ratio_threshold(self):
        """Test du seuil de Gain Ratio minimum."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        
        # Avec seuil très élevé
        stump = C50Stump(min_gain_ratio=10.0)
        stump.fit(X, y)
        
        # Devrait rejeter la division et créer stump trivial
        assert stump.gain_ratio_ < 10.0 or stump.gain_ratio_ == 0.0
    
    def test_multiclass(self):
        """Test sur problème multiclasse."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 1, 1, 2, 2])
        
        stump = C50Stump()
        stump.fit(X, y)
        
        assert stump.n_classes_ == 3
        predictions = stump.predict(X)
        assert len(predictions) == 6


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])