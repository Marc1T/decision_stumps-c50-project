"""
Exemple 2 : Comparaison Decision Stump vs C5.0 Stump

Ce script démontre les améliorations du C5.0 Stump:
1. Gain Ratio vs Gain d'Information (correction du biais)
2. Gestion des valeurs manquantes
3. Élagage pessimiste
4. Matrice de coûts
5. Comparaison des performances

Auteur: Équipe ENSAM Meknès
Date: 2024-2025
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')

from decision_stump import DecisionStump
from c50.stump import C50Stump


def example_1_gain_ratio_bias():
    """Exemple 1: Correction du biais par Gain Ratio."""
    print("="*70)
    print("EXEMPLE 1: GAIN RATIO CORRIGE LE BIAIS")
    print("="*70 + "\n")
    
    # Dataset conçu pour montrer le biais du Gain d'Information
    # Feature 0: Beaucoup de valeurs distinctes (ID-like)
    # Feature 1: Peu de valeurs mais informative
    np.random.seed(42)
    
    n = 100
    # Feature 0: ID quasi-unique (biais pour Gain d'Info)
    X_feat0 = np.arange(n) + np.random.randn(n) * 0.1
    
    # Feature 1: Vraiment informative (2 valeurs)
    X_feat1 = np.repeat([0, 1], n//2)
    
    X = np.column_stack([X_feat0, X_feat1])
    y = np.repeat([0, 1], n//2)  # Parfaitement corrélé avec feature 1
    
    print("📊 Dataset artificiel:")
    print(f"  - Feature 0: {len(np.unique(X_feat0))} valeurs distinctes (quasi-ID)")
    print(f"  - Feature 1: {len(np.unique(X_feat1))} valeurs distinctes (informative)")
    print(f"  - Target parfaitement corrélé avec Feature 1\n")
    
    # Decision Stump classique (avec Entropie = Gain d'Information)
    print("🔵 DECISION STUMP CLASSIQUE (Gain d'Information):")
    print("─"*70)
    ds = DecisionStump(criterion='entropy')
    ds.fit(X, y)
    print(f"  Feature sélectionnée: {ds.feature_index_}")
    print(f"  Seuil: {ds.threshold_:.4f}")
    print(f"  Accuracy: {ds.score(X, y):.2%}\n")
    
    # C5.0 Stump (avec Gain Ratio)
    print("🟢 C5.0 STUMP (Gain Ratio):")
    print("─"*70)
    c50 = C50Stump()
    c50.fit(X, y)
    print(f"  Feature sélectionnée: {c50.feature_index_}")
    print(f"  Seuil: {c50.threshold_:.4f}")
    print(f"  Gain Ratio: {c50.gain_ratio_:.4f}")
    print(f"  Information Gain: {c50.information_gain_:.4f}")
    print(f"  Split Info: {c50.split_info_:.4f}")
    print(f"  Accuracy: {c50.score(X, y):.2%}\n")
    
    print("💡 Interprétation:")
    print("  - Gain d'Information favorise les attributs avec beaucoup de valeurs")
    print("  - Gain Ratio pénalise cette tendance via Split Info")
    print("  - C5.0 devrait choisir Feature 1 (plus simple et tout aussi précise)\n")


def example_2_missing_values():
    """Exemple 2: Gestion des valeurs manquantes."""
    print("\n" + "="*70)
    print("EXEMPLE 2: GESTION DES VALEURS MANQUANTES")
    print("="*70 + "\n")
    
    # Dataset avec valeurs manquantes
    X_train = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [np.nan, 30.0],  # Valeur manquante
        [4.0, np.nan],    # Valeur manquante
        [5.0, 50.0],
        [6.0, 60.0]
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    print("📊 Données d'entraînement avec valeurs manquantes (NaN):")
    print("X_train:")
    for i, (x, label) in enumerate(zip(X_train, y_train)):
        print(f"  Exemple {i}: {x} → classe {label}")
    print()
    
    # Decision Stump classique (ne gère pas NaN nativement)
    print("🔵 DECISION STUMP CLASSIQUE:")
    print("─"*70)
    try:
        ds = DecisionStump()
        ds.fit(X_train, y_train)
        print(f"  ✅ Entraîné (ignore probablement les NaN)")
        print(f"  Feature: {ds.feature_index_}, Seuil: {ds.threshold_:.2f}")
        
        # Test avec NaN
        X_test = np.array([[2.5, 25.0], [np.nan, np.nan]])
        y_pred = ds.predict(X_test)
        print(f"  Prédictions: {y_pred}")
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
    
    print()
    
    # C5.0 Stump (gère NaN nativement)
    print("🟢 C5.0 STUMP:")
    print("─"*70)
    c50 = C50Stump(handle_missing=True)
    c50.fit(X_train, y_train)
    
    print(f"  ✅ Entraîné avec gestion des valeurs manquantes")
    print(f"  Feature: {c50.feature_index_}, Seuil: {c50.threshold_:.2f}")
    
    if c50.missing_strategy_:
        print(f"  Stratégie NaN: {c50.missing_strategy_['strategy']}")
        print(f"    - Probabilité gauche: {c50.missing_strategy_['proba_left']:.2%}")
        print(f"    - Probabilité droite: {c50.missing_strategy_['proba_right']:.2%}")
    
    # Test avec NaN
    X_test = np.array([[2.5, 25.0], [np.nan, np.nan], [5.5, 55.0]])
    y_pred = c50.predict(X_test)
    y_proba = c50.predict_proba(X_test)
    
    print(f"\n  📋 Prédictions sur données de test:")
    for i, (x, pred, proba) in enumerate(zip(X_test, y_pred, y_proba)):
        print(f"    {i+1}. {x} → Classe {pred} (proba: {proba})")
    
    print("\n💡 C5.0 utilise une distribution probabiliste pour les valeurs manquantes!")


def example_3_pruning():
    """Exemple 3: Élagage pessimiste."""
    print("\n" + "="*70)
    print("EXEMPLE 3: ÉLAGAGE PESSIMISTE")
    print("="*70 + "\n")
    
    # Données avec un peu de bruit
    np.random.seed(42)
    n = 50
    X = np.random.randn(n, 2)
    # Target avec bruit
    y = (X[:, 0] > 0).astype(int)
    # Ajouter 20% de bruit
    noise_idx = np.random.choice(n, size=n//5, replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    
    print(f"📊 Dataset avec bruit (~20% d'erreurs aléatoires)")
    print(f"  Taille: {n} exemples")
    print(f"  Classes: {np.bincount(y)}\n")
    
    # Sans élagage
    print("🔵 SANS ÉLAGAGE:")
    print("─"*70)
    c50_no_prune = C50Stump(use_pruning=False)
    c50_no_prune.fit(X, y)
    print(f"  Accuracy: {c50_no_prune.score(X, y):.2%}")
    print(f"  Élagué: {c50_no_prune.is_pruned_}")
    print(f"  Classes: gauche={c50_no_prune.left_class_}, droite={c50_no_prune.right_class_}")
    
    print()
    
    # Avec élagage
    print("🟢 AVEC ÉLAGAGE (confidence=0.25):")
    print("─"*70)
    c50_prune = C50Stump(use_pruning=True, confidence_level=0.25)
    c50_prune.fit(X, y)
    print(f"  Accuracy: {c50_prune.score(X, y):.2%}")
    print(f"  Élagué: {c50_prune.is_pruned_}")
    print(f"  Classes: gauche={c50_prune.left_class_}, droite={c50_prune.right_class_}")
    
    if c50_prune.is_pruned_:
        print("\n💡 Le stump a été élagué en une feuille unique (classe majoritaire)")
        print("   car l'erreur estimée de la feuille est inférieure à celle du stump.")


def example_4_cost_matrix():
    """Exemple 4: Matrice de coûts."""
    print("\n" + "="*70)
    print("EXEMPLE 4: MATRICE DE COÛTS D'ERREUR")
    print("="*70 + "\n")
    
    # Dataset médical simulé
    # Classe 0: Sain, Classe 1: Malade
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    print("📊 Cas médical simulé:")
    print("  - Classe 0: Patient sain")
    print("  - Classe 1: Patient malade\n")
    
    # Sans matrice de coûts
    print("🔵 SANS MATRICE DE COÛTS (coûts égaux):")
    print("─"*70)
    c50_equal = C50Stump()
    c50_equal.fit(X, y)
    print(f"  Seuil: {c50_equal.threshold_:.2f}")
    print(f"  Accuracy: {c50_equal.score(X, y):.2%}\n")
    
    # Avec matrice de coûts asymétrique
    # Faux négatif (dire sain alors que malade) coûte très cher
    cost_matrix = np.array([
        [0, 1],   # Vrai sain → Faux malade: coût 1
        [10, 0]   # Vrai malade → Faux sain: coût 10 (DANGEREUX!)
    ])
    
    print("🟢 AVEC MATRICE DE COÛTS:")
    print("  Matrice:")
    print("           Prédit Sain  Prédit Malade")
    print("  Vrai Sain       0           1")
    print("  Vrai Malade    10           0")
    print()
    print("─"*70)
    c50_cost = C50Stump(cost_matrix=cost_matrix)
    c50_cost.fit(X, y)
    print(f"  Seuil: {c50_cost.threshold_:.2f}")
    print(f"  Score (1 - erreur pondérée): {c50_cost.score(X, y):.2%}")
    
    print("\n💡 La matrice de coûts influence la sélection du seuil")
    print("   pour minimiser les erreurs coûteuses (faux négatifs).")


def example_5_comparison_benchmark():
    """Exemple 5: Benchmark complet."""
    print("\n" + "="*70)
    print("EXEMPLE 5: BENCHMARK COMPLET")
    print("="*70 + "\n")
    
    # Générer dataset réaliste
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 5)
    y = (X[:, 0] + X[:, 2] > 0).astype(int)
    
    # Ajouter valeurs manquantes (10%)
    missing_mask = np.random.rand(n, 5) < 0.1
    X[missing_mask] = np.nan
    
    # Split train/test
    split = int(0.7 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"📊 Dataset:")
    print(f"  Train: {len(X_train)} exemples")
    print(f"  Test: {len(X_test)} exemples")
    print(f"  Features: 5")
    print(f"  Valeurs manquantes: ~10%\n")
    
    models = {
        'Decision Stump (Gini)': DecisionStump(criterion='gini'),
        'Decision Stump (Entropy)': DecisionStump(criterion='entropy'),
        'C5.0 Stump (basic)': C50Stump(handle_missing=False, use_pruning=False),
        'C5.0 Stump (full)': C50Stump(handle_missing=True, use_pruning=True)
    }
    
    print("="*70)
    print(f"{'Modèle':<30} {'Train Acc':<12} {'Test Acc':<12} {'Notes':<20}")
    print("="*70)
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            
            notes = ""
            if isinstance(model, C50Stump):
                if model.is_pruned_:
                    notes += "[ÉLAGUÉ] "
                if model.missing_strategy_:
                    notes += "[NaN OK]"
            
            print(f"{name:<30} {train_acc:<12.2%} {test_acc:<12.2%} {notes:<20}")
            
        except Exception as e:
            print(f"{name:<30} {'ERREUR':<12} {'ERREUR':<12} {str(e)[:20]}")
    
    print("="*70)
    
    print("\n💡 Observations:")
    print("  - C5.0 Stump (full) gère les valeurs manquantes nativement")
    print("  - L'élagage peut améliorer la généralisation sur test")
    print("  - Gain Ratio évite le surapprentissage sur features bruitées")


def main():
    """Fonction principale."""
    print("\n" + "🌳"*35)
    print(" "*15 + "DECISION STUMP vs C5.0 STUMP - COMPARAISON")
    print("🌳"*35 + "\n")
    
    try:
        example_1_gain_ratio_bias()
        example_2_missing_values()
        example_3_pruning()
        example_4_cost_matrix()
        example_5_comparison_benchmark()
        
        print("\n\n" + "="*70)
        print("✅ TOUS LES EXEMPLES TERMINÉS AVEC SUCCÈS!")
        print("="*70)
        
        print("\n🎯 RÉSUMÉ DES AMÉLIORATIONS C5.0 STUMP:")
        print("  1. ✅ Gain Ratio corrige biais du Gain d'Information")
        print("  2. ✅ Gestion native des valeurs manquantes")
        print("  3. ✅ Élagage pessimiste pour meilleure généralisation")
        print("  4. ✅ Support de matrice de coûts asymétrique")
        print("  5. ✅ Statistiques détaillées pour analyse")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()