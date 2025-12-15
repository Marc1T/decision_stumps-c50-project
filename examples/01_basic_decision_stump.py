"""
Exemple 1 : Utilisation Basique d'un Decision Stump

Ce script démontre l'utilisation de base d'un Decision Stump:
- Entraînement sur données simples
- Prédictions
- Comparaison des critères (Gini, Entropie, Erreur)
- Visualisation des décisions

Auteur: Équipe ENSAM Meknès
Date: 2024-2025
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')

from decision_stump.stump import DecisionStump


def create_simple_dataset():
    """Crée un dataset simple linéairement séparable."""
    np.random.seed(42)
    
    # Classe 0: valeurs faibles
    X0 = np.random.randn(30, 2) * 0.5 + np.array([2, 2])
    y0 = np.zeros(30)
    
    # Classe 1: valeurs élevées
    X1 = np.random.randn(30, 2) * 0.5 + np.array([5, 5])
    y1 = np.ones(30)
    
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    return X, y


def plot_decision_boundary(stump, X, y, title="Decision Stump"):
    """Visualise la frontière de décision du stump."""
    plt.figure(figsize=(10, 6))
    
    # Tracer les points
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Classe 0', 
                alpha=0.6, edgecolors='k', s=50)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Classe 1', 
                alpha=0.6, edgecolors='k', s=50)
    
    # Tracer la frontière de décision
    if stump.is_fitted_:
        feature_idx = stump.feature_index_
        threshold = stump.threshold_
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        if feature_idx == 0:
            # Division verticale (sur x)
            plt.axvline(x=threshold, color='green', linestyle='--', 
                       linewidth=2, label=f'Seuil: x={threshold:.2f}')
            
            # Régions de décision
            plt.fill_betweenx([y_min, y_max], x_min, threshold, 
                             alpha=0.1, color='blue')
            plt.fill_betweenx([y_min, y_max], threshold, x_max, 
                             alpha=0.1, color='red')
        else:
            # Division horizontale (sur y)
            plt.axhline(y=threshold, color='green', linestyle='--', 
                       linewidth=2, label=f'Seuil: y={threshold:.2f}')
            
            # Régions de décision
            plt.fill_between([x_min, x_max], y_min, threshold, 
                           alpha=0.1, color='blue')
            plt.fill_between([x_min, x_max], threshold, y_max, 
                           alpha=0.1, color='red')
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
    
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def example_1_basic():
    """Exemple 1: Utilisation basique."""
    print("="*70)
    print("EXEMPLE 1: UTILISATION BASIQUE D'UN DECISION STUMP")
    print("="*70 + "\n")
    
    # Créer données
    X, y = create_simple_dataset()
    print(f"Dataset créé: {len(X)} exemples, {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}\n")
    
    # Entraîner un Decision Stump
    print("Entraînement du Decision Stump avec critère Gini...")
    stump = DecisionStump(criterion='gini')
    stump.fit(X, y)
    
    print("\n" + "─"*70)
    print("RÉSULTATS DE L'ENTRAÎNEMENT")
    print("─"*70)
    print(stump)
    
    # Prédictions
    print("\n" + "─"*70)
    print("PRÉDICTIONS")
    print("─"*70)
    y_pred = stump.predict(X)
    accuracy = stump.score(X, y)
    
    print(f"Accuracy sur données d'entraînement: {accuracy:.2%}")
    
    # Matrice de confusion simple
    from collections import Counter
    errors = np.sum(y_pred != y)
    print(f"Nombre d'erreurs: {errors}/{len(y)}")
    
    # Visualiser
    plot_decision_boundary(stump, X, y, 
                          title=f"Decision Stump (Gini) - Accuracy: {accuracy:.2%}")
    plt.savefig('decision_stump_basic.png', dpi=150, bbox_inches='tight')
    print("\n✅ Visualisation sauvegardée: decision_stump_basic.png")


def example_2_compare_criteria():
    """Exemple 2: Comparer les critères."""
    print("\n\n" + "="*70)
    print("EXEMPLE 2: COMPARAISON DES CRITÈRES")
    print("="*70 + "\n")
    
    # Créer données
    X, y = create_simple_dataset()
    
    criteria = ['gini', 'entropy', 'error']
    results = []
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, criterion in enumerate(criteria):
        print(f"\n📊 Critère: {criterion.upper()}")
        print("─"*50)
        
        stump = DecisionStump(criterion=criterion)
        stump.fit(X, y)
        
        accuracy = stump.score(X, y)
        
        print(f"  Feature sélectionnée: {stump.feature_index_}")
        print(f"  Seuil: {stump.threshold_:.4f}")
        print(f"  Gain: {stump.gain_:.4f}")
        print(f"  Accuracy: {accuracy:.2%}")
        
        results.append({
            'criterion': criterion,
            'accuracy': accuracy,
            'gain': stump.gain_,
            'threshold': stump.threshold_
        })
        
        # Visualiser dans subplot
        plt.sca(axes[idx])
        
        # Points
        axes[idx].scatter(X[y==0, 0], X[y==0, 1], c='blue', 
                         alpha=0.6, edgecolors='k', s=30)
        axes[idx].scatter(X[y==1, 0], X[y==1, 1], c='red', 
                         alpha=0.6, edgecolors='k', s=30)
        
        # Frontière
        if stump.feature_index_ == 0:
            axes[idx].axvline(x=stump.threshold_, color='green', 
                            linestyle='--', linewidth=2)
        else:
            axes[idx].axhline(y=stump.threshold_, color='green', 
                            linestyle='--', linewidth=2)
        
        axes[idx].set_title(f"{criterion.capitalize()} - Acc: {accuracy:.1%}",
                           fontweight='bold')
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('decision_stump_criteria_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Visualisation sauvegardée: decision_stump_criteria_comparison.png")
    
    # Tableau récapitulatif
    print("\n" + "="*70)
    print("TABLEAU RÉCAPITULATIF")
    print("="*70)
    print(f"{'Critère':<15} {'Accuracy':<12} {'Gain':<12} {'Seuil':<12}")
    print("─"*70)
    for r in results:
        print(f"{r['criterion']:<15} {r['accuracy']:<12.2%} "
              f"{r['gain']:<12.4f} {r['threshold']:<12.4f}")


def example_3_weighted_samples():
    """Exemple 3: Échantillons pondérés."""
    print("\n\n" + "="*70)
    print("EXEMPLE 3: ÉCHANTILLONS PONDÉRÉS")
    print("="*70 + "\n")
    
    # Créer données déséquilibrées
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    print("Dataset:")
    for i, (x, label) in enumerate(zip(X.ravel(), y)):
        print(f"  Exemple {i}: x={x:.1f}, y={label}")
    
    # Sans poids
    print("\n📊 SANS POIDS:")
    print("─"*50)
    stump_no_weight = DecisionStump()
    stump_no_weight.fit(X, y)
    print(f"  Seuil: {stump_no_weight.threshold_:.2f}")
    print(f"  Accuracy: {stump_no_weight.score(X, y):.2%}")
    
    # Avec poids (donner plus d'importance aux premiers exemples)
    weights = np.array([5, 5, 5, 5, 1, 1, 1, 1])
    print(f"\n📊 AVEC POIDS {weights.tolist()}:")
    print("─"*50)
    stump_weighted = DecisionStump()
    stump_weighted.fit(X, y, sample_weight=weights)
    print(f"  Seuil: {stump_weighted.threshold_:.2f}")
    print(f"  Accuracy (pondérée): {stump_weighted.score(X, y, sample_weight=weights):.2%}")
    
    print("\n💡 Le seuil a changé pour favoriser les exemples avec plus de poids!")


def example_4_test_predictions():
    """Exemple 4: Test de prédictions."""
    print("\n\n" + "="*70)
    print("EXEMPLE 4: TEST DE PRÉDICTIONS")
    print("="*70 + "\n")
    
    # Données d'entraînement
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    # Entraîner
    stump = DecisionStump()
    stump.fit(X_train, y_train)
    
    print("Modèle entraîné:")
    print(stump)
    
    # Nouvelles données de test
    X_test = np.array([
        [1.5, 2.5],
        [3.5, 4.5],
        [5.5, 6.5],
        [0.5, 1.5]
    ])
    
    print("\n📋 PRÉDICTIONS SUR NOUVELLES DONNÉES:")
    print("─"*70)
    y_pred = stump.predict(X_test)
    y_proba = stump.predict_proba(X_test)
    
    for i, (x, pred, proba) in enumerate(zip(X_test, y_pred, y_proba)):
        print(f"Exemple {i+1}: x={x} → Classe prédite: {pred}")
        print(f"            Probabilités: Classe 0={proba[0]:.2f}, Classe 1={proba[1]:.2f}")


def main():
    """Fonction principale."""
    print("\n" + "🌳"*35)
    print(" "*20 + "DECISION STUMP - EXEMPLES BASIQUES")
    print("🌳"*35 + "\n")
    
    try:
        example_1_basic()
        example_2_compare_criteria()
        example_3_weighted_samples()
        example_4_test_predictions()
        
        print("\n\n" + "="*70)
        print("✅ TOUS LES EXEMPLES ONT ÉTÉ EXÉCUTÉS AVEC SUCCÈS!")
        print("="*70)
        print("\n📊 Graphiques sauvegardés:")
        print("  - decision_stump_basic.png")
        print("  - decision_stump_criteria_comparison.png")
        print("\n💡 Consultez les fichiers pour voir les visualisations.")
        
        plt.show()
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()