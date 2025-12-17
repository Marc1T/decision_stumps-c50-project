# Decision Stump & C5.0 Stump - Implementation from Scratch 🌳

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-58%20passed-brightgreen.svg)]()
[![Code Quality](https://img.shields.io/badge/code%20quality-A-brightgreen.svg)]()

> **Implémentation complète from scratch de Decision Stumps et C5.0 Stumps**  
> Projet académique - ENSAM Meknès 2024-2025

---

## 📖 Présentation

Ce projet implémente deux algorithmes fondamentaux d'apprentissage automatique :

### 🔵 Decision Stump (Souche de Décision)
- Arbre de décision de **profondeur 1** (classifieur le plus simple)
- 3 critères d'impureté : **Gini**, **Entropie**, **Erreur de classification**
- Utilisé comme classifieur faible dans **AdaBoost** et **Gradient Boosting**
- Complexité : **O(dn log n)** entraînement, **O(1)** prédiction

### 🟢 C5.0 Stump (Version Optimisée)
- Version avancée avec optimisations de **C5.0** (successeur de C4.5)
- **Gain Ratio** (correction du biais du Gain d'Information)
- Gestion native des **valeurs manquantes** (distribution probabiliste)
- **Élagage pessimiste** pour meilleure généralisation
- Support de **matrices de coûts** asymétriques
- Statistiques détaillées pour analyse

---

## ✨ Fonctionnalités Principales

### Decision Stump
✅ 3 critères d'impureté (Gini, Entropie, Erreur)  
✅ Support des poids d'échantillons  
✅ Compatible scikit-learn  
✅ Ultra-rapide (< 1ms sur 1000 exemples)  
✅ Parfait pour ensembles (AdaBoost)  

### C5.0 Stump
✅ **Gain Ratio** (évite biais attributs multi-valués)  
✅ **Valeurs manquantes** (gestion probabiliste native)  
✅ **Élagage** (pessimistic error-based pruning)  
✅ **Coûts asymétriques** (matrice de coûts personnalisée)  
✅ **Statistiques** (entropie, gain, erreur, etc.)  
✅ Documentation détaillée  

---

## 🚀 Installation

### Prérequis
- Python 3.8+
- pip

### Installation en mode développement

```bash
# Cloner le projet
cd decision_stumps_c50_project

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate  # Windows

# Installer dépendances
pip install -r requirements.txt

# Installer en mode dev
python install_dev.py
```

### Vérification

```bash
# Test rapide
python quick_test_c50.py

# Tests unitaires
pytest tests/ -v

# Exemples
python examples/01_basic_decision_stump.py
python examples/02_c50_stump_comparison.py
```

---

## 💡 Utilisation Rapide

### Decision Stump

```python
from decision_stump import DecisionStump
import numpy as np

# Données
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])

# Créer et entraîner
stump = DecisionStump(criterion='gini')
stump.fit(X, y)

# Prédire
y_pred = stump.predict(X)
print(f"Accuracy: {stump.score(X, y):.2%}")

# Afficher
print(stump)
# Decision Stump:
#   IF feature[0] <= 3.5000:
#     PREDICT class 0
#   ELSE:
#     PREDICT class 1
#   Gain: 0.5000
```

### C5.0 Stump

```python
from c50 import C50Stump
import numpy as np

# Données avec valeurs manquantes
X = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0], [6.0]])
y = np.array([0, 0, 0, 1, 1, 1])

# Créer avec gestion NaN et élagage
stump = C50Stump(
    handle_missing=True,
    use_pruning=True,
    confidence_level=0.25
)

# Entraîner
stump.fit(X, y)

# Prédire (gère automatiquement les NaN)
y_pred = stump.predict(X)

# Statistiques
print(stump.stats_)
# {'n_samples': 6, 'n_features': 1, 'n_classes': 2,
#  'initial_entropy': 1.0, 'final_gain_ratio': 0.918,
#  'error_rate': 0.0, 'is_pruned': False}
```

## 🧪 Tests

Le projet inclut **58 tests unitaires** avec 100% de réussite.

```bash
# Tous les tests
pytest tests/ -v

# Avec coverage
pytest tests/ --cov=src --cov-report=html

# Tests spécifiques
pytest tests/test_decision_stump.py -v  # 32 tests
pytest tests/test_c50_stump.py -v       # 26 tests
```

### Couverture des Tests

- ✅ Critères d'impureté (Gini, Entropie, Erreur)
- ✅ Entraînement et prédiction
- ✅ Gestion des valeurs manquantes
- ✅ Élagage pessimiste
- ✅ Matrice de coûts
- ✅ Cas limites (données vides, une seule classe, etc.)
- ✅ Compatibilité sklearn
- ✅ Poids des échantillons

---

## 📝 Exemples

### Exemple 1 : Utilisation Basique
```bash
python examples/01_basic_decision_stump.py
```

Démontre :
- Entraînement sur données simples
- Comparaison des 3 critères (Gini, Entropie, Erreur)
- Échantillons pondérés
- Visualisations

### Exemple 2 : Comparaison Decision Stump vs C5.0 Stump
```bash
python examples/02_c50_stump_comparison.py
```

Démontre :
1. **Gain Ratio** corrige le biais du Gain d'Information
2. **Gestion des valeurs manquantes** (NaN)
3. **Élagage pessimiste**
4. **Matrice de coûts** asymétriques
5. **Benchmark complet** sur dataset réel

---

## 📚 Documentation

### Structure du Projet

```
decision_stumps_c50_project/
├── src/
│   ├── decision_stump/         # Module Decision Stump
│   │   ├── stump.py           # Classe principale
│   │   └── criteria.py        # Critères d'impureté
│   └── c50/                   # Module C5.0 Stump
│       ├── stump.py           # Classe principale
│       └── README_C50_STUMP.md # Doc détaillée
│
├── tests/                      # Tests unitaires (58 tests)
│   ├── test_decision_stump.py
│   └── test_c50_stump.py
│
├── examples/                   # Exemples d'utilisation
│   ├── 01_basic_decision_stump.py
│   └── 02_c50_stump_comparison.py
│
└── docs/
    └── rapport/
        └── main.tex           # Rapport LaTeX complet
```

### Documentation Détaillée

- 📄 **[README_C50_STUMP.md](src/c50/README_C50_STUMP.md)** : Guide complet C5.0 Stump
- 📄 **[Rapport LaTeX](docs/rapport/main.tex)** : 40+ pages de fondements mathématiques
- 📄 **Docstrings** : Toutes les fonctions documentées (format Google)

---

## 🔬 Fondements Mathématiques

### Gain Ratio (C5.0)

```
Gain Ratio = Information Gain / Split Info

où:
  Information Gain = H(S) - Σ (|Sᵢ|/|S|) × H(Sᵢ)
  Split Info = -Σ (|Sᵢ|/|S|) × log₂(|Sᵢ|/|S|)
```

### Élagage Pessimiste

```
error_rate = (E + 0.5) / (N + 1)  [Laplace smoothing]

pessimistic_error = error_rate + z × √(error_rate × (1-error_rate) / N)

Si error(feuille) ≤ error(stump) → élaguer
```

### Valeurs Manquantes

```
Pour attribut A avec seuil θ:
1. Calculer division sur valeurs valides
2. p_left = |S_left| / |S_valid|
   p_right = |S_right| / |S_valid|
3. Pour x avec A=NaN:
   Assigner à gauche avec probabilité p_left
```

---

## 👥 Contributeurs

**Équipe ENSAM Meknès 2025-2026**

- **Nankouli Marc Thierry**
- **El Khatar Saad**
- **El Filali**

**Encadrant :** Pr Hosni

---

## 📜 Licence

Ce projet est sous licence MIT. Voir [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- **Ross Quinlan** pour les algorithmes C4.5 et C5.0
- **Yoav Freund & Robert Schapire** pour AdaBoost
- **ENSAM Meknès** pour le cadre du projet

---

## 📚 Références

1. Quinlan, J.R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.
2. Quinlan, J.R. (1996). *Improved Use of Continuous Attributes in C4.5*. JAIR, 4:77-90.
3. Breiman, L. et al. (1984). *Classification and Regression Trees*. Wadsworth.
4. Hastie, T., Tibshirani, R., Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.

---
<!-- 
## 📞 Contact

Pour toute question ou suggestion : [GitHub Issues](https://github.com/votre-repo/issues) -->

---

⭐ **Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile !**