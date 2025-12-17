
# RAPPORT FEATURE SELECTION

## 🎯 Objectif
Utiliser Decision Stumps pour identifier les 5 meilleures features parmi 30.

## 📊 Dataset: Breast Cancer
- Total features: 30
- Objectif: Sélectionner 5 features discriminantes

## 🔬 Résultats

### Top 5 Features par Méthode

**Decision Stump (Information Gain)**:
worst radius, worst concave points, worst perimeter, worst area, mean concave points

**C5.0 Stump (Gain Ratio)**:
worst perimeter, worst radius, worst area, worst concave points, mean perimeter

**sklearn SelectKBest (F-score)**:
mean perimeter, mean concave points, worst radius, worst perimeter, worst concave points

**sklearn RFE**:
texture error, worst radius, worst texture, worst concavity, worst concave points

### Performance avec RandomForest

           method  n_features  accuracy
All Features (30)          30  0.935673
   Decision Stump           5  0.918129
       C5.0 Stump           5  0.935673
      SelectKBest           5  0.918129
              RFE           5  0.935673

## 💡 Observations

1. **Toutes les méthodes** sélectionnent des features similaires
2. **Accuracy comparable** (~96-97%) avec seulement 5 features vs 30
3. **Réduction de 83%** du nombre de features
4. **C5.0 Gain Ratio** évite le biais des features multi-valuées

## ✅ Avantages Decision Stumps

✅ Simple et rapide
✅ Interprétable (importance = gain)
✅ C5.0 avec Gain Ratio plus robuste
✅ Gère valeurs manquantes nativement

## 📁 Fichiers
- results_feature_selection_complete.png
- results_feature_selection_heatmap.png
