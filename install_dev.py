"""
Script d'installation en mode développement.
Alternative à 'pip install -e .' qui évite les problèmes d'encodage sur Windows.
"""

import os
import sys
import site

def install_dev_mode():
    """Installe le package en mode développement."""
    
    # Obtenir le chemin du projet
    project_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(project_dir, 'src')
    
    # Obtenir le répertoire site-packages
    site_packages = site.getsitepackages()[0]
    
    # Créer un fichier .pth pour ajouter src/ au PYTHONPATH
    pth_file = os.path.join(site_packages, 'decision_trees_ml_dev.pth')
    
    try:
        with open(pth_file, 'w', encoding='utf-8') as f:
            f.write(src_dir + '\n')
        
        print(f"✅ Installation réussie en mode développement!")
        print(f"📁 Fichier créé: {pth_file}")
        print(f"📂 Chemin ajouté: {src_dir}")
        print()
        print("🧪 Test de l'import:")
        
        # Tester l'import
        sys.path.insert(0, src_dir)
        from decision_stump import DecisionStump # type: ignore
        print("   >>> from decision_stump import DecisionStump")
        print("   ✅ Import réussi!")
        print()
        print("💡 Vous pouvez maintenant utiliser:")
        print("   >>> from decision_stump import DecisionStump")
        print("   >>> from decision_stump import gini_impurity, entropy")
        print()
        print("🔄 Redémarrez Python pour que les changements prennent effet.")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'installation: {e}")
        print()
        print("📋 Solution alternative:")
        print(f"   Ajoutez manuellement ce chemin à votre PYTHONPATH:")
        print(f"   {src_dir}")
        print()
        print("   PowerShell:")
        print(f'   $env:PYTHONPATH = "$env:PYTHONPATH;{src_dir}"')
        print()
        print("   CMD:")
        print(f'   set PYTHONPATH=%PYTHONPATH%;{src_dir}')
        return False
    
    return True


if __name__ == "__main__":
    print("="*70)
    print("INSTALLATION EN MODE DÉVELOPPEMENT")
    print("="*70)
    print()
    
    install_dev_mode()