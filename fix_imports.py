"""
Script pour fixer les imports définitivement.
À exécuter UNE SEULE FOIS à la racine du projet.

Usage:
    python fix_imports.py
"""

import os
import sys
from pathlib import Path

def fix_imports():
    """Installe le package en mode développement."""
    
    # Obtenir le chemin du projet
    project_root = Path(__file__).parent.absolute()
    src_dir = project_root / 'src'
    
    print("="*70)
    print("FIXATION DES IMPORTS - DECISION TREES ML")
    print("="*70)
    print(f"\n📁 Projet : {project_root}")
    print(f"📂 Source : {src_dir}\n")
    
    if not src_dir.exists():
        print("❌ ERREUR: Le dossier 'src/' n'existe pas!")
        return False
    
    # Méthode 1: pip install -e .
    print("🔧 Méthode 1: Installation en mode développement...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-e', str(project_root)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ Installation réussie avec pip install -e .")
            print("\n🧪 Test des imports...")
            
            # Redémarrer l'interpréteur n'est pas possible, donc on ajoute au path
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            
            try:
                from decision_stump import DecisionStump
                from c50 import C50Stump
                print("✅ Import DecisionStump : OK")
                print("✅ Import C50Stump : OK")
                print("\n" + "="*70)
                print("🎉 SUCCÈS TOTAL!")
                print("="*70)
                print("\n💡 Tu peux maintenant utiliser:")
                print("   from decision_stump import DecisionStump")
                print("   from c50 import C50Stump")
                print("\n⚠️  REDÉMARRE ton terminal/IDE pour que les changements prennent effet!")
                return True
                
            except ImportError as e:
                print(f"⚠️  Import test échoué: {e}")
                print("   Essaie de redémarrer Python/IDE")
                return True  # Installation OK quand même
        else:
            print(f"⚠️  pip install a échoué: {result.stderr}")
            print("\n🔧 Tentative méthode alternative...\n")
            raise Exception("pip failed")
            
    except Exception as e:
        print(f"⚠️  Méthode pip a échoué: {e}")
        print("\n🔧 Méthode 2: Ajout direct au PYTHONPATH...\n")
        
        # Méthode 2: Ajouter au PYTHONPATH via fichier .pth
        try:
            import site
            site_packages = site.getsitepackages()[0]
            pth_file = Path(site_packages) / 'decision_trees_ml.pth'
            
            with open(pth_file, 'w') as f:
                f.write(str(src_dir) + '\n')
            
            print(f"✅ Fichier .pth créé: {pth_file}")
            print("✅ Le dossier src/ est maintenant dans PYTHONPATH")
            print("\n⚠️  REDÉMARRE Python pour que ça prenne effet!")
            print("\n💡 Tu peux maintenant utiliser:")
            print("   from decision_stump import DecisionStump")
            print("   from c50 import C50Stump")
            return True
            
        except Exception as e2:
            print(f"❌ Échec méthode 2: {e2}")
            
            # Méthode 3: Instructions manuelles
            print("\n" + "="*70)
            print("⚠️  SOLUTION MANUELLE REQUISE")
            print("="*70)
            print("\nOption A: Ajoute cette ligne au début de tes scripts:")
            print(f"   import sys")
            print(f"   sys.path.insert(0, r'{src_dir}')")
            
            print("\nOption B: Définis PYTHONPATH (permanent):")
            if os.name == 'nt':  # Windows
                print(f"   PowerShell: $env:PYTHONPATH=\"{src_dir}\"")
                print(f"   CMD: set PYTHONPATH={src_dir}")
            else:  # Unix
                print(f"   export PYTHONPATH=\"{src_dir}:$PYTHONPATH\"")
            
            return False

if __name__ == "__main__":
    success = fix_imports()
    
    if success:
        print("\n✅ Configuration terminée!")
        print("🔄 Redémarre ton terminal/IDE maintenant.")
    else:
        print("\n⚠️  Fixation partielle. Suis les instructions ci-dessus.")
    
    input("\n[Appuie sur Entrée pour fermer]")