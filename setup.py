"""
Setup script pour le package decision_trees_ml
Installation : pip install -e .
"""

from setuptools import setup, find_packages
import os

# Lire le README pour la description longue
def read_long_description():
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, 'README.md')
    
    try:
        if os.path.exists(readme_path):
            with open(readme_path, 'r', encoding='utf-8') as f:
                return f.read()
    except UnicodeDecodeError:
        # Fallback si problème d'encodage
        return "Decision Trees ML - Decision Stumps and C5.0 implementation"
    return ''

# Lire les dépendances depuis requirements.txt
def read_requirements():
    here = os.path.abspath(os.path.dirname(__file__))
    req_path = os.path.join(here, 'requirements.txt')
    
    requirements = []
    try:
        if os.path.exists(req_path):
            with open(req_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Ignorer commentaires et lignes vides
                    if line and not line.startswith('#'):
                        # Extraire seulement les dépendances core (pas dev tools)
                        if not any(pkg in line.lower() for pkg in ['pytest', 'sphinx', 'black', 'flake8', 'mypy', 'isort']):
                            requirements.append(line)
    except UnicodeDecodeError:
        pass  # Utiliser les dépendances par défaut
    
    # Dépendances minimales si requirements.txt n'existe pas ou erreur
    if not requirements:
        requirements = [
            'numpy>=1.21.0',
            'pandas>=1.3.0',
            'scikit-learn>=1.0.0',
            'matplotlib>=3.4.0'
        ]
    
    return requirements

setup(
    # Métadonnées du package
    name='decision-trees-ml',
    version='1.0.0',
    author='Équipe ENSAM Meknès',
    author_email='votre.email@example.com',
    description='Implémentation complète de Decision Stumps et C5.0 from scratch',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/votre-username/decision_stumps_c50',
    
    # Configuration du package
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    
    # Fichiers supplémentaires à inclure
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.md', '*.yml', '*.yaml'],
    },
    
    # Dépendances
    install_requires=read_requirements(),
    
    # Dépendances optionnelles
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
            'isort>=5.10.0',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0',
            'sphinx-autodoc-typehints>=1.12.0',
        ],
        'visualization': [
            'graphviz>=0.20.0',
            'plotly>=5.0.0',
            'seaborn>=0.11.0',
        ],
        'deployment': [
            'flask>=2.0.0',
            'fastapi>=0.95.0',
            'uvicorn>=0.20.0',
            'streamlit>=1.20.0',
            'gunicorn>=20.1.0',
        ],
        'notebooks': [
            'jupyter>=1.0.0',
            'notebook>=6.4.0',
            'ipywidgets>=7.6.0',
        ],
    },
    
    # Scripts en ligne de commande (optionnel)
    entry_points={
        'console_scripts': [
            'dt-train=scripts.train_model:main',
            'dt-evaluate=scripts.evaluate_model:main',
            'dt-export=scripts.export_model:main',
        ],
    },
    
    # Métadonnées PyPI
    classifiers=[
        # Statut de développement
        'Development Status :: 4 - Beta',
        
        # Public cible
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        
        # Domaine
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        
        # Licence
        'License :: OSI Approved :: MIT License',
        
        # Versions Python supportées
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        
        # Système d'exploitation
        'Operating System :: OS Independent',
        
        # Type
        'Natural Language :: French',
        'Natural Language :: English',
    ],
    
    # Mots-clés pour recherche PyPI
    keywords='machine-learning decision-trees decision-stumps c50 c4.5 classification ensemble-learning adaboost boosting',
    
    # Version Python minimale
    python_requires='>=3.8',
    
    # Licence
    license='MIT',
    
    # Activation du mode développement
    zip_safe=False,
)

# Instructions post-installation
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║  ✅ Package decision-trees-ml installe avec succes!             ║
║                                                                  ║
║  📚 Prochaines etapes:                                          ║
║                                                                  ║
║  1. Importer le package:                                        ║
║     >>> from src.decision_stump import DecisionStump            ║
║                                                                  ║
║  2. Consulter les exemples:                                     ║
║     $ python examples/01_basic_decision_stump.py                ║
║                                                                  ║
║  3. Lancer les tests:                                           ║
║     $ pytest tests/ -v                                          ║
║                                                                  ║
║  4. Voir la documentation:                                      ║
║     $ cd docs && open api/index.html                            ║
║                                                                  ║
║  📖 Documentation complete: README.md                           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")