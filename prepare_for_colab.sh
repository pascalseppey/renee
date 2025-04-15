#!/bin/bash

# Script pour préparer le projet RLHF WordPress Memory Trainer pour Google Colab

# Créer le répertoire de données s'il n'existe pas
mkdir -p ./data/memory

# Créer un fichier .gitignore pour exclure les fichiers inutiles
cat > .gitignore << EOL
__pycache__/
*.py[cod]
*$py.class
.DS_Store
*.so
.env
checkpoint_*
EOL

# Créer un fichier README.md pour le dépôt GitHub
cat > README.md << EOL
# RLHF WordPress Memory Trainer

Système avancé de Reinforcement Learning from Human Feedback (RLHF) spécialisé pour WordPress et Elementor, avec une gestion sophistiquée de la mémoire hiérarchique.

## Fonctionnalités

- Test de DeepSeek sur sa capacité à utiliser la mémoire hiérarchique
- Évaluation des compétences WordPress et Elementor
- Tests d'interaction avec l'API REST de WordPress
- Capacité à naviguer sur internet via Crawl4ai
- Feedback automatisé via GPT-4o (modèle critique)

## Utilisation sur Google Colab

Voir le notebook \`colab_trainer.ipynb\` pour exécuter l'entraînement sur Google Colab.
EOL

# Créer un notebook Colab pour l'exécution
cat > colab_trainer.ipynb << EOL
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# RLHF WordPress Memory Trainer sur Google Colab\n",
    "\n",
    "Ce notebook permet d'exécuter le trainer RLHF de DeepSeek sur Google Colab avec accélération GPU."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Installation des dépendances"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cloner le dépôt\n",
    "!git clone https://github.com/VOTRE_USERNAME/renee-project.git\n",
    "# Remplacez VOTRE_USERNAME par votre nom d'utilisateur GitHub\n",
    "%cd renee-project"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Installer les dépendances nécessaires\n",
    "!pip install openai requests"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Configuration de l'environnement Colab"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Créer les répertoires nécessaires\n",
    "!mkdir -p ./data/memory\n",
    "!mkdir -p ./checkpoints"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Configuration pour utiliser PyTorch avec GPU"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Vérifier si le GPU est disponible\n",
    "import torch\n",
    "print(f\"GPU disponible: {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"Nom du GPU: {torch.cuda.get_device_name(0)}\")\n",
    "    print(f\"Mémoire totale: {torch.cuda.get_device_properties(0).total_memory / 1e9} GB\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Préparer les clés API"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Configurer la clé API OpenAI\n",
    "import os\n",
    "os.environ[\"OPENAI_API_KEY\"] = \"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\"\n",
    "# Remplacez par votre clé API si nécessaire"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Installation d'Ollama pour exécuter DeepSeek localement\n",
    "\n",
    "Étant donné que Google Colab est un environnement Linux, nous pouvons installer Ollama directement."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Installer Ollama pour Linux\n",
    "!curl -fsSL https://ollama.com/install.sh | sh\n",
    "# Démarrer le service Ollama en arrière-plan\n",
    "!ollama serve &"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Télécharger le modèle DeepSeek\n",
    "!ollama pull deepseek:7b-chat"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Exécution de l'entraînement RLHF"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Adapter le script pour Google Colab\n",
    "!python rlhf_wordpress_memory_trainer.py --num_examples 10 --verbose"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Vérification de la mémoire hiérarchique"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Vérifier le contenu du fichier de mémoire hiérarchique\n",
    "import json\n",
    "with open('./data/memory/hierarchical_memory.json', 'r') as f:\n",
    "    memory_data = json.load(f)\n",
    "\n",
    "# Afficher les statistiques\n",
    "print(\"Statistiques de la mémoire:\")\n",
    "for level, memories in memory_data[\"memory_levels\"].items():\n",
    "    print(f\"  {level}: {len(memories)} entrées\")\n",
    "print(f\"Total: {sum(len(memories) for level, memories in memory_data['memory_levels'].items())} entrées\")"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOL

# Création d'un script pour adapter le code à Colab
cat > adapt_for_colab.py << EOL
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour adapter le RLHF WordPress Memory Trainer à Google Colab
"""

import os
import sys
import re

def adapt_file(file_path):
    """Adapte un fichier Python pour Google Colab"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Adaptations à effectuer
    replacements = [
        # Modifier les chemins de fichiers si nécessaire
        (r'./data/memory', './data/memory'),
        
        # Autres adaptations spécifiques à Colab pourraient être ajoutées ici
    ]
    
    # Appliquer les remplacements
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Sauvegarder le fichier modifié
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Le fichier {file_path} a été adapté pour Google Colab")

def main():
    """Fonction principale"""
    # Liste des fichiers à adapter
    files_to_adapt = [
        'rlhf_wordpress_memory_trainer.py',
        'rlhf_deepseek_trainer.py'
    ]
    
    for file in files_to_adapt:
        if os.path.exists(file):
            adapt_file(file)
        else:
            print(f"Le fichier {file} n'existe pas")
    
    print("Adaptation pour Google Colab terminée")

if __name__ == "__main__":
    main()
EOL

# Rendre les scripts exécutables
chmod +x prepare_for_colab.sh
chmod +x adapt_for_colab.py

echo "Scripts de préparation pour Google Colab créés avec succès !"
echo "Exécutez ./prepare_for_colab.sh pour préparer le projet"
