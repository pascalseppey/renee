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
