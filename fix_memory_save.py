#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de correction pour la sauvegarde de mémoire de Renée
Ce script crée et sauvegarde directement la mémoire hiérarchique pour déverrouiller le processus.
"""

import os
import json
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_memory_file(output_path="./debug_memory/hierarchical_memory.json"):
    """Crée directement un fichier de mémoire hiérarchique pour Renée"""
    
    # S'assurer que le répertoire existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Structure de base de la mémoire
    memory_data = {
        "memory_levels": {
            "LEVEL_1": [],  # Mémoire à court terme
            "LEVEL_2": [],  # Mémoire de travail
            "LEVEL_3": [],  # Mémoire à long terme
            "FACTUAL": []   # Mémoire factuelle
        },
        "memory_id_counter": 1,
        "last_saved": datetime.now().isoformat()
    }
    
    # Ajouter quelques mémoires initiales
    memory_data["memory_levels"]["LEVEL_3"].append({
        "id": 1,
        "content": "L'utilisateur préfère utiliser des solutions modernes comme les API REST et React avec WordPress.",
        "level": "LEVEL_3",
        "importance": 0.8,
        "created_at": datetime.now().isoformat(),
        "last_accessed": datetime.now().isoformat(),
        "access_count": 0,
        "metadata": {"source": "memory_fix", "tags": ["wordpress", "api", "rest", "react"]}
    })
    
    # Sauvegarder dans le fichier de sortie
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)
        
        # Vérifier la création
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"✅ Fichier de mémoire créé avec succès: {output_path} ({file_size} octets)")
            return True
        else:
            logger.error(f"❌ Échec de création du fichier: {output_path}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la création du fichier de mémoire: {e}")
        return False

def main():
    """Fonction principale"""
    logger.info("==== CRÉATION DU FICHIER DE MÉMOIRE HIÉRARCHIQUE ====")
    
    # Chemins à tester
    paths = [
        "./debug_memory/hierarchical_memory.json",
        "/tmp/hierarchical_memory.json"
    ]
    
    success = False
    for path in paths:
        if create_memory_file(path):
            success = True
    
    if success:
        logger.info("==== PROCESSUS TERMINÉ AVEC SUCCÈS ====")
    else:
        logger.error("==== ÉCHEC DU PROCESSUS ====")
        sys.exit(1)

if __name__ == "__main__":
    main()
