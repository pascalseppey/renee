#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test direct de la sauvegarde de mémoire hiérarchique
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importer la classe HierarchicalMemory du module principal
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rlhf_wordpress_memory_trainer import HierarchicalMemory

def test_memory_save():
    """Test direct de la sauvegarde de mémoire"""
    # Créer une instance de mémoire
    memory = HierarchicalMemory()
    
    # Ajouter quelques mémoires de test
    memory.add_memory(
        content="Test mémoire court terme",
        level="LEVEL_1",
        importance=0.5,
        metadata={"source": "test_direct"}
    )
    
    memory.add_memory(
        content="Test mémoire de travail",
        level="LEVEL_2",
        importance=0.7,
        metadata={"source": "test_direct"}
    )
    
    memory.add_memory(
        content="Test mémoire long terme",
        level="LEVEL_3",
        importance=0.8,
        metadata={"source": "test_direct"}
    )
    
    # Tester la sauvegarde avec différents chemins
    paths_to_test = [
        "./debug_memory/direct_test_memory.json",
        "./direct_test_memory.json",
        "/tmp/direct_test_memory.json"
    ]
    
    for test_path in paths_to_test:
        try:
            # Sauvegarde directe
            memory.save_memory_to_file(test_path)
            
            # Vérification
            if os.path.exists(test_path):
                logger.info(f"✅ Sauvegarde réussie: {test_path}")
                
                # Afficher la taille du fichier
                file_size = os.path.getsize(test_path)
                logger.info(f"Taille du fichier: {file_size} octets")
                
                # Lire le contenu pour vérification
                with open(test_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Vérifier les données
                level_counts = {level: len(memories) for level, memories in data["memory_levels"].items()}
                logger.info(f"Contenu: {level_counts}")
            else:
                logger.error(f"❌ Échec de sauvegarde: {test_path}")
        except Exception as e:
            logger.error(f"❌ Erreur lors du test de sauvegarde {test_path}: {e}")

if __name__ == "__main__":
    logger.info("==== TEST DIRECT DE SAUVEGARDE MÉMOIRE ====")
    test_memory_save()
    logger.info("==== TEST TERMINÉ ====")
