#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de test pour l'enregistrement de la mémoire hiérarchique
"""

import os
import json
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_file_write_simple():
    """Test simple d'écriture de fichier"""
    test_path = "./debug_memory/test_simple.txt"
    
    # Création du répertoire si nécessaire
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    
    try:
        # Écriture simple
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(f"Test simple d'écriture: {datetime.now().isoformat()}\n")
        
        logger.info(f"✅ Test simple réussi: {test_path}")
        
        # Vérification
        if os.path.exists(test_path):
            with open(test_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Contenu: {content}")
            return True
        else:
            logger.error(f"❌ Fichier non créé: {test_path}")
            return False
    except Exception as e:
        logger.error(f"❌ Erreur lors du test simple: {e}")
        return False

def test_json_write():
    """Test d'écriture de fichier JSON"""
    test_path = "./debug_memory/test_json.json"
    
    # Création du répertoire si nécessaire
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    
    try:
        # Création d'un dictionnaire de test
        test_data = {
            "timestamp": datetime.now().isoformat(),
            "test_key": "test_value",
            "nested": {
                "level1": {
                    "level2": "deep value"
                }
            },
            "array": [1, 2, 3, "test"]
        }
        
        # Écriture au format JSON
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Test JSON réussi: {test_path}")
        
        # Vérification
        if os.path.exists(test_path):
            with open(test_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            logger.info(f"Clés JSON: {list(content.keys())}")
            return True
        else:
            logger.error(f"❌ Fichier JSON non créé: {test_path}")
            return False
    except Exception as e:
        logger.error(f"❌ Erreur lors du test JSON: {e}")
        return False

def test_memory_structure():
    """Test de la structure de mémoire hiérarchique"""
    test_path = "./debug_memory/test_memory_structure.json"
    
    # Création du répertoire si nécessaire
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    
    try:
        # Création d'une structure de mémoire simplifiée
        memory_data = {
            "memory_levels": {
                "LEVEL_1": [
                    {
                        "id": 1,
                        "content": "Test mémoire court terme",
                        "level": "LEVEL_1",
                        "importance": 0.5,
                        "created_at": datetime.now().isoformat(),
                        "last_accessed": datetime.now().isoformat(),
                        "access_count": 1,
                        "metadata": {"source": "test"}
                    }
                ],
                "LEVEL_2": [
                    {
                        "id": 2,
                        "content": "Test mémoire travail",
                        "level": "LEVEL_2",
                        "importance": 0.7,
                        "created_at": datetime.now().isoformat(),
                        "last_accessed": datetime.now().isoformat(),
                        "access_count": 0,
                        "metadata": {"source": "test"}
                    }
                ],
                "LEVEL_3": [],
                "FACTUAL": []
            },
            "memory_id_counter": 3,
            "last_saved": datetime.now().isoformat()
        }
        
        # Écriture au format JSON
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Test structure mémoire réussi: {test_path}")
        
        # Vérification
        if os.path.exists(test_path):
            with open(test_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            level1_count = len(content["memory_levels"]["LEVEL_1"])
            logger.info(f"Structure mémoire: {level1_count} éléments en LEVEL_1")
            return True
        else:
            logger.error(f"❌ Fichier structure mémoire non créé: {test_path}")
            return False
    except Exception as e:
        logger.error(f"❌ Erreur lors du test structure mémoire: {e}")
        return False

def test_alternative_paths():
    """Test de chemins alternatifs pour l'enregistrement"""
    
    paths_to_test = [
        "./memory_test.json",                 # Répertoire racine
        "/tmp/memory_test.json",              # Répertoire temporaire système
        os.path.expanduser("~/memory_test.json")  # Répertoire utilisateur
    ]
    
    results = {}
    
    for test_path in paths_to_test:
        try:
            # Création du répertoire si nécessaire (pour les chemins avec sous-dossiers)
            directory = os.path.dirname(test_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            # Écriture simple
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write(f"Test de chemin alternatif: {datetime.now().isoformat()}\n")
            
            # Vérification
            if os.path.exists(test_path):
                results[test_path] = "✅ RÉUSSI"
                logger.info(f"✅ Test chemin {test_path} réussi")
            else:
                results[test_path] = "❌ ÉCHEC (non créé)"
                logger.error(f"❌ Échec chemin {test_path}: fichier non créé")
        except Exception as e:
            results[test_path] = f"❌ ERREUR: {str(e)}"
            logger.error(f"❌ Erreur chemin {test_path}: {e}")
    
    return results

if __name__ == "__main__":
    logger.info("==== DÉBUT DES TESTS D'ENREGISTREMENT MÉMOIRE ====")
    
    # Exécution des tests
    test_file_write_simple()
    test_json_write()
    test_memory_structure()
    path_results = test_alternative_paths()
    
    # Affichage des résultats des tests de chemins
    logger.info("==== RÉSULTATS DES TESTS DE CHEMINS ALTERNATIFS ====")
    for path, result in path_results.items():
        logger.info(f"{path}: {result}")
    
    logger.info("==== FIN DES TESTS D'ENREGISTREMENT MÉMOIRE ====")
