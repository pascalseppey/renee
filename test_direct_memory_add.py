#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test d'ajout direct à la mémoire hiérarchique et vérification de la mise à jour
"""

import os
import sys
import json
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_to_memory_file(memory_file="./debug_memory/hierarchical_memory.json"):
    """Ajoute directement une entrée au fichier de mémoire"""
    try:
        # Vérifier si le fichier existe
        if not os.path.exists(memory_file):
            logger.error(f"❌ Le fichier {memory_file} n'existe pas")
            return False
        
        # Lire le fichier
        with open(memory_file, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
        
        # Sauvegarder l'état avant modification
        logger.info(f"État initial: {sum(len(memories) for level, memories in memory_data['memory_levels'].items())} mémoires")
        logger.info(f"Compteur initial: {memory_data['memory_id_counter']}")
        
        # Incrémenter le compteur de mémoire
        new_id = memory_data['memory_id_counter'] + 1
        memory_data['memory_id_counter'] = new_id
        
        # Ajouter une nouvelle entrée en mémoire factuelle
        new_memory = {
            "id": new_id,
            "content": f"Test d'ajout direct à la mémoire #{new_id}. La mémoire hiérarchique est cruciale pour Renée.",
            "level": "FACTUAL",
            "importance": 0.9,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "access_count": 0,
            "metadata": {
                "source": "test_direct_add",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Ajouter à la liste des mémoires factuelles
        memory_data["memory_levels"]["FACTUAL"].append(new_memory)
        
        # Mettre à jour la date de dernière sauvegarde
        memory_data["last_saved"] = datetime.now().isoformat()
        
        # Écrire le fichier
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Ajout réussi à la mémoire - ID: {new_id}")
        
        # Vérifier la mise à jour
        with open(memory_file, 'r', encoding='utf-8') as f:
            updated_data = json.load(f)
        
        new_memory_count = sum(len(memories) for level, memories in updated_data["memory_levels"].items())
        logger.info(f"État après mise à jour: {new_memory_count} mémoires")
        logger.info(f"Compteur après mise à jour: {updated_data['memory_id_counter']}")
        
        # Vérifier que la nouvelle mémoire est bien présente
        factual_memories = updated_data["memory_levels"]["FACTUAL"]
        found = any(mem["id"] == new_id for mem in factual_memories)
        
        if found:
            logger.info(f"✅ Vérification réussie: mémoire ID {new_id} trouvée")
            return True
        else:
            logger.error(f"❌ Vérification échouée: mémoire ID {new_id} non trouvée")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'ajout direct à la mémoire: {e}")
        return False

if __name__ == "__main__":
    logger.info("==== TEST D'AJOUT DIRECT À LA MÉMOIRE HIÉRARCHIQUE ====")
    success = add_to_memory_file()
    
    if success:
        logger.info("==== TEST RÉUSSI ====")
    else:
        logger.error("==== TEST ÉCHOUÉ ====")
        sys.exit(1)
