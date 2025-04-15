#!/usr/bin/env python3
# reset_rag_database.py
# Script pour réinitialiser la base de données RAG

import os
import sys
import json
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ReneeRAGReset")

# Ajout du chemin parent au sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import du module RAG
from src.rag.rag_hub import RAGHub

# Chargement de la configuration
config_path = os.path.join(os.path.dirname(__file__), "config/config.json")
with open(config_path) as f:
    config = json.load(f)


def reset_rag_database():
    """Réinitialise la base de données RAG."""
    logger.info("Initialisation du RAG Hub...")
    
    # Initialisation du RAG Hub
    rag_hub = RAGHub(
        vector_db=config["components"]["rag"]["vector_db"],
        embedding_model=config["components"]["rag"]["embedding_model"],
        reranking_enabled=config["components"]["rag"]["reranking_enabled"]
    )
    
    logger.info("Réinitialisation de la base de données RAG...")
    success = rag_hub.reset_collection()
    
    if success:
        logger.info("La base de données RAG a été réinitialisée avec succès.")
        logger.info("Tous les documents ont été supprimés.")
        
        # Vérifier si le dossier processed existe
        processed_dir = os.path.expanduser("~/Desktop/ReneeRAGDocs/processed")
        if os.path.exists(processed_dir):
            num_files = len([f for f in os.listdir(processed_dir) if os.path.isfile(os.path.join(processed_dir, f))])
            if num_files > 0:
                logger.info(f"Vous avez {num_files} fichiers dans le dossier 'processed'.")
                logger.info("Si vous souhaitez les réimporter, déplacez-les dans le dossier '~/Desktop/ReneeRAGDocs/'.")
    else:
        logger.error("Échec de la réinitialisation de la base de données RAG.")
        logger.error("Vérifiez les logs pour plus de détails.")


if __name__ == "__main__":
    reset_rag_database()
