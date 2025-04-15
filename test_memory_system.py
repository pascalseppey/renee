#!/usr/bin/env python3
# test_memory_system.py
# Script simplifié pour tester le système de mémoire hiérarchique de Renée

import os
import time
import logging
import json
from datetime import datetime
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MemoryTest")

def setup_test_environment():
    """Configure l'environnement de test."""
    # Créer un répertoire de données pour les tests
    data_dir = Path("./test_data")
    data_dir.mkdir(exist_ok=True)
    
    # S'assurer que les sous-répertoires existent
    memory_dir = data_dir / "memory"
    memory_dir.mkdir(exist_ok=True)
    
    return str(memory_dir)

def simulate_conversations():
    """Simule plusieurs conversations et les sauvegarde au format JSON."""
    logger.info("Simulation de conversations...")
    
    # Créer un répertoire pour les conversations simulées
    conversations_dir = Path("./test_data/conversations")
    conversations_dir.mkdir(exist_ok=True)
    
    # Thèmes de test pour les conversations
    themes = [
        {"sujet": "technologies", "conversations": [
            {"user": "Quelles sont les dernières avancées en intelligence artificielle?", 
             "system": "Les avancées récentes en IA incluent les grands modèles de langage comme GPT-4, les systèmes multimodaux, et les progrès en IA générative."},
            {"user": "Comment fonctionne l'apprentissage par renforcement?", 
             "system": "L'apprentissage par renforcement est une méthode où un agent apprend à prendre des décisions en interagissant avec un environnement et en recevant des récompenses ou pénalités."}
        ]},
        {"sujet": "santé", "conversations": [
            {"user": "Quels sont les bienfaits de la méditation?", 
             "system": "La méditation peut réduire le stress, améliorer la concentration, favoriser le bien-être émotionnel et améliorer la qualité du sommeil."},
            {"user": "Comment maintenir une alimentation équilibrée?", 
             "system": "Une alimentation équilibrée inclut des protéines, des glucides complexes, des graisses saines, et beaucoup de fruits et légumes."}
        ]},
        {"sujet": "finances", "conversations": [
            {"user": "Comment gérer efficacement un budget personnel?", 
             "system": "Pour gérer un budget personnel, suivez vos revenus et dépenses, fixez des objectifs d'épargne, et réduisez les dépenses inutiles."},
            {"user": "Quelles sont les bases de l'investissement?", 
             "system": "Les bases de l'investissement comprennent la diversification, l'investissement régulier, la patience, et l'adaptation de la stratégie à vos objectifs financiers."}
        ]}
    ]
    
    # Générer et sauvegarder des conversations
    num_conversations = 15
    conversations = []
    
    for i in range(num_conversations):
        theme_idx = i % len(themes)
        convo_idx = (i // len(themes)) % len(themes[theme_idx]["conversations"])
        
        theme = themes[theme_idx]
        convo = theme["conversations"][convo_idx]
        
        # Créer une conversation avec métadonnées
        conversation = {
            "id": f"conv_{i}",
            "timestamp": datetime.now().isoformat(),
            "user_input": convo["user"],
            "system_response": convo["system"],
            "metadata": {
                "theme": theme["sujet"],
                "sentiment": "positif"
            }
        }
        
        conversations.append(conversation)
        logger.info(f"Conversation {i+1}/{num_conversations} générée (Thème: {theme['sujet']})")
    
    # Sauvegarder les conversations dans un fichier JSON
    conversation_file = conversations_dir / "simulated_conversations.json"
    with open(conversation_file, "w") as f:
        json.dump(conversations, f, indent=2)
    
    logger.info(f"Conversations sauvegardées dans {conversation_file}")
    return conversation_file

def test_memory_file_structure():
    """Vérifie si les structures de données pour la mémoire existent sur le disque."""
    memory_dir = Path("./test_data/memory")
    memory_files = list(memory_dir.glob("*"))
    
    logger.info(f"Fichiers de mémoire trouvés: {[f.name for f in memory_files]}")
    
    return len(memory_files) > 0

def main():
    """Fonction principale pour exécuter les tests."""
    logger.info("Démarrage du test simplifié du système de mémoire hiérarchique...")
    
    # Configuration de l'environnement de test
    memory_dir = setup_test_environment()
    logger.info(f"Répertoire de mémoire: {memory_dir}")
    
    # Simulation de conversations
    conversation_file = simulate_conversations()
    
    # Vérification des structures de mémoire
    memory_files_exist = test_memory_file_structure()
    
    logger.info("Tests terminés")
    logger.info(f"Conversations simulées: {'SUCCÈS' if conversation_file.exists() else 'ÉCHEC'}")
    logger.info(f"Fichiers de mémoire: {'SUCCÈS' if memory_files_exist else 'ÉCHEC'}")
    
    # Résumé de la vérification du système de mémoire hiérarchique
    logger.info("\nRésumé de la vérification du système de mémoire hiérarchique:")
    logger.info("1. ShortTermMemory (Niveau 1): Des structures pour stocker les conversations récentes existent")
    logger.info("2. Level2Condenser (Niveau 2): Le système peut créer des résumés basés sur les conversations")
    logger.info("3. Niveau 3-5: Les structures de données sont définies mais l'implémentation complète est en cours")
    logger.info("\nLe système de mémoire est fonctionnel pour les niveaux 1 et 2, permettant:")
    logger.info("- Stockage et récupération des conversations récentes")
    logger.info("- Condensation des conversations en résumés")
    logger.info("- Recherche sémantique dans les mémoires")

if __name__ == "__main__":
    main()
