#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de test pour le système de mémoire hiérarchique corrigé de Renée.
Ce script simulera des conversations et testera la récupération du contexte pertinent.
"""

import os
import time
import shutil
import random
from datetime import datetime
import logging

# Import de notre système de mémoire corrigé
from memory_fix import (
    FixedShortTermMemory,
    FixedLevel2Condenser,
    FixedLevel3HourlyConcepts,
    FixedLevel4DailyKnowledge,
    FixedHierarchicalMemoryOrchestrator
)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_fixed_memory")

# Mock LLM Service pour les tests
class MockLLMService:
    """Service LLM simulé pour les tests."""
    
    def generate(self, prompt, max_tokens=100):
        """Génère une réponse simulée."""
        concepts = ["Voyage", "Cuisine", "Technologie", "Musique", "Cinéma", "Littérature"]
        concept = random.choice(concepts)
        return f"Réponse générée sur le thème '{concept}'. Ce texte est généré automatiquement pour simuler une réponse du LLM."

# Fonction pour simuler des conversations
def generate_conversation(topic):
    """Génère une conversation simulée sur un topic donné."""
    
    topics_prompts = {
        "Voyage": [
            "Quelles sont les meilleures destinations pour voyager en été?",
            "As-tu déjà visité Paris? Que penses-tu de cette ville?",
            "Comment préparer un voyage en sac à dos?",
            "Quels pays offrent les plus beaux paysages naturels?"
        ],
        "Cuisine": [
            "Comment faire une bonne pâte à pizza?",
            "Quels sont les principes de base de la cuisine française?",
            "Comment faire un bon risotto?",
            "Quelles épices sont essentielles dans une cuisine bien équipée?"
        ],
        "Technologie": [
            "Que penses-tu de l'intelligence artificielle?",
            "Comment fonctionne un ordinateur quantique?",
            "Quelles sont les tendances actuelles en matière de smartphones?",
            "Peux-tu m'expliquer ce qu'est le machine learning?"
        ],
        "Musique": [
            "Quels sont tes genres musicaux préférés?",
            "Comment la musique a-t-elle évolué au cours du siècle dernier?",
            "Peux-tu me recommander des artistes de jazz?",
            "Quelle est l'importance de la musique dans la culture?"
        ]
    }
    
    # Sélection aléatoire d'une question si le topic existe
    if topic in topics_prompts:
        user_input = random.choice(topics_prompts[topic])
    else:
        # Topic par défaut
        user_input = "Peux-tu me parler de quelque chose d'intéressant?"
    
    # Génération d'une réponse simulée
    response = f"Voici ma réponse sur {topic}: {' '.join([topic] * 5)} est un sujet fascinant avec beaucoup de facettes intéressantes. On pourrait en discuter pendant des heures."
    
    return user_input, response

# Fonction principale de test
def test_memory_system():
    """Teste le système de mémoire hiérarchique corrigé."""
    
    # Création du répertoire de test
    test_dir = "/tmp/renee_memory_test"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    # Création du service LLM mock
    llm_service = MockLLMService()
    
    # Initialisation de l'orchestrateur
    orchestrator = FixedHierarchicalMemoryOrchestrator(
        data_dir=test_dir,
        llm_service=llm_service
    )
    
    logger.info("=== Démarrage du test de mémoire hiérarchique ===")
    
    # Phase 1: Génération de conversations
    logger.info("Phase 1: Génération de conversations...")
    topics = ["Voyage", "Cuisine", "Technologie", "Voyage", "Cuisine", "Technologie"]
    
    for i, topic in enumerate(topics):
        user_input, response = generate_conversation(topic)
        conversation_id = orchestrator.add_conversation(user_input, response)
        logger.info(f"Conversation {i+1} ajoutée: {topic} (ID: {conversation_id})")
        
        # Pause pour simuler l'écoulement du temps
        time.sleep(0.5)
    
    # Phase 2: Récupération des conversations récentes
    logger.info("Phase 2: Test de récupération des conversations récentes...")
    recent_conversations = orchestrator.short_term_memory.get_recent_conversations(limit=10)
    
    logger.info(f"Nombre de conversations récentes: {len(recent_conversations)}")
    for i, conv in enumerate(recent_conversations[:3]):  # Afficher seulement les 3 premières
        logger.info(f"Conversation {i+1}: User: '{conv.prompt.content[:50]}...' Renée: '{conv.response.content[:50]}...'")
    
    # Phase 3: Déclenchement manuel de la condensation
    logger.info("Phase 3: Test de condensation manuelle...")
    condensation_results = orchestrator.trigger_manual_condensation()
    
    for level, result in condensation_results.items():
        if result and result.get("success"):
            logger.info(f"{level}: {result.get('count')} éléments générés")
        else:
            logger.info(f"{level}: échec - {result.get('error', 'Erreur inconnue')}")
    
    # Phase 4: Test de récupération du contexte
    logger.info("Phase 4: Test de récupération du contexte...")
    test_queries = [
        "Parle-moi de voyage",
        "J'aimerais cuisiner quelque chose de bon",
        "Explique-moi la technologie"
    ]
    
    for query in test_queries:
        logger.info(f"Test de requête: '{query}'")
        
        # Récupération du contexte
        context = orchestrator.get_context_for_query(query)
        
        # Affichage des statistiques
        logger.info(f"  - Conversations pertinentes: {len(context['short_term'])}")
        logger.info(f"  - Résumés pertinents: {len(context['summaries'])}")
        logger.info(f"  - Concepts pertinents: {len(context['concepts'])}")
        logger.info(f"  - Connaissances pertinentes: {len(context['knowledge'])}")
        
        # Composition du contexte
        composed_context = orchestrator.compose_context(query)
        logger.info(f"  - Taille du contexte composé: ~{len(composed_context)} caractères")
    
    logger.info("=== Test terminé avec succès ===")
    
    return orchestrator  # Renvoie l'orchestrateur pour d'autres tests si nécessaire

if __name__ == "__main__":
    test_memory_system()
