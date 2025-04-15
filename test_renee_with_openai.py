#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de test intégrant le système de mémoire hiérarchique corrigé avec Renée
et le service OpenAI pour générer des réponses cohérentes.
"""

import os
import time
import shutil
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Any, Optional

# Import des modules nécessaires
from memory_fix import (
    FixedHierarchicalMemoryOrchestrator,
    UserPrompt, SystemResponse, Conversation, Summary, MetaConcept, KnowledgeItem
)
from openai_service import OpenAIService

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("renee_openai_test")

# Classe simulant le système complet de Renée avec OpenAI
class ReneeWithOpenAI:
    """Système complet de Renée utilisant OpenAI pour la génération de réponses."""
    
    def __init__(self, data_dir: str, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialise le système Renée avec OpenAI.
        
        Args:
            data_dir: Répertoire pour les données persistantes
            api_key: Clé API OpenAI (facultatif, peut être définie via env var OPENAI_API_KEY)
            model: Modèle OpenAI à utiliser
        """
        self.data_dir = data_dir
        
        # Création du répertoire de données s'il n'existe pas
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialisation du service OpenAI
        self.openai_service = OpenAIService(api_key=api_key, model=model)
        
        # Initialisation du système de mémoire hiérarchique
        self.memory = FixedHierarchicalMemoryOrchestrator(
            data_dir=os.path.join(data_dir, "memory"),
            llm_service=self.openai_service
        )
        
        # Historique des conversations pour le test
        self.conversation_history = []
        
        logger.info(f"Système Renée initialisé avec mémoire hiérarchique et OpenAI ({model})")
    
    def process_input(self, user_input: str, user_id: str = "test_user", 
                      include_memory: bool = True) -> Dict[str, Any]:
        """
        Traite une entrée utilisateur et génère une réponse cohérente.
        
        Args:
            user_input: Message de l'utilisateur
            user_id: Identifiant de l'utilisateur
            include_memory: Si True, utilise la mémoire pour enrichir la réponse
            
        Returns:
            Dictionnaire contenant la réponse et les métadonnées
        """
        start_time = datetime.now()
        
        # Métadonnées de l'entrée utilisateur
        user_metadata = {
            "user_id": user_id,
            "timestamp": start_time.isoformat(),
            "session_id": "test_session"
        }
        
        # Préparation du prompt pour le LLM
        memory_context = ""
        if include_memory:
            memory_context = self.memory.compose_context(user_input)
        
        # Génération de la réponse avec OpenAI
        if include_memory and memory_context:
            # Génération avec contexte mémoire
            response_text = self.openai_service.generate_coherent_response(user_input, memory_context)
        else:
            # Génération simple sans contexte
            response_text = self.openai_service.generate(user_input)
        
        end_time = datetime.now()
        
        # Métadonnées de la réponse
        response_metadata = {
            "timestamp": end_time.isoformat(),
            "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
            "include_memory": include_memory,
            "model": self.openai_service.model
        }
        
        # Ajout à la mémoire
        conversation_id = self.memory.add_conversation(
            user_input, response_text,
            user_metadata, response_metadata
        )
        
        # Enregistrement dans l'historique
        self.conversation_history.append({
            "id": conversation_id,
            "user_input": user_input,
            "response": response_text,
            "timestamp": end_time.isoformat(),
            "include_memory": include_memory
        })
        
        return {
            "response": response_text,
            "conversation_id": conversation_id,
            "memory_used": include_memory,
            "memory_context_length": len(memory_context) if memory_context else 0,
            "processing_time_ms": response_metadata["processing_time_ms"]
        }
    
    def trigger_memory_condensation(self):
        """Force la condensation de la mémoire à tous les niveaux (pour les tests)."""
        result = self.memory.trigger_manual_condensation()
        return result
    
    def print_memory_statistics(self):
        """Affiche des statistiques sur la mémoire actuelle."""
        stats = {
            "short_term_count": len(self.memory.short_term_memory.conversations),
            "level2_summaries_count": len(self.memory.level2_condenser.summaries),
            "level3_concepts_count": len(self.memory.level3_concepts.meta_concepts),
            "level4_knowledge_count": len(self.memory.level4_knowledge.knowledge_items),
        }
        
        logger.info("=== Statistiques de mémoire ===")
        logger.info(f"Conversations en mémoire à court terme: {stats['short_term_count']}")
        logger.info(f"Résumés de niveau 2: {stats['level2_summaries_count']}")
        logger.info(f"Concepts de niveau 3: {stats['level3_concepts_count']}")
        logger.info(f"Connaissances de niveau 4: {stats['level4_knowledge_count']}")
        
        return stats

# Fonction de test simulant un écoulement de temps accéléré
def test_with_time_simulation():
    """
    Teste le système Renée avec une simulation d'écoulement de temps accéléré.
    """
    # Création du répertoire de test
    test_dir = "/tmp/renee_openai_test"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    # Initialisation du système Renée
    renee = ReneeWithOpenAI(data_dir=test_dir)
    
    logger.info("=== Début du test Renée avec OpenAI et simulation temporelle ===")
    
    # Phase 1: Conversations initiales
    logger.info("Phase 1: Conversations initiales...")
    
    initial_conversations = [
        "Bonjour, je m'appelle Pascal. J'adore la programmation et l'intelligence artificielle.",
        "Que penses-tu de Python comme langage de programmation?",
        "As-tu déjà entendu parler des systèmes de mémoire hiérarchique pour les IA?",
        "Peux-tu m'expliquer comment fonctionne l'embedding de texte?",
        "Je travaille sur un projet d'IA qui utilise des réseaux de neurones. As-tu des conseils?"
    ]
    
    for i, user_input in enumerate(initial_conversations):
        logger.info(f"Conversation {i+1}: {user_input}")
        result = renee.process_input(user_input)
        logger.info(f"Réponse: {result['response']}")
        logger.info(f"Temps de traitement: {result['processing_time_ms']:.2f} ms")
        
        # Petite pause pour simuler le temps réel
        time.sleep(0.5)
    
    # Affichage des statistiques après la première phase
    renee.print_memory_statistics()
    
    # Phase 2: Simulation du passage du temps (5 minutes) et déclenchement de la condensation
    logger.info("Phase 2: Simulation du passage de 5 minutes et condensation niveau 2...")
    
    # Forcer la condensation au niveau 2
    condensation_result = renee.trigger_memory_condensation()
    
    # Affichage des résultats de condensation
    for level, result in condensation_result.items():
        if result and result.get("success"):
            count = result.get("count", 0)
            logger.info(f"{level}: {count} éléments générés")
            
            # Affichage du contenu des résumés générés si présent
            if level == "level2" and count > 0:
                for i, summary in enumerate(result.get("items", [])):
                    logger.info(f"Résumé {i+1}: {summary.content[:150]}...")
        else:
            logger.info(f"{level}: échec - {result.get('error', 'Erreur inconnue')}")
    
    # Phase 3: Conversations de suivi faisant référence aux conversations précédentes
    logger.info("Phase 3: Conversations de suivi avec référence au passé...")
    
    follow_up_conversations = [
        "Tu te souviens quand je t'ai parlé de mon projet sur les réseaux de neurones?",
        "Pourrais-tu me rappeler ce que tu m'as dit sur Python?",
        "J'aimerais en savoir plus sur les embeddings dont on a parlé précédemment.",
        "Peux-tu résumer ce dont nous avons discuté jusqu'à présent?"
    ]
    
    for i, user_input in enumerate(follow_up_conversations):
        logger.info(f"Conversation de suivi {i+1}: {user_input}")
        result = renee.process_input(user_input, include_memory=True)
        logger.info(f"Réponse: {result['response']}")
        logger.info(f"Taille du contexte mémoire: {result['memory_context_length']} caractères")
        
        # Petite pause
        time.sleep(0.5)
    
    # Affichage des statistiques finales
    stats = renee.print_memory_statistics()
    
    logger.info("=== Test terminé avec succès ===")
    
    return renee

if __name__ == "__main__":
    test_with_time_simulation()
