#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de test intégrant le système de mémoire hiérarchique corrigé avec Renée.
Ce script simule des interactions complètes avec Renée et teste sa capacité à
se souvenir d'informations à différents niveaux temporels.
"""

import os
import time
import random
import json
import shutil
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional

# Import de notre système de mémoire corrigé
from memory_fix import (
    FixedHierarchicalMemoryOrchestrator,
    UserPrompt, SystemResponse, Conversation, Summary, MetaConcept, KnowledgeItem
)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("renee_integration_test")

# Classe simulant le service LLM de Renée
class ReneeLLMService:
    """Service LLM de Renée simulé pour les tests."""
    
    def generate(self, prompt, max_tokens=500):
        """Génère une réponse simulée en fonction du prompt."""
        topics = {
            "voyage": [
                "J'adore voyager! Les destinations tropicales sont mes préférées.",
                "Paris est une ville magnifique avec beaucoup de culture et d'histoire.",
                "Pour préparer un bon voyage, il faut bien s'informer sur la destination."
            ],
            "cuisine": [
                "La cuisine italienne est connue pour ses pâtes, pizzas et son utilisation d'huile d'olive.",
                "Pour faire une bonne pâte à pizza, il faut de la farine, de l'eau, de la levure et du sel.",
                "Les bases de la cuisine française incluent les sauces mères et les techniques de cuisson précises."
            ],
            "technologie": [
                "L'intelligence artificielle révolutionne de nombreux domaines comme la médecine et les transports.",
                "Les ordinateurs quantiques utilisent les principes de la mécanique quantique pour effectuer des calculs.",
                "Le machine learning est une branche de l'IA qui permet aux systèmes d'apprendre à partir de données."
            ],
            "musique": [
                "Le jazz est né au début du 20ème siècle à La Nouvelle-Orléans.",
                "La musique classique comprend des périodes comme le baroque, le classique et le romantique.",
                "Le rock a émergé dans les années 1950 et a connu de nombreuses évolutions depuis."
            ]
        }
        
        # Détection du sujet principal du prompt
        prompt_lower = prompt.lower()
        detected_topic = None
        for topic in topics:
            if topic in prompt_lower:
                detected_topic = topic
                break
        
        if not detected_topic:
            # Sujet par défaut si aucun n'est détecté
            detected_topic = random.choice(list(topics.keys()))
        
        # Construction de la réponse en fonction du prompt et du contexte
        if "contexte" in prompt_lower and "précédentes conversations" in prompt_lower:
            return f"Je me souviens de nos précédentes conversations sur {detected_topic}. {random.choice(topics[detected_topic])}"
        
        return random.choice(topics[detected_topic])

# Classe simulant le système complet de Renée
class ReneeSystem:
    """Simulation du système complet de Renée intégrant notre mémoire hiérarchique corrigée."""
    
    def __init__(self, data_dir: str):
        """
        Initialise le système Renée simulé.
        
        Args:
            data_dir: Répertoire pour les données persistantes
        """
        self.data_dir = data_dir
        
        # Création du répertoire de données s'il n'existe pas
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialisation du service LLM
        self.llm_service = ReneeLLMService()
        
        # Initialisation du système de mémoire hiérarchique
        self.memory = FixedHierarchicalMemoryOrchestrator(
            data_dir=os.path.join(data_dir, "memory"),
            llm_service=self.llm_service
        )
        
        # Historique des conversations pour le test
        self.conversation_history = []
        
        logger.info(f"Système Renée initialisé avec mémoire hiérarchique dans: {data_dir}")
    
    def process_input(self, user_input: str, user_id: str = "test_user", 
                      include_memory: bool = True) -> Dict[str, Any]:
        """
        Traite une entrée utilisateur et génère une réponse.
        
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
        prompt = user_input
        
        # Ajout du contexte mémoire si demandé
        memory_context = ""
        if include_memory:
            memory_context = self.memory.compose_context(user_input)
            if memory_context:
                prompt = f"{memory_context}\n\nEntrée utilisateur: {user_input}"
        
        # Génération de la réponse
        response_text = self.llm_service.generate(prompt)
        
        end_time = datetime.now()
        
        # Métadonnées de la réponse
        response_metadata = {
            "timestamp": end_time.isoformat(),
            "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
            "include_memory": include_memory
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
            "memory_context_length": len(memory_context) if memory_context else 0
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

# Fonction de test principal
def test_renee_integration():
    """
    Teste l'intégration du système de mémoire hiérarchique avec Renée.
    """
    # Création du répertoire de test
    test_dir = "/tmp/renee_integration_test"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    # Initialisation du système Renée
    renee = ReneeSystem(data_dir=test_dir)
    
    logger.info("=== Début du test d'intégration de Renée avec la mémoire hiérarchique ===")
    
    # Phase 1: Conversations initiales sans références au passé
    logger.info("Phase 1: Conversations initiales...")
    
    initial_conversations = [
        "Bonjour, je m'appelle Pascal. Comment ça va?",
        "Peux-tu me parler de la cuisine italienne?",
        "Comment fonctionne un ordinateur quantique?",
        "Quelles sont les meilleures destinations pour voyager en été?",
        "J'aimerais apprendre le piano. As-tu des conseils?"
    ]
    
    for i, user_input in enumerate(initial_conversations):
        logger.info(f"Conversation {i+1}: {user_input}")
        result = renee.process_input(user_input, include_memory=True)
        logger.info(f"Réponse: {result['response']}")
        
        # Pause pour simuler l'écoulement du temps
        time.sleep(0.5)
    
    # Affichage des statistiques
    renee.print_memory_statistics()
    
    # Phase 2: Déclenchement manuel de la condensation
    logger.info("Phase 2: Déclenchement de la condensation de mémoire...")
    condensation_result = renee.trigger_memory_condensation()
    
    # Affichage des résultats de condensation
    for level, result in condensation_result.items():
        if result and result.get("success"):
            logger.info(f"{level}: {result.get('count')} éléments générés")
        else:
            logger.info(f"{level}: échec - {result.get('error', 'Erreur inconnue')}")
    
    # Affichage des statistiques après condensation
    renee.print_memory_statistics()
    
    # Phase 3: Conversations avec références au passé
    logger.info("Phase 3: Conversations faisant référence au passé...")
    
    follow_up_conversations = [
        "Tu te souviens quand on a parlé de cuisine italienne?",
        "Est-ce que tu peux me rappeler ce que tu m'as dit sur les ordinateurs quantiques?",
        "Je te remercie pour tes conseils sur les destinations de voyage.",
        "Quel était ton conseil principal pour apprendre le piano?"
    ]
    
    for i, user_input in enumerate(follow_up_conversations):
        logger.info(f"Conversation de suivi {i+1}: {user_input}")
        result = renee.process_input(user_input, include_memory=True)
        logger.info(f"Réponse: {result['response']}")
        logger.info(f"Taille du contexte mémoire: {result['memory_context_length']} caractères")
        
        # Pause pour simuler l'écoulement du temps
        time.sleep(0.5)
    
    # Affichage des statistiques finales
    renee.print_memory_statistics()
    
    logger.info("=== Test d'intégration terminé avec succès ===")
    
    return renee  # Renvoie l'instance pour d'autres tests si nécessaire

if __name__ == "__main__":
    test_renee_integration()
