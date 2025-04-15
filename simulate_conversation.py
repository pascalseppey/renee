#!/usr/bin/env python3
# simulate_conversation.py
# Script pour simuler une conversation avec Renée et tester sa mémoire

import os
import time
import uuid
import logging
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ConversationTest")

# Répertoire de données pour les tests
CONVO_DIR = Path("./conversation_test")
CONVO_DIR.mkdir(exist_ok=True)
MEMORY_DIR = CONVO_DIR / "memory"
MEMORY_DIR.mkdir(exist_ok=True, parents=True)

# Importer les classes nécessaires depuis le système de mémoire hiérarchique
try:
    from src.memory.hierarchical_memory import (
        ShortTermMemory, UserPrompt, SystemResponse, Conversation,
        Level2Condenser, Level3HourlyConcepts, Level4DailyKnowledge, Level5Orchestrator,
        Summary, MetaConcept
    )
    USE_REAL_MEMORY = True
    logger.info("Utilisation du vrai système de mémoire hiérarchique")
except ImportError:
    logger.warning("Impossible d'importer le système de mémoire réel, utilisation d'une simulation")
    USE_REAL_MEMORY = False

# Classe pour simuler Renée
class Renee:
    """Simulation de Renée avec son système de mémoire"""
    
    def __init__(self):
        """Initialisation de Renée et de son système de mémoire"""
        self.short_term_memory = self.initialize_memory()
        self.conversation_history = []
        
        if USE_REAL_MEMORY:
            # Initialisation du système de mémoire hiérarchique complet
            self.level2_condenser = Level2Condenser(str(MEMORY_DIR), self.short_term_memory)
            self.level3_concepts = Level3HourlyConcepts(str(MEMORY_DIR), self.level2_condenser, llm_service=None)
            self.level4_knowledge = Level4DailyKnowledge(str(MEMORY_DIR), self.level3_concepts, llm_service=None)
            self.orchestrator = Level5Orchestrator(
                self.short_term_memory, 
                self.level2_condenser, 
                self.level3_concepts, 
                self.level4_knowledge
            )
        
        logger.info("Renée initialisée avec son système de mémoire")
    
    def initialize_memory(self):
        """Initialisation de la mémoire (réelle ou simulée)"""
        if USE_REAL_MEMORY:
            # Utilisation de la vraie mémoire à court terme
            memory = ShortTermMemory(str(MEMORY_DIR), max_items=20)
        else:
            # Utilisation d'une mémoire simulée
            from test_memory_levels import MockShortTermMemory
            memory = MockShortTermMemory()
        
        return memory
    
    def process_input(self, user_input: str) -> str:
        """Traite l'entrée utilisateur et génère une réponse"""
        # Récupération du contexte pour la réponse
        context = self.get_conversation_context(user_input)
        
        # Simulation d'une réponse basée sur le contexte
        response = self.generate_response(user_input, context)
        
        # Ajout à la mémoire
        self.add_to_memory(user_input, response)
        
        return response
    
    def get_conversation_context(self, user_input: str) -> str:
        """Récupère le contexte pertinent pour la conversation"""
        if USE_REAL_MEMORY and hasattr(self, 'orchestrator'):
            # Utilisation de l'orchestrateur pour composer le contexte
            context = self.orchestrator.compose_context(user_input, max_tokens=1024)
        else:
            # Composition manuelle d'un contexte simplifié
            similar_convos = self.short_term_memory.get_similar_conversations(user_input, k=2)
            
            context_parts = []
            context_parts.append("## Conversations récentes pertinentes")
            
            for convo in similar_convos:
                user_text = convo.get('user_input', convo.get('prompt', {}).get('content', 'N/A'))
                system_text = convo.get('system_response', convo.get('response', {}).get('content', 'N/A'))
                
                context_parts.append(f"Utilisateur: {user_text}")
                context_parts.append(f"Renée: {system_text}")
                context_parts.append("")
            
            context = "\n".join(context_parts)
        
        return context
    
    def generate_response(self, user_input: str, context: str) -> str:
        """Génère une réponse basée sur l'entrée utilisateur et le contexte"""
        # Simulation de réponses en fonction de l'entrée utilisateur
        # Dans un système réel, cette fonction appellerait un LLM avec le contexte
        
        user_input_lower = user_input.lower()
        
        if "bonjour" in user_input_lower or "salut" in user_input_lower:
            return "Bonjour ! Je suis Renée. Comment puis-je vous aider aujourd'hui ?"
        
        elif "comment" in user_input_lower and "tu" in user_input_lower:
            return "Je vais bien, merci ! Je suis ici pour vous aider et répondre à vos questions."
        
        elif "souviens" in user_input_lower or "rappelle" in user_input_lower:
            # Vérifier le contexte pour trouver des informations précédentes
            if context and len(context) > 20:  # Si nous avons un contexte significatif
                return f"Oui, je me souviens de notre conversation. Nous avons parlé de {self.extract_topic_from_context(context)}."
            else:
                return "Je n'ai pas beaucoup d'informations sur nos conversations précédentes."
        
        elif "nom" in user_input_lower:
            return "Je m'appelle Renée, je suis une intelligence artificielle conversationnelle avec un système de mémoire hiérarchique."
        
        elif "mémoire" in user_input_lower or "système" in user_input_lower:
            return """Mon système de mémoire est organisé en 5 niveaux:
1. Mémoire à court terme: conversations récentes
2. Résumés (toutes les 5 minutes)
3. Méta-concepts (toutes les heures)
4. Connaissances consolidées (tous les jours)
5. Orchestrateur pour assembler le contexte optimal"""
        
        else:
            # Réponse générique avec un peu d'information du contexte
            if context and len(context) > 100:
                return f"D'après ce que nous avons discuté, je pense pouvoir vous aider sur ce sujet. Pouvez-vous préciser votre question ?"
            else:
                return "Je comprends votre question. Pouvez-vous me donner plus de détails pour que je puisse mieux vous aider ?"
    
    def extract_topic_from_context(self, context: str) -> str:
        """Extrait le sujet principal du contexte"""
        # Analyse simplifiée - dans un système réel, cela serait fait par un modèle NLP
        if "ia" in context.lower() or "intelligence" in context.lower():
            return "l'intelligence artificielle"
        elif "mémoire" in context.lower() or "système" in context.lower():
            return "mon système de mémoire"
        elif "apprentissage" in context.lower():
            return "l'apprentissage automatique"
        else:
            return "différents sujets"
    
    def add_to_memory(self, user_input: str, response: str) -> None:
        """Ajoute la conversation à la mémoire"""
        # Ajout à la mémoire à court terme
        self.short_term_memory.add_conversation(
            user_input,
            response,
            user_metadata={"timestamp": datetime.now().isoformat()},
            response_metadata={"timestamp": datetime.now().isoformat()}
        )
        
        # Ajout à l'historique local
        self.conversation_history.append({
            "user": user_input,
            "renee": response,
            "timestamp": datetime.now()
        })
        
        logger.info(f"Conversation ajoutée à la mémoire: '{user_input[:30]}...' -> '{response[:30]}...'")
    
    def trigger_memory_condensation(self):
        """Déclenche manuellement les processus de condensation de la mémoire"""
        if not USE_REAL_MEMORY:
            logger.warning("Condensation non disponible en mode simulation")
            return
        
        logger.info("Déclenchement manuel des processus de condensation de la mémoire")
        
        # Niveau 2: Condensation des conversations en résumés
        summaries = self.level2_condenser.condense_recent_memory()
        logger.info(f"Condensation niveau 2: {len(summaries) if summaries else 0} résumés générés")
        
        # Niveau 3: Génération de méta-concepts
        self.level3_concepts.condense_level2_to_level3()
        logger.info(f"Condensation niveau 3: {len(self.level3_concepts.meta_concepts)} méta-concepts")
        
        # Niveau 4: Consolidation journalière
        self.level4_knowledge.daily_consolidation()
        logger.info(f"Consolidation niveau 4: {len(self.level4_knowledge.long_term_knowledge)} connaissances")

def simulate_conversation(minutes=4):
    """Simule une conversation avec Renée pendant plusieurs minutes"""
    logger.info(f"Démarrage d'une simulation de conversation de {minutes} minutes")
    
    # Initialisation de Renée
    renee = Renee()
    
    # Questions préparées pour la simulation
    questions = [
        "Bonjour Renée, comment vas-tu aujourd'hui ?",
        "Peux-tu me parler de ton système de mémoire ?",
        "Comment fonctionne ton niveau de mémoire à court terme ?",
        "Quels sont les avantages du niveau 3 de ta mémoire ?",
        "Comment ton orchestrateur utilise-t-il les différents niveaux ?",
        "Est-ce que tu te souviens de quoi on parlait au début de notre conversation ?",
        "Peux-tu me dire quel est ton nom complet ?",
        "Comment la mémoire hiérarchique améliore-t-elle tes capacités ?",
        "Quelles sont les limitations actuelles de ton système de mémoire ?",
        "Te souviens-tu de ce que je t'ai demandé sur l'orchestrateur ?",
        "Comment gères-tu l'oubli dans ton système ?",
        "Quel est le rôle de la consolidation journalière ?",
        "Peux-tu récapituler les sujets dont nous avons discuté aujourd'hui ?"
    ]
    
    # Calcul du temps disponible par question
    seconds_per_question = (minutes * 60) / len(questions)
    seconds_per_question = min(seconds_per_question, 30)  # Maximum 30 secondes par question
    
    # Simulation de la conversation
    for i, question in enumerate(questions):
        logger.info(f"\n--- Question {i+1}/{len(questions)} ---")
        
        # Affichage de la question
        logger.info(f"Utilisateur: {question}")
        
        # Traitement par Renée
        start_time = time.time()
        response = renee.process_input(question)
        
        # Affichage de la réponse
        logger.info(f"Renée: {response}")
        
        # Déclenchement périodique de la condensation de la mémoire
        if i > 0 and i % 4 == 0:
            logger.info("Déclenchement de la condensation de la mémoire")
            renee.trigger_memory_condensation()
        
        # Attente avant la prochaine question (sauf pour la dernière)
        if i < len(questions) - 1:
            # Calcul du temps d'attente restant
            elapsed = time.time() - start_time
            wait_time = max(0, seconds_per_question - elapsed)
            
            if wait_time > 0:
                logger.info(f"Attente de {wait_time:.1f} secondes avant la prochaine question...")
                time.sleep(wait_time)
    
    # Test final de mémoire
    logger.info("\n=== Test final de mémoire ===")
    memory_test = "Peux-tu me dire de quoi nous avons parlé pendant toute cette conversation ?"
    logger.info(f"Utilisateur: {memory_test}")
    
    # Déclenchement de la condensation avant le test final
    renee.trigger_memory_condensation()
    
    # Réponse finale
    final_response = renee.process_input(memory_test)
    logger.info(f"Renée: {final_response}")
    
    logger.info("\nFin de la simulation de conversation")

if __name__ == "__main__":
    try:
        # Simulation d'une conversation de 4 minutes
        simulate_conversation(minutes=4)
    except Exception as e:
        logger.error(f"Erreur lors de la simulation: {e}")
        import traceback
        traceback.print_exc()
