#!/usr/bin/env python3
# memory_system_usage.py
# Exemples d'utilisation du système de mémoire hiérarchique

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MemoryExamples")

# Assurez-vous que le répertoire examples existe
Path("./examples").mkdir(exist_ok=True)

# Import des modules mémoire
from src.memory.hierarchical_memory import (
    ShortTermMemory, 
    Level2Condenser, 
    Level3HourlyConcepts, 
    Level4DailyKnowledge, 
    Level5Orchestrator
)

# Fonction utilitaire pour créer une ligne de séparation dans les logs
def log_separator(title):
    separator = "=" * 50
    logger.info(f"\n{separator}\n{title}\n{separator}")

# Simuler un service LLM pour les tests
class SimpleLLMService:
    def generate(self, prompt):
        """Simule une génération de texte via un LLM."""
        if "résumé" in prompt.lower():
            return "Résumé: L'utilisateur s'intéresse à l'intelligence artificielle et à son fonctionnement."
        elif "concept" in prompt.lower():
            return "Concept: Intelligence Artificielle - Discipline qui cherche à créer des systèmes capables d'apprendre et de raisonner comme des humains."
        elif "règle" in prompt.lower():
            return "1. Adapter les explications au niveau technique de l'utilisateur.\n2. Commencer par les concepts de base avant d'aborder les détails complexes."
        else:
            return "Réponse générique du LLM."

def initialize_memory_system(data_dir):
    """Initialise le système de mémoire hiérarchique complet."""
    log_separator("Initialisation du système de mémoire")
    
    # Simuler un service LLM
    llm_service = SimpleLLMService()
    
    # Niveau 1: Mémoire à court terme
    logger.info("Initialisation de la mémoire à court terme (Niveau 1)")
    short_term_memory = ShortTermMemory(data_dir, max_items=20)
    
    # Niveau 2: Condensateur (5 minutes)
    logger.info("Initialisation du condensateur de résumés (Niveau 2)")
    level2_condenser = Level2Condenser(data_dir, short_term_memory)
    
    # Niveau 3: Concepts horaires
    logger.info("Initialisation du générateur de méta-concepts (Niveau 3)")
    level3_concepts = Level3HourlyConcepts(data_dir, level2_condenser, llm_service=llm_service)
    
    # Niveau 4: Connaissances journalières
    logger.info("Initialisation du consolidateur journalier (Niveau 4)")
    level4_knowledge = Level4DailyKnowledge(data_dir, level3_concepts, llm_service=llm_service)
    
    # Niveau 5: Orchestrateur
    logger.info("Initialisation de l'orchestrateur (Niveau 5)")
    orchestrator = Level5Orchestrator(short_term_memory, level2_condenser, level3_concepts, level4_knowledge)
    
    logger.info("Système de mémoire hiérarchique initialisé avec succès")
    
    return (short_term_memory, level2_condenser, level3_concepts, level4_knowledge, orchestrator)

def example_add_conversations(short_term_memory):
    """Exemple: Ajout de conversations à la mémoire à court terme."""
    log_separator("Exemple 1: Ajout de conversations")
    
    # Exemple de conversations
    conversations = [
        {
            "user": "Comment fonctionne l'intelligence artificielle?",
            "system": "L'intelligence artificielle repose sur des algorithmes qui permettent aux machines d'apprendre à partir de données et d'améliorer leurs performances au fil du temps."
        },
        {
            "user": "Qu'est-ce que l'apprentissage profond?",
            "system": "L'apprentissage profond est une branche de l'IA qui utilise des réseaux de neurones à plusieurs couches pour analyser des données complexes comme les images ou le texte."
        },
        {
            "user": "Quelles sont les applications concrètes de l'IA?",
            "system": "L'IA est utilisée dans de nombreux domaines: assistants vocaux, voitures autonomes, diagnostic médical, traduction automatique, recommandations personnalisées, etc."
        },
        {
            "user": "Est-ce que l'IA peut être dangereuse?",
            "system": "Comme toute technologie puissante, l'IA comporte des risques: biais dans les décisions automatisées, perte d'emplois, surveillance de masse, ou perte de contrôle si mal conçue."
        }
    ]
    
    # Ajout des conversations
    for i, convo in enumerate(conversations):
        logger.info(f"Ajout de la conversation {i+1}/{len(conversations)}")
        short_term_memory.add_conversation(
            convo["user"],
            convo["system"],
            user_metadata={"interaction_id": i, "topic": "AI"},
            response_metadata={"confidence": 0.95, "sources": ["knowledge_base"]}
        )
    
    # Vérification
    recent = short_term_memory.get_recent_conversations(limit=10)
    logger.info(f"Nombre de conversations en mémoire: {len(recent)}")
    
    return recent

def example_similarity_search(short_term_memory):
    """Exemple: Recherche de similarité dans la mémoire à court terme."""
    log_separator("Exemple 2: Recherche de similarité")
    
    # Requête de recherche
    query = "Quels sont les dangers de l'intelligence artificielle?"
    logger.info(f"Recherche de conversations similaires à: '{query}'")
    
    # Recherche
    similar_convos = short_term_memory.get_similar_conversations(query, k=2)
    
    # Affichage des résultats
    logger.info(f"Trouvé {len(similar_convos)} conversations similaires:")
    for i, convo in enumerate(similar_convos):
        logger.info(f"Résultat {i+1}:")
        logger.info(f"- User: {convo.get('user_input', convo.get('prompt', {}).get('content', 'N/A'))}")
        logger.info(f"- System: {convo.get('system_response', convo.get('response', {}).get('content', 'N/A'))}")
    
    return similar_convos

def example_condensation(level2_condenser):
    """Exemple: Condensation de la mémoire à court terme en résumés."""
    log_separator("Exemple 3: Condensation de niveau 2")
    
    # Déclenchement manuel de la condensation
    logger.info("Déclenchement de la condensation de niveau 2 (normalement toutes les 5 minutes)")
    summaries = level2_condenser.condense_recent_memory()
    
    # Affichage des résultats
    if summaries:
        logger.info(f"Générés {len(summaries)} résumés:")
        for i, summary in enumerate(summaries):
            logger.info(f"Résumé {i+1}: {summary.content[:100]}...")
    else:
        logger.info("Aucun résumé généré (pas assez de données ou groupes trop petits)")
    
    return summaries

def example_meta_concepts(level3_concepts):
    """Exemple: Génération de méta-concepts à partir de résumés."""
    log_separator("Exemple 4: Génération de méta-concepts (niveau 3)")
    
    # Déclenchement manuel de la condensation de niveau 3
    logger.info("Déclenchement de la génération de méta-concepts (normalement toutes les heures)")
    concepts = level3_concepts.condense_level2_to_level3()
    
    # Affichage des résultats
    if concepts:
        logger.info(f"Générés {len(concepts)} méta-concepts:")
        for i, concept in enumerate(concepts):
            logger.info(f"Concept {i+1}: {concept.name} - {concept.description[:100]}...")
    else:
        logger.info("Aucun méta-concept généré (pas assez de résumés)")
    
    return concepts

def example_daily_knowledge(level4_knowledge):
    """Exemple: Consolidation journalière des méta-concepts."""
    log_separator("Exemple 5: Consolidation journalière (niveau 4)")
    
    # Déclenchement manuel de la consolidation journalière
    logger.info("Déclenchement de la consolidation journalière (normalement une fois par jour)")
    level4_knowledge.daily_consolidation()
    
    # Affichage des résultats
    rules = level4_knowledge.derived_rules
    knowledge = level4_knowledge.long_term_knowledge
    
    logger.info(f"Résultats de la consolidation:")
    logger.info(f"- Règles dérivées: {len(rules)}")
    logger.info(f"- Connaissances à long terme: {len(knowledge)}")
    
    # Afficher quelques exemples
    if rules:
        rule_id = list(rules.keys())[0]
        logger.info(f"Exemple de règle: {rules[rule_id].get('content', 'N/A')}")
    
    if knowledge:
        knowledge_id = list(knowledge.keys())[0]
        logger.info(f"Exemple de connaissance: {knowledge[knowledge_id].get('content', 'N/A')}")
    
    return (rules, knowledge)

def example_context_composition(orchestrator):
    """Exemple: Composition du contexte par l'orchestrateur."""
    log_separator("Exemple 6: Composition du contexte (niveau 5)")
    
    # Requête utilisateur
    query = "Comment puis-je utiliser l'IA pour améliorer mon entreprise?"
    logger.info(f"Composition d'un contexte pour la requête: '{query}'")
    
    # Composition du contexte
    context = orchestrator.compose_context(query, max_tokens=1024)
    
    # Affichage du résultat
    logger.info(f"Contexte généré ({len(context)} caractères):")
    logger.info("---")
    logger.info(context[:300] + "..." if len(context) > 300 else context)
    logger.info("---")
    
    return context

def main():
    """Fonction principale pour exécuter les exemples."""
    try:
        # Chemin de données
        data_dir = "./examples/memory_data"
        Path(data_dir).mkdir(exist_ok=True, parents=True)
        
        # Initialisation du système
        memory_system = initialize_memory_system(data_dir)
        short_term, level2, level3, level4, orchestrator = memory_system
        
        # Exemples d'utilisation
        example_add_conversations(short_term)
        example_similarity_search(short_term)
        example_condensation(level2)
        example_meta_concepts(level3)
        example_daily_knowledge(level4)
        example_context_composition(orchestrator)
        
        log_separator("Exemples terminés")
        logger.info("Tous les exemples d'utilisation ont été exécutés avec succès.")
        logger.info("Ces exemples illustrent le fonctionnement du système de mémoire hiérarchique.")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution des exemples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
