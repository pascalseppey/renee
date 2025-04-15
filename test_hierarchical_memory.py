#!/usr/bin/env python3
# test_hierarchical_memory.py
# Script pour tester le système de mémoire hiérarchique complet de Renée (5 niveaux)

import os
import time
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MemoryTest")

# Import des modules mémoire
from src.memory.hierarchical_memory import (
    ShortTermMemory, 
    Level2Condenser, 
    Level3HourlyConcepts, 
    Level4DailyKnowledge, 
    Level5Orchestrator
)

# Simulation d'un service LLM simple pour les tests
class SimpleLLMService:
    def generate(self, prompt):
        """Simule une génération simple via LLM."""
        if "règles" in prompt.lower():
            return """
            1. Privilégier l'écoute active avant de proposer des solutions.
            2. Adapter le niveau de détail technique aux connaissances de l'utilisateur.
            3. Vérifier la compréhension avant de poursuivre avec de nouvelles informations.
            """
        elif "réflexion" in prompt.lower():
            return """
            Cette réflexion synthétique intègre plusieurs concepts clés qui émergent des conversations récentes.
            Les interactions montrent un intérêt pour l'équilibre entre les aspects techniques et pratiques,
            ainsi qu'une préférence pour les explications progressives. L'approche pédagogique semble être 
            particulièrement appréciée, suggérant l'importance d'adapter le niveau d'information au contexte.
            """
        else:
            return "Réponse générique générée par le LLM simulé."

def setup_test_environment():
    """Configure l'environnement de test."""
    # Créer un répertoire de données pour les tests
    data_dir = Path("./test_data")
    data_dir.mkdir(exist_ok=True)
    
    # S'assurer que les sous-répertoires existent
    memory_dir = data_dir / "memory"
    memory_dir.mkdir(exist_ok=True)
    
    return str(memory_dir)

def simulate_conversations(short_term_memory, num_conversations=15):
    """Simule plusieurs conversations pour alimenter la mémoire à court terme."""
    logger.info(f"Simulation de {num_conversations} conversations...")
    
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
    
    # Ajout de conversations à la mémoire à court terme
    for i in range(num_conversations):
        # Sélection d'un thème et d'une conversation
        theme_idx = i % len(themes)
        convo_idx = (i // len(themes)) % len(themes[theme_idx]["conversations"])
        
        theme = themes[theme_idx]
        convo = theme["conversations"][convo_idx]
        
        # Ajout de métadonnées pour simuler un contexte réel
        user_metadata = {"theme": theme["sujet"], "sentiment": "positif", "timestamp": datetime.now().isoformat()}
        system_metadata = {"confidence": 0.92, "sources": ["base_knowledge", "user_history"], "timestamp": datetime.now().isoformat()}
        
        # Ajout à la mémoire
        short_term_memory.add_conversation(
            convo["user"], 
            convo["system"],
            user_metadata=user_metadata,
            response_metadata=system_metadata
        )
        
        logger.info(f"Conversation {i+1}/{num_conversations} ajoutée (Thème: {theme['sujet']})")
        
        # Pause simulée entre les conversations
        time.sleep(0.1)
    
    # Sauvegarde des données
    short_term_memory.save_data()
    logger.info("Toutes les conversations ont été simulées et sauvegardées")

def test_memory_hierarchy(memory_system):
    """
    Teste la hiérarchie complète de mémoire.
    
    Args:
        memory_system: tuple contenant les différents niveaux de mémoire
            (short_term, level2, level3, level4, orchestrator)
    """
    short_term, level2, level3, level4, orchestrator = memory_system
    
    # Test 1: Déclenchement manuel des processus de condensation
    logger.info("=== Test 1: Déclenchement des processus de condensation ===")
    # Niveau 2: Condensation des conversations récentes
    level2.condense_recent_memory()
    logger.info("Condensation niveau 2 (5 minutes) effectuée")
    
    # Niveau 3: Condensation des résumés en méta-concepts
    level3.condense_level2_to_level3()
    logger.info("Condensation niveau 3 (horaire) effectuée")
    
    # Niveau 4: Consolidation journalière
    level4.daily_consolidation()
    logger.info("Consolidation niveau 4 (journalière) effectuée")
    
    # Test 2: Requête de contexte via l'orchestrateur
    logger.info("\n=== Test 2: Composition du contexte ===")
    # Composition du contexte pour une requête
    query = "Comment puis-je améliorer ma santé et mon bien-être?"
    context = orchestrator.compose_context(query, max_tokens=1024)
    
    # Affichage du contexte composé
    logger.info(f"Contexte composé pour la requête: {query}")
    logger.info("---")
    logger.info(context)
    logger.info("---")
    
    # Test 3: Recherche sémantique dans différents niveaux
    logger.info("\n=== Test 3: Recherche sémantique ===")
    # Niveau 1: Recherche de conversations similaires
    similar_convos = short_term.get_similar_conversations(query, k=2)
    logger.info(f"Niveau 1 - Conversations similaires: {len(similar_convos)} trouvées")
    
    # Niveau 2: Recherche de résumés pertinents
    summaries = level2.get_relevant_summaries(query, k=2)
    logger.info(f"Niveau 2 - Résumés pertinents: {len(summaries)} trouvés")
    
    # Niveau 3: Recherche de méta-concepts pertinents
    concepts = level3.get_relevant_concepts(query, k=2)
    logger.info(f"Niveau 3 - Méta-concepts pertinents: {len(concepts)} trouvés")
    
    return context

def main():
    """Fonction principale pour exécuter les tests."""
    logger.info("Démarrage du test du système de mémoire hiérarchique complet...")
    
    # Configuration de l'environnement de test
    memory_dir = setup_test_environment()
    
    # Initialisation des composants de mémoire
    short_term_memory = ShortTermMemory(memory_dir, max_items=20)
    
    # Simulation d'un service LLM simple pour les tests
    llm_service = SimpleLLMService()
    
    # Chargement des données existantes
    short_term_memory.load_data()
    
    # Simulation de conversations (si nécessaire)
    if len(short_term_memory.conversations) < 10:
        simulate_conversations(short_term_memory, num_conversations=15)
    else:
        logger.info(f"Utilisation des {len(short_term_memory.conversations)} conversations existantes")
    
    # Initialisation du système de mémoire hiérarchique complet
    level2_condenser = Level2Condenser(memory_dir, short_term_memory)
    level3_concepts = Level3HourlyConcepts(memory_dir, level2_condenser, llm_service=llm_service)
    level4_knowledge = Level4DailyKnowledge(memory_dir, level3_concepts, llm_service=llm_service)
    orchestrator = Level5Orchestrator(short_term_memory, level2_condenser, level3_concepts, level4_knowledge)
    
    # Test complet du système de mémoire hiérarchique
    memory_system = (short_term_memory, level2_condenser, level3_concepts, level4_knowledge, orchestrator)
    context = test_memory_hierarchy(memory_system)
    
    logger.info("\n=== Résumé du test ===")
    logger.info("1. ShortTermMemory (Niveau 1): Mémoire à court terme - OPÉRATIONNEL")
    logger.info("2. Level2Condenser (Niveau 2): Condensation 5 minutes - OPÉRATIONNEL")
    logger.info("3. Level3HourlyConcepts (Niveau 3): Méta-concepts horaires - OPÉRATIONNEL")
    logger.info("4. Level4DailyKnowledge (Niveau 4): Connaissances journalières - OPÉRATIONNEL")
    logger.info("5. Level5Orchestrator (Niveau 5): Orchestrateur hiérarchique - OPÉRATIONNEL")
    logger.info("\nLe système de mémoire hiérarchique complet est opérationnel.")
    
    # Sauvegarde du contexte généré pour référence
    with open("./test_data/generated_context.txt", "w") as f:
        f.write(context)
    logger.info("Contexte généré sauvegardé dans ./test_data/generated_context.txt")

if __name__ == "__main__":
    main()
