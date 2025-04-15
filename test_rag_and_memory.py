#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de test pour évaluer la capacité de Renée à utiliser conjointement
son système de mémoire hiérarchique et son système RAG.
"""

import os
import time
import shutil
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import des modules nécessaires
from memory_fix import (
    FixedHierarchicalMemoryOrchestrator,
    UserPrompt, SystemResponse, Conversation
)
from deepseek_service import DeepSeekService

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag_memory_test")

class ReneeRAGMemoryTester:
    """Classe pour tester l'utilisation conjointe du RAG et de la mémoire par Renée."""
    
    def __init__(self, data_dir: str):
        """
        Initialise le testeur RAG + mémoire de Renée.
        
        Args:
            data_dir: Répertoire pour les données persistantes
        """
        self.data_dir = data_dir
        
        # Création du répertoire de données s'il n'existe pas
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialisation du service LLM
        self.llm_service = DeepSeekService(model_name="deepseek-wordpress:latest")
        
        # Initialisation du système de mémoire hiérarchique
        self.memory = FixedHierarchicalMemoryOrchestrator(
            data_dir=os.path.join(data_dir, "memory"),
            llm_service=self.llm_service
        )
        
        # Simulation d'une base de connaissances RAG
        self.rag_knowledge_base = self._create_sample_rag_knowledge()
        
        # Historique des conversations
        self.conversation_history = []
        
        logger.info(f"Testeur RAG+Mémoire de Renée initialisé dans: {data_dir}")
    
    def _create_sample_rag_knowledge(self) -> Dict[str, str]:
        """
        Crée une base de connaissances RAG simulée.
        
        Returns:
            Dictionnaire avec des clés thématiques et des valeurs de contenu
        """
        knowledge_base = {
            "rag_definition": """
            Le RAG (Retrieval-Augmented Generation) est une architecture d'IA qui combine la récupération
            d'informations à partir d'une base de connaissances externe avec la génération de texte par un LLM.
            Cette approche améliore la précision factuelle des réponses en s'appuyant sur des sources vérifiées
            plutôt que sur les seules connaissances internes du modèle.
            """,
            
            "rag_components": """
            Un système RAG comporte généralement quatre composants principaux:
            1. Une base de données vectorielle pour stocker les documents et leurs embeddings
            2. Un modèle d'embedding pour convertir les textes en vecteurs numériques
            3. Un mécanisme de recherche pour trouver les documents pertinents
            4. Un LLM qui génère des réponses en utilisant les documents récupérés comme contexte
            """,
            
            "rag_benefits": """
            Les avantages principaux du RAG sont:
            - Précision factuelle accrue grâce à l'accès à des informations externes vérifiées
            - Réduction des hallucinations du modèle de langage
            - Capacité à accéder à des informations récentes ou spécifiques non présentes dans l'entraînement du LLM
            - Transparence quant aux sources d'information utilisées pour générer une réponse
            """,
            
            "memory_vs_rag": """
            Différences entre mémoire hiérarchique et RAG:
            
            Mémoire hiérarchique:
            - Stocke l'historique des conversations avec l'utilisateur
            - Organise l'information en niveaux d'abstraction croissante
            - Permet de maintenir le contexte sur de longues périodes
            - Aide à la personnalisation des réponses
            
            Système RAG:
            - Accède à des connaissances externes factuelles
            - Ne dépend pas de l'historique de conversation
            - Fournit des informations objectives et vérifiables
            - Peut être mis à jour avec de nouvelles informations
            """,
            
            "vector_databases": """
            Les bases de données vectorielles les plus utilisées dans les systèmes RAG sont:
            - Pinecone: service cloud optimisé pour la recherche vectorielle à grande échelle
            - Weaviate: base de données vectorielle open-source avec capacités GraphQL
            - Milvus: solution open-source hautement évolutive
            - Qdrant: optimisée pour les requêtes filtrées et les métadonnées
            - ChromaDB: solution légère pour le prototypage rapide
            - FAISS (Facebook AI Similarity Search): bibliothèque efficace pour la recherche de similarité
            """,
            
            "embedding_models": """
            Modèles d'embedding couramment utilisés:
            - OpenAI: text-embedding-ada-002 (1536 dimensions)
            - Sentence-BERT: modèles comme 'all-MiniLM-L6-v2' ou 'all-mpnet-base-v2'
            - BGE: BAAI/bge-large-en-v1.5 (performant pour les requêtes en anglais)
            - E5: intfloat/e5-large-v2 (bon équilibre performance/coût)
            - GTE: Google/gte-large (efficace pour la recherche sémantique)
            """,
            
            "rag_limitations": """
            Limitations des systèmes RAG:
            - La qualité des réponses dépend fortement de la qualité des documents dans la base de connaissances
            - Des latences accrues dues au temps nécessaire pour récupérer les documents pertinents
            - Difficulté à pondérer l'importance relative des différentes sources d'information
            - Problèmes potentiels lorsque les documents récupérés contiennent des informations contradictoires
            - Coûts de stockage et de calcul plus élevés par rapport à un LLM seul
            """,
            
            "renee_architecture": """
            L'architecture de Renée combine:
            
            1. Un système de mémoire hiérarchique à 4 niveaux:
               - Niveau 1: Mémoire à court terme (conversations récentes)
               - Niveau 2: Condensateur (résumés de conversations)
               - Niveau 3: Générateur de concepts (patterns abstraits)
               - Niveau 4: Base de connaissances (informations consolidées)
            
            2. Un système RAG intégré qui:
               - Utilise OpenAI pour générer des embeddings précis
               - S'appuie sur une base de données vectorielle pour stocker les documents
               - Récupère automatiquement les informations pertinentes pour chaque requête
               - Enrichit les réponses avec des informations factuelles externes
            """
        }
        
        return knowledge_base
    
    def _retrieve_rag_context(self, query: str, top_k: int = 2) -> str:
        """
        Simule la récupération d'information du système RAG.
        
        Args:
            query: Requête utilisateur
            top_k: Nombre de résultats à retourner
            
        Returns:
            Contexte RAG formaté
        """
        # Simulation simple de recherche par mots-clés
        query_lower = query.lower()
        scored_results = []
        
        for key, content in self.rag_knowledge_base.items():
            # Calcul d'un score basique basé sur le nombre de mots-clés correspondants
            score = 0
            for word in query_lower.split():
                if len(word) > 3 and word in content.lower():  # Ignorer les mots courts
                    score += 1
            
            if score > 0:
                scored_results.append((key, content, score))
        
        # Trier par score décroissant et prendre les top_k
        scored_results.sort(key=lambda x: x[2], reverse=True)
        top_results = scored_results[:top_k]
        
        # Formatage du contexte RAG
        if not top_results:
            return ""
            
        rag_context = "\n\n".join([
            f"[Document: {key.replace('_', ' ').title()}]\n{content.strip()}"
            for key, content, _ in top_results
        ])
        
        return rag_context
    
    def interact(self, user_input: str, include_memory: bool = True, include_rag: bool = True) -> Dict[str, Any]:
        """
        Interagit avec Renée en utilisant la mémoire et le RAG.
        
        Args:
            user_input: Message de l'utilisateur
            include_memory: Si True, utilise la mémoire pour enrichir la réponse
            include_rag: Si True, utilise le RAG pour enrichir la réponse
            
        Returns:
            Dictionnaire contenant la réponse et les métriques
        """
        start_time = datetime.now()
        
        # Métadonnées de l'entrée utilisateur
        user_metadata = {
            "timestamp": start_time.isoformat(),
            "test_id": f"test_{len(self.conversation_history)}"
        }
        
        # Récupération du contexte mémoire
        memory_context = ""
        if include_memory:
            memory_context = self.memory.compose_context(user_input)
        
        # Récupération du contexte RAG
        rag_context = ""
        if include_rag:
            rag_context = self._retrieve_rag_context(user_input)
        
        # Génération de la réponse
        if include_memory or include_rag:
            response_text = self.llm_service.generate_coherent_response(
                user_input, memory_context, rag_context
            )
        else:
            response_text = self.llm_service.generate(user_input)
        
        end_time = datetime.now()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Métadonnées de la réponse
        response_metadata = {
            "timestamp": end_time.isoformat(),
            "processing_time_ms": processing_time_ms,
            "include_memory": include_memory,
            "include_rag": include_rag
        }
        
        # Ajout à la mémoire
        conversation_id = self.memory.add_conversation(
            user_input, response_text,
            user_metadata, response_metadata
        )
        
        # Enregistrement dans l'historique
        interaction_data = {
            "id": conversation_id,
            "user_input": user_input,
            "response": response_text,
            "response_time_ms": processing_time_ms,
            "memory_used": include_memory,
            "rag_used": include_rag,
            "memory_context_size": len(memory_context) if memory_context else 0,
            "rag_context_size": len(rag_context) if rag_context else 0
        }
        self.conversation_history.append(interaction_data)
        
        return interaction_data
    
    def run_rag_memory_test(self) -> Dict[str, Any]:
        """
        Exécute un test complet sur l'utilisation conjointe du RAG et de la mémoire.
        
        Returns:
            Résultats du test
        """
        logger.info("=== Début du test RAG + Mémoire ===")
        
        # Liste des questions pour le test
        rag_memory_questions = [
            # Questions sur le RAG
            "Qu'est-ce qu'un système RAG et comment fonctionne-t-il?",
            "Quels sont les composants principaux d'un système RAG?",
            "Quels sont les avantages d'utiliser un système RAG par rapport à un LLM standard?",
            "Quelles bases de données vectorielles sont couramment utilisées dans les systèmes RAG?",
            "Comment les embeddings sont-ils utilisés dans un système RAG?",
            "Quelles sont les limitations des systèmes RAG?",
            "Comment peut-on évaluer l'efficacité d'un système RAG?",
            "Quelle est la différence entre RAG et fine-tuning d'un modèle?",
            "Comment les documents sont-ils indexés dans un système RAG?",
            "Quelle est l'importance du re-ranking dans un système RAG?",
            
            # Questions sur la mémoire hiérarchique
            "Comment fonctionne le système de mémoire hiérarchique de Renée?",
            "Quels sont les 4 niveaux de la mémoire hiérarchique?",
            "Comment les informations sont-elles condensées d'un niveau à l'autre?",
            "Quelle est la différence entre la mémoire hiérarchique et le système RAG?",
            "Comment la mémoire à court terme stocke-t-elle les conversations?",
            "Qu'est-ce qu'un méta-concept dans le contexte de la mémoire hiérarchique?",
            "Comment les connaissances sont-elles consolidées au niveau 4?",
            "Comment le système détermine-t-il quelles informations sont pertinentes pour une requête?",
            "Comment les embeddings sont-ils utilisés dans le système de mémoire?",
            "Comment l'architecture de Renée combine-t-elle mémoire hiérarchique et RAG?"
        ]
        
        # Exécution du test avec les questions
        results = []
        for i, question in enumerate(rag_memory_questions):
            logger.info(f"Question {i+1}/{len(rag_memory_questions)}: {question}")
            
            # Alternance entre différentes configurations pour tester diverses capacités
            use_memory = True
            use_rag = True
            
            # Pour certaines questions, tester sans mémoire ou sans RAG
            if i % 5 == 3:  # Toutes les 5 questions, tester sans mémoire
                use_memory = False
                logger.info("Test sans mémoire (RAG uniquement)")
            elif i % 5 == 4:  # Toutes les 5 questions, tester sans RAG
                use_rag = False
                logger.info("Test sans RAG (mémoire uniquement)")
            
            # Interaction avec le système
            result = self.interact(question, include_memory=use_memory, include_rag=use_rag)
            results.append(result)
            
            logger.info(f"Temps de réponse: {result['response_time_ms']:.2f} ms")
            logger.info(f"Taille du contexte mémoire: {result['memory_context_size']} caractères")
            logger.info(f"Taille du contexte RAG: {result['rag_context_size']} caractères")
            logger.info(f"Réponse: {result['response']}")
            logger.info("---")
            
            # Petite pause entre les questions
            time.sleep(0.5)
        
        # Sauvegarde des résultats
        results_file = self.save_results()
        
        logger.info("=== Test RAG + Mémoire terminé ===")
        
        return {
            "questions_count": len(rag_memory_questions),
            "results": results,
            "results_file": results_file
        }
    
    def save_results(self, filename="rag_memory_test_results.json") -> str:
        """
        Sauvegarde les résultats du test dans un fichier JSON.
        
        Args:
            filename: Nom du fichier pour sauvegarder les résultats
            
        Returns:
            Chemin vers le fichier de résultats
        """
        result_data = {
            "timestamp": datetime.now().isoformat(),
            "interactions": self.conversation_history,
            "memory_stats": {
                "short_term_count": len(self.memory.short_term_memory.conversations),
                "level2_summaries_count": len(self.memory.level2_condenser.summaries),
                "level3_concepts_count": len(self.memory.level3_concepts.meta_concepts),
                "level4_knowledge_count": len(self.memory.level4_knowledge.knowledge_items)
            },
            "rag_stats": {
                "knowledge_base_entries": len(self.rag_knowledge_base)
            }
        }
        
        output_path = os.path.join(self.data_dir, filename)
        
        with open(output_path, "w") as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"Résultats sauvegardés dans: {output_path}")
        
        return output_path

def main():
    """Fonction principale d'exécution du test."""
    # Création du répertoire de test
    test_dir = "/tmp/renee_rag_memory_test"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    # Initialisation du testeur
    tester = ReneeRAGMemoryTester(data_dir=test_dir)
    
    # Exécution du test
    results = tester.run_rag_memory_test()
    
    # Affichage du résumé final
    print("\n=== RÉSUMÉ DU TEST RAG + MÉMOIRE ===")
    print(f"Nombre de questions testées: {results['questions_count']}")
    print(f"Résultats complets sauvegardés dans: {results['results_file']}")
    print("================================\n")
    
    return tester

if __name__ == "__main__":
    main()
