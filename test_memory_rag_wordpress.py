#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de test pour évaluer l'accès de Renée aux informations sur WordPress et Elementor
en alternant entre la mémoire et le RAG.
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
logger = logging.getLogger("wordpress_elementor_test")

class ReneeWordPressElementorTester:
    """Classe pour tester l'accès de Renée aux informations sur WordPress et Elementor."""
    
    def __init__(self, data_dir: str):
        """
        Initialise le testeur WordPress+Elementor de Renée.
        
        Args:
            data_dir: Répertoire pour les données persistantes
        """
        self.data_dir = data_dir
        
        # Création du répertoire de données s'il n'existe pas
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialisation du service LLM
        self.llm_service = DeepSeekService()
        
        # Initialisation du système de mémoire hiérarchique
        self.memory = FixedHierarchicalMemoryOrchestrator(
            data_dir=os.path.join(data_dir, "memory"),
            llm_service=self.llm_service
        )
        
        # Historique des conversations
        self.conversation_history = []
        
        logger.info(f"Testeur WordPress+Elementor de Renée initialisé dans: {data_dir}")
    
    def _retrieve_rag_context(self, query: str, top_k: int = 2) -> str:
        """
        Simule la récupération d'information du système RAG.
        
        Args:
            query: Requête utilisateur
            top_k: Nombre de résultats à retourner
            
        Returns:
            Contexte RAG formaté
        """
        # Pour ce test, on simule un contexte RAG spécifique pour WordPress et Elementor
        query_lower = query.lower()
        
        if "wordpress" in query_lower or "wp" in query_lower or "api" in query_lower:
            return """
            [Document: Guide complet WordPress API]
            WordPress fournit une API REST complète qui permet d'interagir avec presque tous les aspects d'un site WordPress.
            Les principaux endpoints sont: /wp/v2/posts, /wp/v2/pages, /wp/v2/media, /wp/v2/users, et /wp/v2/settings.
            L'authentification peut se faire via JWT, OAuth ou Application Passwords.
            
            [Document: Guide API WordPress détaillé]
            Pour utiliser l'API WordPress, vous devez d'abord obtenir un jeton d'authentification. 
            Avec les mots de passe d'application (disponibles depuis WordPress 5.6), allez dans votre profil utilisateur 
            et générez un nouveau mot de passe d'application. Ce mot de passe peut ensuite être utilisé avec 
            l'authentification Basic (username:app_password) encodée en Base64.
            """
        
        elif "elementor" in query_lower or "élément" in query_lower or "widget" in query_lower:
            return """
            [Document: Guide Elementor Modifications API]
            Elementor fournit plusieurs hooks et filtres qui permettent d'étendre ses fonctionnalités:
            - elementor/widgets/widgets_registered: pour enregistrer des widgets personnalisés
            - elementor/controls/controls_registered: pour ajouter des contrôles personnalisés
            - elementor/elements/categories_registered: pour ajouter des catégories personnalisées
            
            Pour créer un widget Elementor personnalisé, créez une classe qui étend \Widget_Base et implémentez
            les méthodes get_name(), get_title(), get_icon(), get_categories(), _register_controls() et render().
            """
        
        return ""
    
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
    
    def run_wordpress_elementor_test(self) -> Dict[str, Any]:
        """
        Exécute un test complet sur l'accès aux informations WordPress et Elementor.
        
        Returns:
            Résultats du test
        """
        logger.info("=== Début du test WordPress + Elementor ===")
        
        # Liste des questions pour le test, alternant entre mémoire et RAG
        questions = [
            # Q1: Mémoire - Introduction
            "Bonjour Renée, peux-tu te présenter et me parler de tes capacités?",
            
            # Q2: RAG - WordPress API
            "Comment authentifier une requête à l'API WordPress?",
            
            # Q3: Mémoire - Rappel de capacités
            "Quel est ton système de mémoire principal?",
            
            # Q4: RAG - Elementor
            "Comment créer un widget personnalisé dans Elementor?",
            
            # Q5: Mémoire - Test de mémoire
            "Te souviens-tu de ce que je t'ai demandé sur WordPress au début de notre conversation?",
            
            # Q6: RAG - WordPress fonctionnalités
            "Quels sont les principaux endpoints de l'API WordPress?",
            
            # Q7: Mémoire - Mémoire hiérarchique
            "Peux-tu m'expliquer comment fonctionne ta mémoire hiérarchique?",
            
            # Q8: RAG - Elementor hooks
            "Quels sont les principaux hooks disponibles dans Elementor?",
            
            # Q9: Mémoire - Test de rétention
            "Quelles informations as-tu retenues sur Elementor jusqu'à présent?",
            
            # Q10: RAG - WordPress sécurité
            "Quelles sont les meilleures pratiques de sécurité pour l'API WordPress?",
            
            # Q11: Mémoire - Système RAG
            "Explique-moi comment ton système RAG complète ta mémoire",
            
            # Q12: RAG - Elementor styles
            "Comment personnaliser les styles des widgets Elementor?",
            
            # Q13: Mémoire - Test d'intégration
            "Quels avantages y a-t-il à combiner WordPress et Elementor?",
            
            # Q14: RAG - WordPress plugins
            "Comment développer un plugin WordPress qui s'intègre avec l'API REST?",
            
            # Q15: Mémoire - Résumé de conversation
            "Peux-tu me résumer notre conversation jusqu'à présent?",
            
            # Q16: RAG - Elementor formulaires
            "Comment créer un formulaire personnalisé avec Elementor?",
            
            # Q17: Mémoire - Kilimanjaro (test de rétention du fait unique)
            "Te souviens-tu d'informations sur une montagne africaine dont nous avons parlé?",
            
            # Q18: RAG - WordPress multisite
            "Comment fonctionne l'API REST de WordPress dans un environnement multisite?",
            
            # Q19: Mémoire - Test de transition
            "Quelles questions t'ai-je posées sur les formulaires?",
            
            # Q20: RAG - Elementor Pro vs Free
            "Quelles sont les différences entre Elementor Free et Elementor Pro?"
        ]
        
        # Exécution du test avec les questions
        results = []
        for i, question in enumerate(questions):
            logger.info(f"Question {i+1}/{len(questions)}: {question}")
            
            # Déterminer si on utilise la mémoire, le RAG ou les deux
            use_memory = True
            use_rag = True
            
            # Pour les questions impaires (1, 3, 5...), on met l'accent sur la mémoire
            if (i + 1) % 2 == 1:
                logger.info("Test avec accent sur la mémoire")
                use_rag = False  # Désactiver le RAG pour ces questions
            
            # Pour les questions paires (2, 4, 6...), on met l'accent sur le RAG
            else:
                logger.info("Test avec accent sur le RAG")
                use_memory = False  # Désactiver la mémoire pour ces questions
            
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
        
        logger.info("=== Test WordPress + Elementor terminé ===")
        
        return {
            "questions_count": len(questions),
            "results": results,
            "results_file": results_file
        }
    
    def save_results(self, filename="wordpress_elementor_test_results.json") -> str:
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
    test_dir = "/tmp/renee_wordpress_elementor_test"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    # Initialisation du testeur
    tester = ReneeWordPressElementorTester(data_dir=test_dir)
    
    # Exécution du test
    results = tester.run_wordpress_elementor_test()
    
    # Affichage du résumé final
    print("\n=== RÉSUMÉ DU TEST WORDPRESS + ELEMENTOR ===")
    print(f"Nombre de questions testées: {results['questions_count']}")
    print(f"Résultats complets sauvegardés dans: {results['results_file']}")
    print("================================\n")
    
    return tester

if __name__ == "__main__":
    main()
