#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de test pour mesurer la capacité de rétention de mémoire de Renée
et analyser les temps de réponse avec le système de mémoire hiérarchique.
"""

import os
import time
import shutil
import random
import statistics
from datetime import datetime, timedelta
import logging
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

# Import des modules nécessaires
from memory_fix import (
    FixedHierarchicalMemoryOrchestrator,
    UserPrompt, SystemResponse, Conversation, Summary, MetaConcept, KnowledgeItem
)
from deepseek_service import DeepSeekService

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("memory_retention_test")

class ReneeMemoryTester:
    """Classe pour tester la capacité de rétention de mémoire de Renée."""
    
    def __init__(self, data_dir: str):
        """
        Initialise le testeur de mémoire Renée.
        
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
        
        # Historique des conversations et performances
        self.conversation_history = []
        self.performance_metrics = {
            "response_times": [],
            "context_sizes": [],
            "timestamps": []
        }
        
        logger.info(f"Testeur de mémoire Renée initialisé dans: {data_dir}")
    
    def interact(self, user_input: str, include_memory: bool = True) -> Dict[str, Any]:
        """
        Interagit avec Renée et enregistre les métriques de performance.
        
        Args:
            user_input: Message de l'utilisateur
            include_memory: Si True, utilise la mémoire pour enrichir la réponse
            
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
        
        # Génération de la réponse
        if include_memory and memory_context:
            response_text = self.llm_service.generate_coherent_response(user_input, memory_context)
        else:
            response_text = self.llm_service.generate(user_input)
        
        end_time = datetime.now()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Métadonnées de la réponse
        response_metadata = {
            "timestamp": end_time.isoformat(),
            "processing_time_ms": processing_time_ms,
            "include_memory": include_memory
        }
        
        # Ajout à la mémoire
        conversation_id = self.memory.add_conversation(
            user_input, response_text,
            user_metadata, response_metadata
        )
        
        # Enregistrement des métriques
        self.performance_metrics["response_times"].append(processing_time_ms)
        self.performance_metrics["context_sizes"].append(len(memory_context) if memory_context else 0)
        self.performance_metrics["timestamps"].append(end_time)
        
        # Enregistrement dans l'historique
        self.conversation_history.append({
            "id": conversation_id,
            "user_input": user_input,
            "response": response_text,
            "response_time_ms": processing_time_ms,
            "context_size": len(memory_context) if memory_context else 0,
            "timestamp": end_time.isoformat()
        })
        
        return {
            "response": response_text,
            "conversation_id": conversation_id,
            "response_time_ms": processing_time_ms,
            "context_size": len(memory_context) if memory_context else 0
        }
    
    def trigger_condensation(self):
        """Force la condensation de la mémoire à tous les niveaux."""
        condensation_result = self.memory.trigger_manual_condensation()
        return condensation_result
    
    def run_memory_test(self, rounds: int = 10, test_interval: int = 3):
        """
        Exécute un test de mémoire en posant des questions au modèle,
        puis en testant sa capacité à se souvenir des informations.
        
        Args:
            rounds: Nombre de rounds d'interactions
            test_interval: Intervalle entre les tests de mémoire
            
        Returns:
            Résultats du test de mémoire
        """
        # Thème du test : voyages et destinations
        topic = "Voyages et destinations"
        unique_fact = f"La montagne Kilimandjaro_ID{random.randint(1000, 9999)} en Tanzanie est le plus haut sommet d'Afrique avec 5895 mètres d'altitude."
        
        logger.info(f"=== Début du test de mémoire sur le thème: {topic} ===")
        logger.info(f"Fait unique à retenir: {unique_fact}")
        
        # Phase 1: Introduction du fait unique
        logger.info(f"Phase 1: Introduction du fait unique...")
        intro_response = self.interact(
            f"Je voudrais te partager cette information: {unique_fact} Peux-tu t'en souvenir pour plus tard?"
        )
        logger.info(f"Temps de réponse: {intro_response['response_time_ms']:.2f} ms")
        logger.info(f"Réponse: {intro_response['response']}")
        
        # Petite pause
        time.sleep(1)
        
        # Phase 2: Conversations diverses (distractions)
        logger.info(f"Phase 2: Conversations diverses...")
        
        # Liste de questions de distraction
        distraction_questions = [
            "Quelle est la capitale de la France?",
            "Quels sont les meilleurs langages de programmation en 2025?",
            "Peux-tu me parler de l'intelligence artificielle?",
            "Quelle est la différence entre le machine learning et le deep learning?",
            "Comment fonctionne un réseau de neurones?",
            "Quels sont les meilleurs films de science-fiction?",
            "Raconte-moi l'histoire d'Internet.",
            "Quels pays ont les cuisines les plus réputées?",
            "Quelles sont les meilleures universités du monde?",
            "Comment fonctionne la blockchain?"
        ]
        
        # Poser des questions de distraction
        for i in range(rounds):
            question = distraction_questions[i % len(distraction_questions)]
            logger.info(f"Distraction {i+1}: {question}")
            
            result = self.interact(question)
            logger.info(f"Temps de réponse: {result['response_time_ms']:.2f} ms")
            
            # Test de mémoire à intervalles réguliers
            if (i + 1) % test_interval == 0:
                logger.info(f"Test de mémoire après {i+1} distractions...")
                
                memory_test = self.interact(
                    "Te souviens-tu du fait que je t'ai partagé sur une montagne au début de notre conversation?"
                )
                
                logger.info(f"Temps de réponse: {memory_test['response_time_ms']:.2f} ms")
                logger.info(f"Taille du contexte: {memory_test['context_size']} caractères")
                logger.info(f"Réponse: {memory_test['response']}")
                
                # Vérifier si le modèle a conservé le fait unique
                if "Kilimandjaro" in memory_test['response'] and "Tanzanie" in memory_test['response']:
                    logger.info("✅ Le modèle a correctement retenu le fait unique!")
                else:
                    logger.info("❌ Le modèle n'a pas retenu le fait unique correctement.")
            
            # Petite pause entre les questions
            time.sleep(0.5)
        
        # Phase 3: Test final de mémoire
        logger.info(f"Phase 3: Test final de mémoire après toutes les distractions...")
        
        # Déclencher une condensation de mémoire
        logger.info("Déclenchement de la condensation de mémoire...")
        condensation_result = self.trigger_condensation()
        
        # Test final
        final_test = self.interact(
            "Quelle est la montagne dont je t'ai parlé au tout début de notre conversation?"
        )
        
        logger.info(f"Temps de réponse final: {final_test['response_time_ms']:.2f} ms")
        logger.info(f"Taille du contexte final: {final_test['context_size']} caractères")
        logger.info(f"Réponse finale: {final_test['response']}")
        
        # Vérifier si le modèle a conservé le fait unique
        if "Kilimandjaro" in final_test['response'] and "Tanzanie" in final_test['response']:
            logger.info("✅ Test final réussi! Le modèle a correctement retenu le fait unique!")
            success = True
        else:
            logger.info("❌ Test final échoué. Le modèle n'a pas retenu le fait unique correctement.")
            success = False
        
        # Affichage des statistiques de mémoire
        self.print_memory_statistics()
        
        # Analyse des temps de réponse
        self.analyze_response_times()
        
        logger.info("=== Test de mémoire terminé ===")
        
        return {
            "topic": topic,
            "unique_fact": unique_fact,
            "rounds": rounds,
            "final_success": success,
            "avg_response_time": statistics.mean(self.performance_metrics["response_times"]) if self.performance_metrics["response_times"] else 0,
            "max_context_size": max(self.performance_metrics["context_sizes"]) if self.performance_metrics["context_sizes"] else 0
        }
    
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
    
    def analyze_response_times(self):
        """Analyse les temps de réponse et génère des statistiques."""
        if not self.performance_metrics["response_times"]:
            logger.info("Aucune donnée de temps de réponse disponible pour l'analyse.")
            return
        
        response_times = self.performance_metrics["response_times"]
        context_sizes = self.performance_metrics["context_sizes"]
        
        # Statistiques de base
        avg_time = statistics.mean(response_times)
        median_time = statistics.median(response_times)
        std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0
        min_time = min(response_times)
        max_time = max(response_times)
        
        logger.info("=== Analyse des temps de réponse ===")
        logger.info(f"Temps moyen: {avg_time:.2f} ms")
        logger.info(f"Temps médian: {median_time:.2f} ms")
        logger.info(f"Écart-type: {std_dev:.2f} ms")
        logger.info(f"Temps minimum: {min_time:.2f} ms")
        logger.info(f"Temps maximum: {max_time:.2f} ms")
        
        # Relation avec la taille du contexte
        if context_sizes and all(s > 0 for s in context_sizes):
            avg_context = statistics.mean(context_sizes)
            logger.info(f"Taille moyenne du contexte: {avg_context:.2f} caractères")
            
            # Calcul d'une corrélation simplifiée
            context_with_time = [(c, t) for c, t in zip(context_sizes, response_times) if c > 0]
            if context_with_time:
                contexts, times = zip(*context_with_time)
                
                # Calcul d'une tendance simple
                if len(context_with_time) > 1:
                    correlation = sum((c - statistics.mean(contexts)) * (t - statistics.mean(times)) 
                                    for c, t in context_with_time) / len(context_with_time)
                    normalized = correlation / (statistics.stdev(contexts) * statistics.stdev(times))
                    
                    logger.info(f"Corrélation contexte-temps: {normalized:.2f}")
                    
                    if normalized > 0.5:
                        logger.info("Forte corrélation positive: le temps de réponse augmente avec la taille du contexte")
                    elif normalized < -0.5:
                        logger.info("Forte corrélation négative: le temps de réponse diminue avec la taille du contexte")
                    else:
                        logger.info("Pas de corrélation forte entre la taille du contexte et le temps de réponse")
        
        # Tendance au fil du temps
        if len(response_times) > 1:
            first_half = response_times[:len(response_times)//2]
            second_half = response_times[len(response_times)//2:]
            
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)
            
            logger.info(f"Évolution du temps de réponse: Première moitié: {first_avg:.2f} ms, Seconde moitié: {second_avg:.2f} ms")
            
            if second_avg > first_avg * 1.2:
                logger.info("⚠️ Le temps de réponse a significativement augmenté au fil du temps")
            elif first_avg > second_avg * 1.2:
                logger.info("✅ Le temps de réponse a significativement diminué au fil du temps")
            else:
                logger.info("Le temps de réponse est resté relativement stable au fil du temps")
        
        return {
            "avg_time": avg_time,
            "median_time": median_time,
            "std_dev": std_dev,
            "min_time": min_time,
            "max_time": max_time
        }
    
    def save_results(self, filename="memory_test_results.json"):
        """
        Sauvegarde les résultats du test dans un fichier JSON.
        
        Args:
            filename: Nom du fichier pour sauvegarder les résultats
        """
        result_data = {
            "conversation_history": self.conversation_history,
            "performance_metrics": {
                "response_times": [float(t) for t in self.performance_metrics["response_times"]],
                "context_sizes": self.performance_metrics["context_sizes"],
                "timestamps": [ts.isoformat() for ts in self.performance_metrics["timestamps"]]
            },
            "memory_stats": {
                "short_term_count": len(self.memory.short_term_memory.conversations),
                "level2_summaries_count": len(self.memory.level2_condenser.summaries),
                "level3_concepts_count": len(self.memory.level3_concepts.meta_concepts),
                "level4_knowledge_count": len(self.memory.level4_knowledge.knowledge_items)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        output_path = os.path.join(self.data_dir, filename)
        
        with open(output_path, "w") as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"Résultats sauvegardés dans: {output_path}")
        
        return output_path

def main():
    """Fonction principale d'exécution du test."""
    # Création du répertoire de test
    test_dir = "/tmp/renee_memory_retention_test"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    # Initialisation du testeur
    tester = ReneeMemoryTester(data_dir=test_dir)
    
    # Exécution du test de mémoire avec 10 rounds et test tous les 3 rounds
    results = tester.run_memory_test(rounds=10, test_interval=3)
    
    # Sauvegarde des résultats
    results_file = tester.save_results()
    
    # Affichage du résumé final
    print("\n=== RÉSUMÉ DU TEST DE MÉMOIRE ===")
    print(f"Thème testé: {results['topic']}")
    print(f"Fait unique: {results['unique_fact']}")
    print(f"Nombre de distractions: {results['rounds']}")
    print(f"Test final réussi: {'OUI' if results['final_success'] else 'NON'}")
    print(f"Temps de réponse moyen: {results['avg_response_time']:.2f} ms")
    print(f"Taille maximale du contexte: {results['max_context_size']} caractères")
    print(f"Résultats complets sauvegardés dans: {results_file}")
    print("================================\n")
    
    return tester

if __name__ == "__main__":
    main()
