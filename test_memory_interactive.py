import os
import sys
import json
import time
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Ajouter le chemin racine au PATH pour l'importation des modules
root_path = Path(__file__).parent.absolute()
sys.path.append(str(root_path))

# Importer les modules nécessaires
from memory_fix import FixedHierarchicalMemoryOrchestrator
from deepseek_service import DeepSeekService
from fast_model_service_fixed import FastModelService

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("memory_test")

class Timer:
    """Utilitaire simple pour mesurer le temps d'exécution"""
    
    def __init__(self, name):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        logger.info(f"⏱️ TEMPS - {self.name}: {elapsed:.3f} secondes")

class MemoryTester:
    """
    Testeur interactif de la mémoire de Renée avec DeepSeek via Ollama
    """
    
    def __init__(self):
        """Initialise le testeur interactif"""
        # Créer un répertoire temporaire pour les données de mémoire
        with Timer("Initialisation du répertoire de données"):
            self.data_dir = tempfile.mkdtemp(prefix="renee_memory_test_")
            logger.info(f"Répertoire de données de mémoire: {self.data_dir}")
        
        # Définir les variables d'environnement nécessaires
        os.environ["OLLAMA_ENDPOINT"] = "http://localhost:11434/api/chat"
        
        # Initialiser les composants de Renée
        with Timer("Initialisation de la mémoire hiérarchique"):
            self.memory = FixedHierarchicalMemoryOrchestrator(data_dir=self.data_dir)
        
        with Timer("Initialisation du service FastModel"):
            self.model_service = FastModelService(
                base_model_name="deepseek-ai/deepseek-r1-llm-7b",
                adapter_path="/Users/pascalseppey/Downloads/final-deep",
                memory_orchestrator=self.memory
            )
        
        # Vérifier si le modèle est disponible
        if self.model_service.model is None:
            logger.warning("Le modèle FastModel n'a pas pu être chargé, utilisant DeepSeek comme fallback")
            with Timer("Initialisation du service DeepSeek (fallback)"):
                self.deepseek = DeepSeekService(model_name="deepseek-coder:1.3b")
                self.use_fast_model = False
        else:
            logger.info("Utilisation du modèle DeepSeek fine-tuné pour des réponses rapides")
            self.deepseek = None  # Pas besoin du service DeepSeek si on utilise le FastModel
            self.use_fast_model = True
        
        # Historique des conversations pour le test
        self.conversation_history = []
        self.affirmations = []
        
        # Résumé des temps d'exécution
        self.time_stats = {
            "get_context": [],
            "generate_response": [],
            "add_to_memory": [],
            "total_per_interaction": []
        }
        
    def run_memory_test(self):
        """Exécute le test interactif de mémoire"""
        with Timer("Test complet de mémoire"):
            logger.info("=== Début du test interactif de mémoire ===")
            
            # Phase 1: Faire 3 affirmations à Renée
            self._make_affirmations()
            
            # Phase 2: Poser des questions sur les affirmations pour tester la mémoire
            self._test_memory()
            
            # Afficher les statistiques de temps
            self._print_time_stats()
            
            logger.info("=== Test interactif de mémoire terminé ===")
    
    def _make_affirmations(self):
        """Phase 1: Faire 3 affirmations à Renée sur WordPress et Elementor"""
        logger.info("--- Phase 1: Affirmations ---")
        
        # Liste réduite à 3 affirmations au lieu de 20
        affirmations = [
            "WordPress a été créé en 2003 par Matt Mullenweg et Mike Little.",
            "Elementor a été lancé en 2016 et est devenu l'un des page builders les plus populaires pour WordPress.",
            "WordPress utilise PHP comme langage de programmation principal."
        ]
        
        # Faire les affirmations et enregistrer les réponses
        for i, affirmation in enumerate(affirmations):
            logger.info(f"Affirmation {i+1}/{len(affirmations)}: {affirmation}")
            self.affirmations.append(affirmation)
            
            interaction_start = time.time()
            
            # Récupérer le contexte mémoire
            with Timer(f"Récupération du contexte mémoire (affirmation {i+1})"):
                memory_context = self.memory.get_context_for_query(affirmation)
                self.time_stats["get_context"].append(time.time() - interaction_start)
            
            # Générer une réponse
            with Timer(f"Génération de réponse (affirmation {i+1})"):
                if self.use_fast_model:
                    response, exec_time = self.model_service.generate_response(
                        user_message=affirmation,
                        memory_context=memory_context
                    )
                else:
                    response, exec_time = self.deepseek.generate_response(
                        user_message=affirmation,
                        conversation_history=self.conversation_history,
                        memory_context=memory_context
                    )
                self.time_stats["generate_response"].append(exec_time)
            
            # Mettre à jour l'historique de conversation
            self.conversation_history.append({"role": "user", "content": affirmation})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Enregistrer dans la mémoire
            with Timer(f"Ajout à la mémoire (affirmation {i+1})"):
                self.memory.add_conversation(affirmation, response)
                self.time_stats["add_to_memory"].append(time.time() - (interaction_start + self.time_stats["get_context"][-1] + self.time_stats["generate_response"][-1]))
            
            # Calculer le temps total pour cette interaction
            total_time = time.time() - interaction_start
            self.time_stats["total_per_interaction"].append(total_time)
            logger.info(f"⏱️ Temps total (affirmation {i+1}): {total_time:.3f} secondes")
            
            logger.info(f"Réponse: {response}")
            logger.info("---")
            
            # Pause de 1 seconde entre les affirmations
            time.sleep(1)
    
    def _test_memory(self):
        """Phase 2: Tester la mémoire en posant 3 questions sur les affirmations précédentes"""
        logger.info("--- Phase 2: Test de mémoire ---")
        
        # Questions basées sur les affirmations précédentes (réduites à 3)
        questions = [
            "Qui a créé WordPress et en quelle année?",
            "En quelle année Elementor a-t-il été lancé?",
            "Quel langage de programmation utilise WordPress?"
        ]
        
        # Poser les questions et évaluer les réponses
        for i, question in enumerate(questions):
            logger.info(f"Question {i+1}/{len(questions)}: {question}")
            
            interaction_start = time.time()
            
            # Récupérer le contexte mémoire
            with Timer(f"Récupération du contexte mémoire (question {i+1})"):
                memory_context = self.memory.get_context_for_query(question)
                self.time_stats["get_context"].append(time.time() - interaction_start)
            
            # Générer une réponse
            with Timer(f"Génération de réponse (question {i+1})"):
                if self.use_fast_model:
                    response, exec_time = self.model_service.generate_response(
                        user_message=question,
                        memory_context=memory_context
                    )
                else:
                    response, exec_time = self.deepseek.generate_response(
                        user_message=question,
                        conversation_history=self.conversation_history,
                        memory_context=memory_context
                    )
                self.time_stats["generate_response"].append(exec_time)
            
            # Mettre à jour l'historique de conversation
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Enregistrer dans la mémoire
            with Timer(f"Ajout à la mémoire (question {i+1})"):
                self.memory.add_conversation(question, response)
                self.time_stats["add_to_memory"].append(time.time() - (interaction_start + self.time_stats["get_context"][-1] + self.time_stats["generate_response"][-1]))
            
            # Calculer le temps total pour cette interaction
            total_time = time.time() - interaction_start
            self.time_stats["total_per_interaction"].append(total_time)
            logger.info(f"⏱️ Temps total (question {i+1}): {total_time:.3f} secondes")
            
            logger.info(f"Contexte mémoire: {len(memory_context)} caractères")
            logger.info(f"Réponse: {response}")
            logger.info("---")
            
            # Pause de 1 seconde entre les questions
            time.sleep(1)
            
        # Une seule question globale au lieu de 5
        meta_questions = [
            "Résume ce que nous avons discuté jusqu'à présent."
        ]
        
        logger.info("--- Phase 3: Question sur la conversation globale ---")
        
        for i, question in enumerate(meta_questions):
            logger.info(f"Question globale {i+1}/{len(meta_questions)}: {question}")
            
            interaction_start = time.time()
            
            # Récupérer le contexte mémoire
            with Timer(f"Récupération du contexte mémoire (question globale {i+1})"):
                memory_context = self.memory.get_context_for_query(question)
                self.time_stats["get_context"].append(time.time() - interaction_start)
            
            # Générer une réponse
            with Timer(f"Génération de réponse (question globale {i+1})"):
                if self.use_fast_model:
                    response, exec_time = self.model_service.generate_response(
                        user_message=question,
                        memory_context=memory_context
                    )
                else:
                    response, exec_time = self.deepseek.generate_response(
                        user_message=question,
                        conversation_history=self.conversation_history,
                        memory_context=memory_context
                    )
                self.time_stats["generate_response"].append(exec_time)
            
            # Mettre à jour l'historique de conversation
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Enregistrer dans la mémoire
            with Timer(f"Ajout à la mémoire (question globale {i+1})"):
                self.memory.add_conversation(question, response)
                self.time_stats["add_to_memory"].append(time.time() - (interaction_start + self.time_stats["get_context"][-1] + self.time_stats["generate_response"][-1]))
            
            # Calculer le temps total pour cette interaction
            total_time = time.time() - interaction_start
            self.time_stats["total_per_interaction"].append(total_time)
            logger.info(f"⏱️ Temps total (question globale {i+1}): {total_time:.3f} secondes")
            
            logger.info(f"Contexte mémoire: {len(memory_context)} caractères")
            logger.info(f"Réponse: {response}")
            logger.info("---")
            
            # Pause de 1 seconde entre les questions
            time.sleep(1)
    
    def _print_time_stats(self):
        """Affiche les statistiques de temps d'exécution"""
        logger.info("=== STATISTIQUES DE TEMPS D'EXÉCUTION ===")
        
        # Fonction pour calculer des statistiques de base
        def calc_stats(data):
            if not data:
                return {"min": 0, "max": 0, "avg": 0, "total": 0}
            return {
                "min": min(data),
                "max": max(data),
                "avg": sum(data) / len(data),
                "total": sum(data)
            }
        
        # Calculer les statistiques pour chaque étape
        stats = {key: calc_stats(values) for key, values in self.time_stats.items()}
        
        # Afficher les statistiques
        logger.info(f"Temps total du test: {stats['total_per_interaction']['total']:.3f} secondes")
        logger.info(f"Nombre d'interactions: {len(self.time_stats['total_per_interaction'])}")
        logger.info("")
        
        logger.info("Récupération du contexte mémoire:")
        logger.info(f"  - Temps min: {stats['get_context']['min']:.3f} secondes")
        logger.info(f"  - Temps max: {stats['get_context']['max']:.3f} secondes")
        logger.info(f"  - Temps moyen: {stats['get_context']['avg']:.3f} secondes")
        logger.info(f"  - Temps total: {stats['get_context']['total']:.3f} secondes ({stats['get_context']['total']/stats['total_per_interaction']['total']*100:.1f}%)")
        
        logger.info("Génération de réponse:")
        logger.info(f"  - Temps min: {stats['generate_response']['min']:.3f} secondes")
        logger.info(f"  - Temps max: {stats['generate_response']['max']:.3f} secondes")
        logger.info(f"  - Temps moyen: {stats['generate_response']['avg']:.3f} secondes")
        logger.info(f"  - Temps total: {stats['generate_response']['total']:.3f} secondes ({stats['generate_response']['total']/stats['total_per_interaction']['total']*100:.1f}%)")
        
        logger.info("Ajout à la mémoire:")
        logger.info(f"  - Temps min: {stats['add_to_memory']['min']:.3f} secondes")
        logger.info(f"  - Temps max: {stats['add_to_memory']['max']:.3f} secondes")
        logger.info(f"  - Temps moyen: {stats['add_to_memory']['avg']:.3f} secondes")
        logger.info(f"  - Temps total: {stats['add_to_memory']['total']:.3f} secondes ({stats['add_to_memory']['total']/stats['total_per_interaction']['total']*100:.1f}%)")

def main():
    """Fonction principale pour exécuter le test"""
    tester = MemoryTester()
    tester.run_memory_test()

if __name__ == "__main__":
    main()
