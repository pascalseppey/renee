#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour implémenter un système RLHF pour DeepSeek 7B
utilisant OpenAI (GPT-4) comme modèle critique.

Ce script permet :
1. De générer des réponses avec DeepSeek via Ollama
2. D'évaluer ces réponses avec OpenAI
3. De fournir des instructions d'amélioration
4. De collecter ces données pour le fine-tuning
"""

import os
import json
import time
import random
import requests
import argparse
from openai import OpenAI
from typing import Dict, List, Any, Tuple
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration par défaut
DEFAULT_OPENAI_KEY = "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"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_DEEPSEEK_MODEL = "deepseek-r1:7b"
DEFAULT_CRITIC_MODEL = "gpt-4o"
DEFAULT_OUTPUT_FILE = "./data/rlhf_feedback.jsonl"

class RLHFTrainer:
    """
    Classe pour entraîner DeepSeek avec RLHF en utilisant OpenAI comme critique
    """
    
    def __init__(self, 
                 openai_key: str, 
                 ollama_url: str, 
                 deepseek_model: str, 
                 critic_model: str,
                 output_file: str):
        """
        Initialise le trainer RLHF
        
        Args:
            openai_key: Clé API OpenAI
            ollama_url: URL du serveur Ollama
            deepseek_model: Nom du modèle DeepSeek dans Ollama
            critic_model: Nom du modèle critique OpenAI
            output_file: Fichier de sortie pour les données RLHF
        """
        self.openai_client = OpenAI(api_key=openai_key)
        self.ollama_url = ollama_url
        self.deepseek_model = deepseek_model
        self.critic_model = critic_model
        self.output_file = output_file
        
        # Vérifier que Ollama est disponible
        self.check_ollama_available()
        
        # Créer le répertoire de sortie si nécessaire
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        logger.info(f"RLHF Trainer initialisé avec DeepSeek ({deepseek_model}) et critique ({critic_model})")
    
    def check_ollama_available(self) -> bool:
        """Vérifie si Ollama est disponible et si le modèle DeepSeek est présent"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                # Vérifier si le modèle DeepSeek est disponible
                for model in models:
                    if self.deepseek_model.lower() in model.get("name", "").lower():
                        logger.info(f"Modèle DeepSeek trouvé dans Ollama: {model.get('name')}")
                        self.deepseek_model = model.get("name")
                        return True
                
                logger.warning(f"Le modèle {self.deepseek_model} n'a pas été trouvé dans Ollama")
                return False
            else:
                logger.error(f"Erreur lors de la requête à Ollama: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Erreur lors de la connexion à Ollama: {e}")
            return False
    
    def generate_deepseek_response(self, 
                                  user_message: str, 
                                  memory_context: str = "", 
                                  rag_context: str = "") -> str:
        """
        Génère une réponse avec DeepSeek via Ollama
        
        Args:
            user_message: Message de l'utilisateur
            memory_context: Contexte de la mémoire hiérarchique
            rag_context: Contexte RAG
            
        Returns:
            Réponse générée par DeepSeek
        """
        # Construire le prompt système
        system_prompt = f"""Tu es Renée, une assistante IA francophone qui suit ces règles strictes:
1. Réponds UNIQUEMENT en utilisant les informations des CONTEXTES ci-dessous
2. Si l'information n'est pas dans les CONTEXTES, réponds simplement "Information non disponible"
3. Sois TRÈS CONCISE (20-30 mots maximum)
4. Ne génère PAS de section <think> - réponds directement
5. Utilise un langage simple et direct

"""
        
        # Ajouter les contextes
        if memory_context:
            system_prompt += f"""
CONTEXTE DE LA MÉMOIRE HIÉRARCHIQUE (UTILISE UNIQUEMENT CES INFORMATIONS):
{memory_context}
"""
        
        if rag_context:
            system_prompt += f"""
CONTEXTE DU SYSTÈME DE RAG (UTILISE UNIQUEMENT CES INFORMATIONS):
{rag_context}
"""
        
        system_prompt += """
RAPPEL FINAL: Réponds UNIQUEMENT avec les informations présentes dans les CONTEXTES ci-dessus.
Sois ultra concise. Si l'information demandée n'est pas disponible, dis simplement "Information non disponible".
"""
        
        # Construire les messages pour l'API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        try:
            # Appel à l'API Ollama
            payload = {
                "model": self.deepseek_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_tokens": 100
                }
            }
            
            response = requests.post(f"{self.ollama_url}/api/chat", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                logger.error(f"Erreur lors de la génération avec Ollama: {response.status_code}")
                return f"Erreur: {response.status_code}"
        
        except Exception as e:
            logger.error(f"Exception lors de la génération avec Ollama: {e}")
            return f"Exception: {str(e)}"
    
    def evaluate_response(self, 
                         question: str, 
                         memory_context: str, 
                         rag_context: str, 
                         response: str) -> Tuple[float, str, str]:
        """
        Évalue la réponse de DeepSeek avec le modèle critique OpenAI
        
        Args:
            question: Question de l'utilisateur
            memory_context: Contexte de la mémoire
            rag_context: Contexte RAG
            response: Réponse générée par DeepSeek
            
        Returns:
            Tuple (score, critique détaillée, instructions d'amélioration)
        """
        try:
            # Construire le prompt d'évaluation
            eval_prompt = f"""
Tu es un évaluateur d'IA expert dont le rôle est d'évaluer la qualité des réponses générées par un LLM nommé DeepSeek.
DeepSeek doit répondre aux questions en utilisant UNIQUEMENT les informations des contextes fournis.

QUESTION DE L'UTILISATEUR:
{question}

CONTEXTE DE MÉMOIRE DISPONIBLE:
{memory_context}

CONTEXTE RAG DISPONIBLE:
{rag_context}

RÉPONSE GÉNÉRÉE PAR DEEPSEEK:
{response}

Ton travail est d'évaluer la réponse selon ces critères:

1. ADHÉRENCE AU CONTEXTE (0-10): La réponse utilise-t-elle UNIQUEMENT les informations des contextes fournis?
2. CONCISION (0-10): La réponse est-elle concise (20-30 mots maximum)?
3. PERTINENCE (0-10): La réponse répond-elle correctement à la question posée?
4. FORMAT (0-10): La réponse évite-t-elle tout contenu superflu (introductions, formules de politesse)?
5. PRÉCISION (0-10): La réponse indique-t-elle "Information non disponible" quand approprié?

Fournis une évaluation structurée avec:
1. Un score numérique pour chaque critère (0-10)
2. Une critique détaillée expliquant les forces et faiblesses
3. Des instructions d'amélioration TRÈS PRÉCISES, expliquant EXACTEMENT comment DeepSeek devrait modifier sa réponse

Format de ta réponse:
```
SCORES:
- Adhérence au contexte: X/10
- Concision: X/10  
- Pertinence: X/10
- Format: X/10
- Précision: X/10
- SCORE GLOBAL: X/10

CRITIQUE DÉTAILLÉE:
[Ta critique détaillée ici]

INSTRUCTIONS D'AMÉLIORATION:
[Tes instructions précises ici]
```
"""
            
            # Appel à l'API OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.critic_model,
                messages=[
                    {"role": "system", "content": "Tu es un évaluateur expert de LLMs qui fournit des critiques précises et instructives."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            eval_text = response.choices[0].message.content
            
            # Extraire les informations structurées de l'évaluation
            score_global = self._extract_global_score(eval_text)
            critique = self._extract_section(eval_text, "CRITIQUE DÉTAILLÉE:")
            instructions = self._extract_section(eval_text, "INSTRUCTIONS D'AMÉLIORATION:")
            
            return score_global, critique, instructions
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation avec OpenAI: {e}")
            return 0.0, f"Erreur: {str(e)}", "Pas d'instructions disponibles"
    
    def _extract_global_score(self, eval_text: str) -> float:
        """Extrait le score global de l'évaluation"""
        try:
            if "SCORE GLOBAL:" in eval_text:
                score_line = [line for line in eval_text.split('\n') if "SCORE GLOBAL:" in line][0]
                score_str = score_line.split("SCORE GLOBAL:")[1].strip().split('/')[0].strip()
                return float(score_str)
            return 5.0  # Valeur par défaut
        except Exception:
            return 5.0  # Valeur par défaut en cas d'erreur
    
    def _extract_section(self, eval_text: str, section_header: str) -> str:
        """Extrait une section spécifique de l'évaluation"""
        try:
            if section_header in eval_text:
                sections = eval_text.split(section_header)
                if len(sections) > 1:
                    section_content = sections[1].strip()
                    # Chercher le début de la section suivante
                    next_section_markers = ["CRITIQUE DÉTAILLÉE:", "INSTRUCTIONS D'AMÉLIORATION:"]
                    for marker in next_section_markers:
                        if marker in section_content and marker != section_header:
                            section_content = section_content.split(marker)[0].strip()
                    return section_content
            return ""
        except Exception:
            return ""
    
    def run_training_example(self, 
                            question: str, 
                            memory_context: str = "", 
                            rag_context: str = "") -> Dict[str, Any]:
        """
        Exécute une itération complète de l'entraînement RLHF
        
        Args:
            question: Question de l'utilisateur
            memory_context: Contexte de mémoire (optionnel)
            rag_context: Contexte RAG (optionnel)
            
        Returns:
            Dictionnaire contenant les données de l'itération
        """
        logger.info(f"Exécution d'un exemple RLHF pour la question: {question}")
        
        # Étape 1: Générer une réponse avec DeepSeek
        start_time = time.time()
        response = self.generate_deepseek_response(question, memory_context, rag_context)
        generation_time = time.time() - start_time
        logger.info(f"Réponse générée en {generation_time:.2f}s: {response}")
        
        # Étape 2: Évaluer la réponse avec le modèle critique
        start_time = time.time()
        score, critique, instructions = self.evaluate_response(question, memory_context, rag_context, response)
        evaluation_time = time.time() - start_time
        logger.info(f"Évaluation en {evaluation_time:.2f}s - Score: {score}/10")
        
        # Étape 3: Générer une réponse améliorée en tenant compte des instructions
        improved_prompt = f"""
Tu es Renée, une assistante IA francophone. Réponds à cette question en suivant STRICTEMENT ces instructions d'amélioration:

QUESTION: {question}

CONTEXTE DE MÉMOIRE:
{memory_context}

CONTEXTE RAG:
{rag_context}

TA RÉPONSE PRÉCÉDENTE:
{response}

INSTRUCTIONS D'AMÉLIORATION À SUIVRE OBLIGATOIREMENT:
{instructions}

Génère maintenant une réponse améliorée en suivant ces instructions.
"""
        
        start_time = time.time()
        improved_response = self.generate_deepseek_response(improved_prompt, memory_context, rag_context)
        improvement_time = time.time() - start_time
        logger.info(f"Réponse améliorée en {improvement_time:.2f}s: {improved_response}")
        
        # Étape 4: Évaluer la réponse améliorée
        start_time = time.time()
        improved_score, improved_critique, _ = self.evaluate_response(question, memory_context, rag_context, improved_response)
        improved_evaluation_time = time.time() - start_time
        logger.info(f"Évaluation améliorée en {improved_evaluation_time:.2f}s - Score: {improved_score}/10")
        
        # Construire l'exemple RLHF
        rlhf_example = {
            "question": question,
            "memory_context": memory_context,
            "rag_context": rag_context,
            "original_response": {
                "text": response,
                "score": score,
                "critique": critique,
                "instructions": instructions,
                "generation_time": generation_time
            },
            "improved_response": {
                "text": improved_response,
                "score": improved_score,
                "critique": improved_critique,
                "generation_time": improvement_time
            },
            "score_difference": improved_score - score,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Sauvegarder l'exemple
        self._save_example(rlhf_example)
        
        return rlhf_example
    
    def _save_example(self, example: Dict[str, Any]):
        """Sauvegarde un exemple dans le fichier de sortie"""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    def run_training_session(self, num_examples: int = 5):
        """
        Exécute une session d'entraînement avec plusieurs exemples
        
        Args:
            num_examples: Nombre d'exemples à générer
        """
        logger.info(f"Début de la session d'entraînement RLHF avec {num_examples} exemples")
        
        # Exemples de questions et contextes
        questions = [
            "Qui a créé WordPress et quand?",
            "Quels sont les principaux avantages d'Elementor?",
            "Comment fonctionne le système de hiérarchie de mémoire?",
            "Quelle est la différence entre WordPress.com et WordPress.org?",
            "Comment intégrer un formulaire de contact dans WordPress?",
            "Quelles sont les meilleures pratiques SEO pour WordPress?",
            "Comment configurer un système multisite avec WordPress?",
            "Quels sont les langages de programmation utilisés dans WordPress?",
            "Comment protéger un site WordPress contre les attaques?",
            "Quelles sont les tendances actuelles du web design?"
        ]
        
        # Exemples de contextes mémoire
        memory_contexts = [
            "L'utilisateur a précédemment demandé des informations sur WordPress. Il a mentionné qu'il était débutant et cherchait à créer un site pour son entreprise de photographie.",
            "L'utilisateur a déjà travaillé avec Elementor et a eu des problèmes avec les widgets personnalisés. Il utilise WordPress depuis 2 ans et a une expérience intermédiaire.",
            "L'utilisateur a exprimé son intérêt pour les systèmes de mémoire hiérarchique et comment ils améliorent les capacités de rappel des IA.",
            "L'utilisateur a créé plusieurs sites WordPress pour des clients et cherche maintenant à optimiser ses processus de développement.",
            "Précédemment, l'utilisateur a mentionné avoir des problèmes de sécurité sur son site WordPress après une mise à jour."
        ]
        
        # Exemples de contextes RAG
        rag_contexts = [
            "WordPress a été créé en 2003 par Matt Mullenweg et Mike Little comme une fork de b2/cafelog. Il est maintenant utilisé par plus de 40% des sites web dans le monde.",
            "Elementor est un page builder visuel pour WordPress lancé en 2016. Ses principales fonctionnalités incluent le drag-and-drop, le design responsive, et une bibliothèque de widgets.",
            "Le système de mémoire hiérarchique utilisé dans les IA modernes comporte généralement plusieurs niveaux : mémoire à court terme pour les interactions récentes, mémoire consolidée pour les concepts importants, et mémoire à long terme pour les connaissances durables.",
            "WordPress.org est la version auto-hébergée et open-source, tandis que WordPress.com est un service commercial hébergé. WordPress.org offre plus de flexibilité mais nécessite plus de gestion technique.",
            "Pour protéger un site WordPress, il est recommandé d'utiliser un plugin de sécurité comme Wordfence ou Sucuri, de maintenir tous les plugins à jour, d'utiliser des mots de passe forts, et d'activer l'authentification à deux facteurs."
        ]
        
        all_examples = []
        for i in range(num_examples):
            # Sélectionner aléatoirement une question et des contextes
            question = random.choice(questions)
            memory_context = random.choice(memory_contexts)
            rag_context = random.choice(rag_contexts)
            
            logger.info(f"Exemple {i+1}/{num_examples}")
            example = self.run_training_example(question, memory_context, rag_context)
            all_examples.append(example)
            
            # Pause pour éviter de surcharger l'API
            if i < num_examples - 1:
                time.sleep(2)
        
        # Afficher les résultats
        score_improvements = [ex["score_difference"] for ex in all_examples]
        avg_improvement = sum(score_improvements) / len(score_improvements) if score_improvements else 0
        
        logger.info(f"Session terminée - Amélioration moyenne des scores: {avg_improvement:.2f} points")
        logger.info(f"Les exemples ont été sauvegardés dans {self.output_file}")
        
        return all_examples

def main():
    parser = argparse.ArgumentParser(description='RLHF Trainer pour DeepSeek avec OpenAI comme critique')
    parser.add_argument('--openai_key', type=str, default=DEFAULT_OPENAI_KEY, help='Clé API OpenAI')
    parser.add_argument('--ollama_url', type=str, default=DEFAULT_OLLAMA_URL, help='URL du serveur Ollama')
    parser.add_argument('--deepseek_model', type=str, default=DEFAULT_DEEPSEEK_MODEL, help='Nom du modèle DeepSeek dans Ollama')
    parser.add_argument('--critic_model', type=str, default=DEFAULT_CRITIC_MODEL, help='Nom du modèle critique OpenAI')
    parser.add_argument('--output_file', type=str, default=DEFAULT_OUTPUT_FILE, help='Fichier de sortie pour les données RLHF')
    parser.add_argument('--num_examples', type=int, default=5, help='Nombre d\'exemples à générer')
    
    # Version compatibles avec Colab et Jupyter qui passent des arguments supplémentaires
    import sys
    
    # Filtrer les arguments qui commencent par '-f' (pour Jupyter/Colab)
    filtered_args = []
    skip_next = False
    for i, arg in enumerate(sys.argv[1:], 1):
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("-f"):
            skip_next = True
            continue
        filtered_args.append(arg)
    
    # Remplacer les arguments filtrés
    sys.argv = [sys.argv[0]] + filtered_args
    
    args = parser.parse_args()
    
    # Initialiser le trainer
    trainer = RLHFTrainer(
        openai_key=args.openai_key,
        ollama_url=args.ollama_url,
        deepseek_model=args.deepseek_model,
        critic_model=args.critic_model,
        output_file=args.output_file
    )
    
    # Exécuter la session d'entraînement
    trainer.run_training_session(num_examples=args.num_examples)

if __name__ == "__main__":
    main()
